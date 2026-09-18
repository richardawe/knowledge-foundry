"""The evaluation runner (§11, §12).

Runs the knowledge area's suites against the current build, grades every
answer, computes the metric panel and applies the publish gate.

Two design points worth defending:

**Answers are recorded, not just scores.** A failing question with no record of
what the system actually said is a failure you cannot diagnose, and the whole
point of §12's PASS/FAIL diagram is that FAIL leads to "investigate", not to
shrugging.

**Retrieval is measured separately from answering.** When a question fails, the
first thing anyone needs to know is whether the evidence was even there. That
is one metric apart, not an afternoon of debugging.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from ..manifest import Manifest
from ..specialist import Specialist
from ..specialist.answer import Answer
from ..storage import Store, utcnow
from ..text import numbers_with_units, quoted_spans
from .dataset import Question, Suite, all_questions, load_suites
from .graders import GradeResult, grade
from .metrics import METRICS, mean


@dataclass
class QuestionResult:
    question: Question
    answer: Answer
    grades: list[GradeResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.grades)

    @property
    def score(self) -> float:
        return mean([g.score for g in self.grades])

    def as_dict(self) -> dict:
        return {
            "id": self.question.id,
            "type": self.question.type,
            "suite": self.question.suite,
            "origin": self.question.origin,
            "question": self.question.question,
            "passed": self.passed,
            "score": round(self.score, 3),
            "status": self.answer.status,
            "grades": [g.as_dict() for g in self.grades],
            "answer": self.answer.answer,
            "cited_sources": sorted(self.answer.cited_source_ids),
            "citation_validity": round(self.answer.citation_validity, 3),
            "grounded_ratio": round(self.answer.grounded_ratio, 3),
        }


@dataclass
class EvaluationReport:
    run_id: str
    knowledge_area: str
    version: str
    started_at: str
    finished_at: str = ""
    suites: list[str] = field(default_factory=list)
    results: list[QuestionResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    threshold_failures: list[str] = field(default_factory=list)
    passed: bool = False

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passes(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failures(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.passed]

    def by_type(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for result in self.results:
            bucket = out.setdefault(result.question.type, {"total": 0, "passed": 0})
            bucket["total"] += 1
            bucket["passed"] += int(result.passed)
        for bucket in out.values():
            bucket["rate"] = round(bucket["passed"] / bucket["total"], 3)
        return out

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "knowledge_area": self.knowledge_area,
            "version": self.version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "suites": self.suites,
            "total": self.total,
            "passed_count": self.passes,
            "pass_rate": round(self.passes / self.total, 3) if self.total else 0.0,
            "by_type": self.by_type(),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "threshold_failures": self.threshold_failures,
            "passed": self.passed,
            "results": [r.as_dict() for r in self.results],
        }

    def render(self) -> str:
        lines = [
            f"evaluation {self.knowledge_area} {self.version} [{self.run_id}]",
            f"  questions: {self.passes}/{self.total} passed "
            f"({(self.passes / self.total * 100 if self.total else 0):.0f}%)",
            "  by type:",
        ]
        for qtype, bucket in sorted(self.by_type().items()):
            lines.append(f"    {qtype:<16} {bucket['passed']}/{bucket['total']}")
        lines.append("  metrics:")
        for name, value in sorted(self.metrics.items()):
            spec = METRICS.get(name)
            arrow = "" if spec is None else (" (lower is better)" if spec.direction == "lower" else "")
            lines.append(f"    {name:<24} {value:.3f}{arrow}")
        if self.threshold_failures:
            lines.append("  THRESHOLD FAILURES:")
            for failure in self.threshold_failures:
                lines.append(f"    {failure}")
        if self.failures:
            lines.append(f"  failing questions ({len(self.failures)}):")
            for result in self.failures[:25]:
                reasons = "; ".join(g.detail for g in result.grades if not g.passed and g.detail)
                lines.append(f"    {result.question.id:<18} {result.question.type:<14} {reasons[:90]}")
            if len(self.failures) > 25:
                lines.append(f"    ... and {len(self.failures) - 25} more")
        lines.append(f"  VERDICT: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _hallucinated(answer: Answer) -> bool:
    """Did the answer state a number or quotation its evidence does not contain?

    Narrower than "unsupported": a sentence can be unsupported because it is
    connective prose, which is untidy. Stating a figure that is not in the
    cited evidence is the failure this domain cannot tolerate.
    """
    for claim in answer.unsupported_claims:
        sentence = claim.get("sentence", "")
        if numbers_with_units(sentence) or quoted_spans(sentence):
            reason = claim.get("reason", "")
            if "absent from the cited evidence" in reason or "quotes" in reason:
                return True
    return False


def compute_metrics(
    results: list[QuestionResult], store: Store, baseline: dict | None = None
) -> dict[str, float]:
    """Build the metric panel (§12). No composite score, by design."""
    if not results:
        return {}

    answered = [r for r in results if not r.question.should_abstain]
    with_expected = [r for r in results if r.question.expected_sources]

    # Retrieval, measured independently of answering.
    precisions, recalls = [], []
    for result in with_expected:
        expected = set(result.question.expected_sources)
        retrieved = {s["source_id"] for s in result.answer.sources}
        if retrieved:
            precisions.append(len(expected & retrieved) / len(retrieved))
        recalls.append(len(expected & retrieved) / len(expected))

    # Citation relevance: of the sources actually cited, how many were expected?
    relevances = []
    for result in with_expected:
        cited = result.answer.cited_source_ids
        if cited:
            relevances.append(len(set(result.question.expected_sources) & cited) / len(cited))

    # Abstention: did it make the right answer/decline decision?
    #
    # Measured only where there IS a right answer. Declining a question with a
    # false premise is correct, and so is answering it with the premise
    # corrected -- so scoring `adversarial` and `ambiguous` questions here would
    # move the metric for reasons that say nothing about the system's judgement.
    # Those types are graded by `must_flag_uncertainty` and `must_not_include`
    # instead, which test what actually matters about them.
    decisive = [
        r for r in results
        if r.question.type not in ("adversarial", "ambiguous")
        or any(g.get("kind") in ("must_abstain", "must_answer") for g in r.question.graders)
    ]
    abstention_correct = [
        1.0 if result.answer.abstained == result.question.should_abstain else 0.0
        for result in decisive
    ]

    cited_anywhere: set[str] = set()
    for result in results:
        cited_anywhere |= result.answer.cited_source_ids
    indexed = store.conn.execute(
        "SELECT COUNT(*) AS n FROM sources WHERE status = 'indexed'"
    ).fetchone()["n"]

    metrics = {
        "retrieval_precision": mean(precisions),
        "retrieval_recall": mean(recalls),
        "answer_correctness": mean([r.score for r in answered]) if answered else 0.0,
        "citation_validity": mean([r.answer.citation_validity for r in results]),
        "citation_relevance": mean(relevances),
        "unsupported_claim_rate": mean([1.0 - r.answer.grounded_ratio for r in results]),
        "hallucination_rate": mean([1.0 if _hallucinated(r.answer) else 0.0 for r in results]),
        "abstention_correctness": mean(abstention_correct),
        "source_coverage": (len(cited_anywhere) / indexed) if indexed else 0.0,
        "contradiction_rate": mean([1.0 if r.answer.contradictions else 0.0 for r in results]),
    }
    freshness = store.freshness_days()
    if freshness is not None:
        metrics["corpus_freshness_days"] = freshness

    # Regression rate is only meaningful against a baseline.
    if baseline:
        previously_passing = {
            row["id"] for row in baseline.get("results", []) if row.get("passed")
        }
        if previously_passing:
            now_failing = {
                r.question.id for r in results
                if not r.passed and r.question.id in previously_passing
            }
            metrics["regression_rate"] = len(now_failing) / len(previously_passing)
        else:
            metrics["regression_rate"] = 0.0
    return metrics


def run_evaluation(
    manifest: Manifest,
    store: Store,
    suites: list[str] | None = None,
    limit: int | None = None,
    specialist: Specialist | None = None,
    baseline: dict | None = None,
    judge_provider=None,
    persist: bool = True,
) -> EvaluationReport:
    """Run the suites and apply the publish gate."""
    loaded: list[Suite] = load_suites(manifest, only=suites)
    questions = all_questions(loaded)
    if limit:
        questions = questions[:limit]

    version = store.get_meta("current_version", "unversioned")
    started = utcnow()
    run_id = "run_" + hashlib.sha256(f"{manifest.id}{started}{version}".encode()).hexdigest()[:16]
    report = EvaluationReport(
        run_id=run_id,
        knowledge_area=manifest.id,
        version=version,
        started_at=started,
        suites=[s.name for s in loaded],
    )

    specialist = specialist or Specialist(manifest, store)
    for question in questions:
        answer = specialist.ask(question.question)
        grades = []
        for spec in question.graders:
            spec = dict(spec)
            if spec.get("kind") == "judge":
                spec["_provider"] = judge_provider
            grades.append(grade(spec, answer))
        report.results.append(QuestionResult(question=question, answer=answer, grades=grades))

    if baseline is None:
        baseline = load_baseline(manifest)
    report.metrics = compute_metrics(report.results, store, baseline=baseline)
    report.finished_at = utcnow()

    # Absolute thresholds from the manifest.
    for name, threshold in manifest.evaluation.thresholds.items():
        if name not in report.metrics:
            continue
        spec = METRICS.get(name)
        value = report.metrics[name]
        if spec and spec.direction == "lower":
            if value > threshold:
                report.threshold_failures.append(f"{name} {value:.3f} > {threshold}")
        elif value < threshold:
            report.threshold_failures.append(f"{name} {value:.3f} < {threshold}")
    report.passed = not report.threshold_failures

    if persist:
        _persist(report, store)
        write_report(manifest, report)
    return report


def _persist(report: EvaluationReport, store: Store) -> None:
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO eval_runs"
            "(id, version, suite, started_at, finished_at, metrics, passed, kind)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report.run_id, report.version, ",".join(report.suites),
                report.started_at, report.finished_at,
                json.dumps(report.metrics), int(report.passed), "evaluation",
            ),
        )
        for result in report.results:
            conn.execute(
                "INSERT OR REPLACE INTO eval_results"
                "(run_id, question_id, question, qtype, passed, scores, answer, notes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.run_id, result.question.id, result.question.question,
                    result.question.type, int(result.passed),
                    json.dumps([g.as_dict() for g in result.grades]),
                    json.dumps(result.answer.as_dict(), default=str),
                    result.question.notes,
                ),
            )
    store.set_meta("evaluation_status", "passed" if report.passed else "failed")
    store.set_meta("last_evaluation", report.finished_at)
    store.set_meta("last_eval_run", report.run_id)


def report_path(manifest: Manifest, run_id: str) -> Path:
    return paths.reports_dir(manifest.id) / f"{run_id}.json"


def write_report(manifest: Manifest, report: EvaluationReport) -> Path:
    path = report_path(manifest, report.run_id)
    path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    latest = paths.reports_dir(manifest.id) / "latest.json"
    latest.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_baseline(manifest: Manifest) -> dict | None:
    """The published version's report, which the regression gate compares against."""
    path = paths.ka_var_dir(manifest.id) / "baseline.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_baseline(manifest: Manifest, report: EvaluationReport) -> Path:
    path = paths.ka_var_dir(manifest.id) / "baseline.json"
    path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return path
