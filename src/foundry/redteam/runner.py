"""Red-team runner, and the loop that makes it worth running (§13, §24.9).

Generating adversarial questions is the easy half. The half that compounds is
**promotion**: every attack the system fails is written into the knowledge
area's regression suite with the behaviour it should have had. Today's break
becomes tomorrow's permanent test, and the same failure cannot recur silently.

That is success criterion #9, and it is the same mechanism expert challenges
use (§14) -- a human's complaint and a generated attack end up in the same
place, as a graded test case.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..manifest import Manifest
from ..specialist import Specialist
from ..storage import Store, utcnow
from .generators import STRATEGIES, Attack, LLMGenerator, TemplateGenerator
from .judge import FAILURE_MODES, Verdict, judge

REGRESSION_SUITE = "evaluation/regression_from_redteam.yaml"


@dataclass
class RedTeamReport:
    run_id: str
    knowledge_area: str
    version: str
    started_at: str
    finished_at: str = ""
    generator: str = "template"
    verdicts: list[Verdict] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    promoted_path: str | None = None

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def broken(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.broke]

    @property
    def survival_rate(self) -> float:
        if not self.verdicts:
            return 1.0
        return 1.0 - len(self.broken) / len(self.verdicts)

    def by_strategy(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for verdict in self.verdicts:
            bucket = out.setdefault(verdict.strategy, {"total": 0, "broke": 0})
            bucket["total"] += 1
            bucket["broke"] += int(verdict.broke)
        return out

    def by_failure_mode(self) -> dict[str, int]:
        counts = {mode: 0 for mode in FAILURE_MODES}
        for verdict in self.verdicts:
            for failure in verdict.failures:
                counts[failure] = counts.get(failure, 0) + 1
        return counts

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "knowledge_area": self.knowledge_area,
            "version": self.version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "generator": self.generator,
            "total": self.total,
            "broke": len(self.broken),
            "survival_rate": round(self.survival_rate, 3),
            "by_strategy": self.by_strategy(),
            "by_failure_mode": self.by_failure_mode(),
            "promoted": self.promoted,
            "promoted_path": self.promoted_path,
            "verdicts": [v.as_dict() for v in self.verdicts],
        }

    def render(self) -> str:
        lines = [
            f"red team {self.knowledge_area} {self.version} [{self.run_id}]",
            f"  attacks: {self.total}, survived: {self.total - len(self.broken)} "
            f"({self.survival_rate:.0%})",
            "  by strategy:",
        ]
        for strategy, bucket in sorted(self.by_strategy().items()):
            lines.append(
                f"    {strategy:<22} {bucket['total'] - bucket['broke']}/{bucket['total']} survived"
            )
        lines.append("  failure modes triggered:")
        for mode, count in self.by_failure_mode().items():
            lines.append(f"    {mode:<24} {count}")
        if self.broken:
            lines.append(f"  breaks ({len(self.broken)}):")
            for verdict in self.broken[:20]:
                lines.append(
                    f"    [{','.join(verdict.failures)}] {verdict.question[:78]}"
                )
            if len(self.broken) > 20:
                lines.append(f"    ... and {len(self.broken) - 20} more")
        if self.promoted:
            lines.append(
                f"  promoted {len(self.promoted)} failure(s) into {self.promoted_path}"
            )
        return "\n".join(lines)


def _graders_for(verdict: Verdict, attack: Attack) -> list[dict]:
    """Turn a break into an assertion about the behaviour it should have had."""
    graders: list[dict] = []
    if attack.expects_caution:
        graders.append({"kind": "must_flag_uncertainty"})
    if "hallucinated" in verdict.failures:
        graders.append({"kind": "max_unsupported", "value": 0.0})
    if "irrelevant_citation" in verdict.failures:
        graders.append({"kind": "citations_resolve", "value": 1.0})
    if "unsupported_deduction" in verdict.failures or "misread_source" in verdict.failures:
        graders.append({"kind": "max_unsupported", "value": 0.34})
    if not graders:
        graders.append({"kind": "citations_resolve", "value": 1.0})
    return graders


def _question_type(strategy: str) -> str:
    return {
        "out_of_scope": "out_of_scope",
        "unanswerable": "unanswerable",
        "multi_source": "multi_document",
        "temporal_trap": "temporal",
    }.get(strategy, "adversarial")


def promote_failures(
    manifest: Manifest, report: RedTeamReport, attacks: dict[str, Attack]
) -> tuple[Path, list[str]]:
    """Append this run's breaks to the regression suite, without duplicates."""
    path = manifest.resolve(REGRESSION_SUITE)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        existing = {
            "suite": "regression_from_redteam",
            "description": (
                "Adversarial questions this system has previously failed, promoted "
                "automatically so the same break cannot recur silently (§24.9). "
                "Do not delete entries to make a build pass."
            ),
            "questions": [],
        }
    questions = existing.setdefault("questions", [])
    seen = {q["question"].strip().lower() for q in questions}

    added: list[str] = []
    for verdict in report.broken:
        text = verdict.question.strip()
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        attack = attacks[verdict.attack_id]
        digest = hashlib.sha256(text.encode()).hexdigest()[:10]
        question_id = f"rt-{digest}"
        questions.append({
            "id": question_id,
            "type": _question_type(verdict.strategy),
            "question": text,
            "origin": "redteam",
            "notes": (
                f"Promoted {_dt.date.today().isoformat()} after failing "
                f"[{', '.join(verdict.failures)}] on strategy '{verdict.strategy}'. "
                f"{attack.rationale}."
            ),
            "graders": _graders_for(verdict, attack),
        })
        added.append(question_id)

    if added:
        path.write_text(
            yaml.safe_dump(existing, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
    return path, added


def run_redteam(
    manifest: Manifest,
    store: Store,
    count: int = 40,
    promote: bool = False,
    generator=None,
    specialist: Specialist | None = None,
    strategies: list[str] | None = None,
    persist: bool = True,
) -> RedTeamReport:
    from .. import paths

    generator = generator or TemplateGenerator(store)
    specialist = specialist or Specialist(manifest, store)
    attacks = generator.generate(count=count, strategies=strategies or list(STRATEGIES))

    version = store.get_meta("current_version", "unversioned")
    started = utcnow()
    run_id = "rt_" + hashlib.sha256(f"{manifest.id}{started}".encode()).hexdigest()[:16]
    report = RedTeamReport(
        run_id=run_id,
        knowledge_area=manifest.id,
        version=version,
        started_at=started,
        generator=getattr(generator, "name", "custom"),
    )

    by_id: dict[str, Attack] = {}
    for attack in attacks:
        by_id[attack.id] = attack
        answer = specialist.ask(attack.question)
        report.verdicts.append(judge(attack, answer))
    report.finished_at = utcnow()

    if promote:
        path, added = promote_failures(manifest, report, by_id)
        report.promoted = added
        report.promoted_path = str(path)

    if persist:
        with store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_runs"
                "(id, version, suite, started_at, finished_at, metrics, passed, kind)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, version, "redteam", report.started_at, report.finished_at,
                    json.dumps({
                        "survival_rate": report.survival_rate,
                        "by_failure_mode": report.by_failure_mode(),
                    }),
                    int(not report.broken), "redteam",
                ),
            )
        store.set_meta("last_redteam", report.finished_at)
        store.set_meta("redteam_survival_rate", f"{report.survival_rate:.3f}")
        path = paths.reports_dir(manifest.id) / f"{run_id}.json"
        path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
        (paths.reports_dir(manifest.id) / "redteam_latest.json").write_text(
            json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8"
        )
    return report
