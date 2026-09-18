"""Evaluation question sets (§11).

Question sets are YAML inside the knowledge area, because they are part of the
knowledge asset -- an evaluation suite is a domain expert's opinion about what
the system must get right, and it belongs next to the sources, not in the code.

Nine question types, each with a different notion of success:

``factual``        one checkable fact from one source
``multi_document`` requires combining passages from different sources
``reasoning``      requires a step the sources do not state directly
``ambiguous``      under-specified; success is asking or qualifying, not answering
``unanswerable``   the corpus cannot support it; success is abstaining
``out_of_scope``   outside the declared domain; success is refusing
``citation``       tests whether the right source is cited, not just the right words
``temporal``       tests currency and version awareness
``adversarial``    false premises and misleading framing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..manifest import Manifest, ManifestError

QUESTION_TYPES = (
    "factual",
    "multi_document",
    "reasoning",
    "ambiguous",
    "unanswerable",
    "out_of_scope",
    "citation",
    "temporal",
    "adversarial",
)

# Types where the correct behaviour is to decline. Used by the abstention
# metric and, more importantly, to stop a suite rewarding confident answers to
# questions that have none.
ABSTAIN_TYPES = frozenset({"unanswerable", "out_of_scope"})


@dataclass
class Question:
    id: str
    question: str
    type: str = "factual"
    graders: list[dict] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    notes: str = ""
    origin: str = "curated"     # curated | redteam | challenge
    suite: str = ""

    @property
    def should_abstain(self) -> bool:
        if self.type in ABSTAIN_TYPES:
            return True
        return any(g.get("kind") == "must_abstain" for g in self.graders)


@dataclass
class Suite:
    name: str
    path: Path | None
    description: str = ""
    questions: list[Question] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)


def parse_suite(data: dict, path: Path | None = None) -> Suite:
    name = data.get("suite") or (path.stem if path else "unnamed")
    suite = Suite(name=name, path=path, description=data.get("description", ""))
    seen: set[str] = set()

    for index, raw in enumerate(data.get("questions") or []):
        if not isinstance(raw, dict):
            raise ManifestError(f"{name}: question {index} is not a mapping")
        for key in ("id", "question"):
            if not raw.get(key):
                raise ManifestError(f"{name}: question {index} is missing '{key}'")
        qtype = raw.get("type", "factual")
        if qtype not in QUESTION_TYPES:
            raise ManifestError(
                f"{name}: question {raw['id']} has unknown type {qtype!r}; "
                f"expected one of {QUESTION_TYPES}"
            )
        if raw["id"] in seen:
            raise ManifestError(f"{name}: duplicate question id {raw['id']!r}")
        seen.add(raw["id"])

        graders = list(raw.get("graders") or [])
        # A question with no grader asserts nothing and would silently inflate
        # every pass rate in the suite.
        if not graders:
            raise ManifestError(
                f"{name}: question {raw['id']} has no graders, so it cannot pass or fail"
            )
        suite.questions.append(
            Question(
                id=raw["id"],
                question=raw["question"],
                type=qtype,
                graders=graders,
                expected_sources=list(raw.get("expected_sources") or []),
                notes=raw.get("notes", ""),
                origin=raw.get("origin", "curated"),
                suite=name,
            )
        )
    return suite


def load_suite(path: Path) -> Suite:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return parse_suite(data, Path(path))


def load_suites(manifest: Manifest, only: list[str] | None = None) -> list[Suite]:
    """Load the manifest's declared suites, skipping ones not yet written.

    A missing suite is not an error: ``regression_from_redteam.yaml`` does not
    exist until the red team has found something, and a knowledge area should
    not fail its own evaluation for not having failed anything yet.
    """
    wanted = only or manifest.evaluation.suites
    suites: list[Suite] = []
    for entry in wanted:
        path = Path(entry)
        if not path.is_absolute() and not path.exists():
            path = manifest.resolve(entry)
        if not path.is_file():
            continue
        suites.append(load_suite(path))
    return suites


def all_questions(suites: list[Suite]) -> list[Question]:
    return [q for suite in suites for q in suite.questions]
