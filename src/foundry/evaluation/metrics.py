"""Metric definitions (§12).

The brief is explicit: *"Do not invent a single meaningless AI accuracy
number."* So there is no headline score. Every metric below is reported
separately, each says which direction is better, and each is gated
independently.

``direction`` drives the regression comparison: for ``higher`` metrics a drop
is a regression, for ``lower`` metrics a rise is. ``gated`` marks the metrics a
build must clear to publish; the rest are diagnostic, because an informative
number is not automatically a target.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str          # "higher" or "lower" is better
    gated: bool
    description: str


METRICS: dict[str, MetricSpec] = {
    m.name: m
    for m in [
        MetricSpec("retrieval_precision", "higher", False,
                   "Share of retrieved evidence coming from a question's expected sources."),
        MetricSpec("retrieval_recall", "higher", True,
                   "Share of expected sources that reached the evidence set."),
        MetricSpec("answer_correctness", "higher", True,
                   "Pass rate of content graders on questions that have an answer."),
        MetricSpec("citation_validity", "higher", True,
                   "Share of citation markers that resolve to retrieved evidence."),
        MetricSpec("citation_relevance", "higher", False,
                   "Share of cited sources that the question expected to be cited."),
        MetricSpec("unsupported_claim_rate", "lower", True,
                   "Share of answer sentences not traceable to their cited evidence."),
        MetricSpec("hallucination_rate", "lower", True,
                   "Share of answers stating a number or quotation absent from cited evidence."),
        MetricSpec("abstention_correctness", "higher", True,
                   "Accuracy of the decision to answer versus decline."),
        MetricSpec("source_coverage", "higher", False,
                   "Share of indexed sources cited at least once across the suite."),
        MetricSpec("corpus_freshness_days", "lower", False,
                   "Median age of source retrieval, in days."),
        MetricSpec("contradiction_rate", "lower", False,
                   "Share of answers surfacing disagreement between sources."),
        MetricSpec("regression_rate", "lower", True,
                   "Share of previously passing questions that now fail."),
    ]
}

GATED_METRICS = [name for name, spec in METRICS.items() if spec.gated]


def is_regression(metric: str, current: float, baseline: float, tolerance: float) -> bool:
    """Has ``metric`` moved in the bad direction by more than ``tolerance``?"""
    spec = METRICS.get(metric)
    if spec is None:
        return False
    if spec.direction == "higher":
        return current < baseline - tolerance
    return current > baseline + tolerance


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
