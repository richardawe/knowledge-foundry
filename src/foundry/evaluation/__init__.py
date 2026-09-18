"""Evaluation: a product feature, not an afterthought (§11, §12)."""

from .dataset import ABSTAIN_TYPES, QUESTION_TYPES, Question, Suite, load_suite, load_suites
from .graders import GradeResult, available_graders, grade
from .metrics import GATED_METRICS, METRICS, is_regression
from .runner import (
    EvaluationReport,
    QuestionResult,
    compute_metrics,
    load_baseline,
    run_evaluation,
    save_baseline,
)

__all__ = [
    "ABSTAIN_TYPES",
    "EvaluationReport",
    "GATED_METRICS",
    "GradeResult",
    "METRICS",
    "QUESTION_TYPES",
    "Question",
    "QuestionResult",
    "Suite",
    "available_graders",
    "compute_metrics",
    "grade",
    "is_regression",
    "load_baseline",
    "load_suite",
    "load_suites",
    "run_evaluation",
    "save_baseline",
]
