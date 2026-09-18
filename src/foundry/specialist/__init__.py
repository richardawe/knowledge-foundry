"""The specialist agent: retrieve, reason, cite, validate, abstain (§9, §10)."""

from .agent import Specialist
from .answer import (
    ABSTENTION_STATUSES,
    ANSWERED,
    ERROR,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_SCOPE,
    UNSUPPORTED,
    Answer,
)
from .validation import ValidationResult, detect_contradictions, validate_answer

__all__ = [
    "ABSTENTION_STATUSES",
    "ANSWERED",
    "Answer",
    "ERROR",
    "INSUFFICIENT_EVIDENCE",
    "OUT_OF_SCOPE",
    "Specialist",
    "UNSUPPORTED",
    "ValidationResult",
    "detect_contradictions",
    "validate_answer",
]
