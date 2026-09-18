"""Automated adversarial evaluation (§13)."""

from .generators import EXPECT_CAUTION, STRATEGIES, Attack, LLMGenerator, TemplateGenerator
from .judge import FAILURE_MODES, Verdict, judge
from .runner import REGRESSION_SUITE, RedTeamReport, promote_failures, run_redteam

__all__ = [
    "Attack",
    "EXPECT_CAUTION",
    "FAILURE_MODES",
    "LLMGenerator",
    "REGRESSION_SUITE",
    "RedTeamReport",
    "STRATEGIES",
    "TemplateGenerator",
    "Verdict",
    "judge",
    "promote_failures",
    "run_redteam",
]
