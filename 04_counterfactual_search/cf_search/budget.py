"""Per-query, per-method limits (model costs are sequence counts)."""
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class SearchBudget:
    model_evaluations: int = 500
    candidate_attempts: int = 10000
    walltime_seconds: float = 60.

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.model_evaluations, self.candidate_attempts)):
            raise ValueError("Budgets must be positive integers")
        if not math.isfinite(self.walltime_seconds) or self.walltime_seconds <= 0:
            raise ValueError("Walltime must be finite and positive")


class BudgetExecutor:
    def __init__(self, budget):
        self.budget = budget
        self.started = time.monotonic()
        self.attempts = 0
        self.status = "completed"

    def check(self):
        if time.monotonic() - self.started >= self.budget.walltime_seconds:
            self.status = "walltime_exhausted"
            return False
        return True

    def visit(self):
        if not self.check():
            return False
        if self.attempts >= self.budget.candidate_attempts:
            self.status = "attempt_budget_exhausted"
            return False
        self.attempts += 1
        return True

    @property
    def elapsed(self):
        return time.monotonic() - self.started
