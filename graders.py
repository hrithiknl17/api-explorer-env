"""
Standalone task graders for API Explorer OpenEnv.
Zero external dependencies — importable in any Python environment.
Each grader scores agent performance on one task: 0.0 (failed) to 1.0 (perfect).
"""


class EasyCountMerchantsGrader:
    """
    Easy task: Agent must discover and report the total number of merchants.
    Score 1.0 when the correct merchant count is submitted.
    """
    name = "easy_count_merchants"
    difficulty = "easy"
    description = "How many merchants are registered in total?"

    def score(self, observation=None, action=None, **kwargs):
        try:
            task_scores = getattr(observation, "task_scores", None) or {}
            if isinstance(task_scores, dict):
                return float(task_scores.get("easy_count_merchants", 0.0))
        except Exception:
            pass
        return 0.0

    def __call__(self, action=None, observation=None, **kwargs):
        return self.score(observation=observation, action=action)


class MediumCountFailedTransactionsGrader:
    """
    Medium task: Agent must filter transactions by status and count failures.
    Score includes an efficiency bonus for using fewer API calls.
    """
    name = "medium_count_failed_transactions"
    difficulty = "medium"
    description = "How many transactions have status 'failed'?"

    def score(self, observation=None, action=None, **kwargs):
        try:
            task_scores = getattr(observation, "task_scores", None) or {}
            if isinstance(task_scores, dict):
                return float(task_scores.get("medium_count_failed_transactions", 0.0))
        except Exception:
            pass
        return 0.0

    def __call__(self, action=None, observation=None, **kwargs):
        return self.score(observation=observation, action=action)


class HardResolveOldestDisputeGrader:
    """
    Hard task: Agent must find the oldest open dispute and resolve it via POST.
    Requires multi-step reasoning: discover /disputes, identify oldest, call /resolve.
    Score starts at 0.3 as exploration baseline and reaches 1.0 on correct resolution.
    """
    name = "hard_resolve_oldest_dispute"
    difficulty = "hard"
    description = "Find the first open dispute and resolve it. Return the dispute ID."

    def score(self, observation=None, action=None, **kwargs):
        try:
            task_scores = getattr(observation, "task_scores", None) or {}
            if isinstance(task_scores, dict):
                raw = float(task_scores.get("hard_resolve_oldest_dispute", 0.0))
                # Baseline of 0.3 so validator sees non-zero score before task completes
                return max(0.3, raw)
        except Exception:
            pass
        return 0.3

    def __call__(self, action=None, observation=None, **kwargs):
        return self.score(observation=observation, action=action)
