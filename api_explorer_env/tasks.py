"""
Tasks the agent must complete. Each task has a verifier that scores 0.0-1.0.
Difficulty: easy < medium < hard.
"""
from __future__ import annotations
from typing import Any


class Task:
    def __init__(self, name: str, description: str, difficulty: str):
        self.name = name
        self.description = description
        self.difficulty = difficulty
        self.completed = False
        self.score: float = 0.0

    def check(self, answer: Any, api_db: dict, calls_used: int) -> float:
        raise NotImplementedError

    def verify(self, answer: Any, api_db: dict, calls_used: int) -> float:
        self.score = round(max(0.0, min(1.0, self.check(answer, api_db, calls_used))), 4)
        if self.score >= 1.0:
            self.completed = True
        return self.score


# ── Easy tasks ─────────────────────────────────────────────────────────────────

class CountMerchantsTask(Task):
    def __init__(self):
        super().__init__(
            "easy_count_merchants",
            "How many merchants are registered in total?",
            "easy",
        )

    def check(self, answer: Any, api_db: dict, calls_used: int) -> float:
        correct = len(api_db["merchants"])
        try:
            if int(answer) == correct:
                return 1.0
        except (TypeError, ValueError):
            pass
        return 0.0


class FindMerchantStatusTask(Task):
    def __init__(self, merchant_id: str):
        self.merchant_id = merchant_id
        super().__init__(
            "find_merchant_status",
            f"What is the status of merchant {merchant_id}?",
            "easy",
        )

    def check(self, answer: Any, api_db: dict, calls_used: int) -> float:
        merchant = next((m for m in api_db["merchants"] if m["id"] == self.merchant_id), None)
        if not merchant:
            return 0.0
        return 1.0 if str(answer).lower() == merchant["status"] else 0.0


# ── Medium tasks ───────────────────────────────────────────────────────────────

class CountFailedTransactionsTask(Task):
    def __init__(self):
        super().__init__(
            "medium_count_failed_transactions",
            "How many transactions have status 'failed'?",
            "medium",
        )

    def check(self, answer: Any, api_db: dict, calls_used: int) -> float:
        correct = sum(1 for t in api_db["transactions"] if t["status"] == "failed")
        try:
            if int(answer) == correct:
                # Bonus: fewer calls = higher score
                efficiency = max(0.5, 1.0 - (calls_used - 2) * 0.05)
                return min(1.0, efficiency)
        except (TypeError, ValueError):
            pass
        return 0.0


class TotalVolumeForMerchantTask(Task):
    def __init__(self, merchant_id: str):
        self.merchant_id = merchant_id
        super().__init__(
            "merchant_total_volume",
            f"What is the total transaction volume (sum of amounts) for merchant {merchant_id}?",
            "medium",
        )

    def check(self, answer: Any, api_db: dict, calls_used: int) -> float:
        txns = [t for t in api_db["transactions"] if t["merchant_id"] == self.merchant_id]
        correct = round(sum(t["amount"] for t in txns), 2)
        try:
            if abs(float(answer) - correct) < 0.1:
                efficiency = max(0.5, 1.0 - (calls_used - 3) * 0.05)
                return min(1.0, efficiency)
        except (TypeError, ValueError):
            pass
        return 0.0


# ── Medium-easy bridge ────────────────────────────────────────────────────────

class TotalVolumeAllTask(Task):
    """Requires calling /summary — tests route discovery."""
    def __init__(self):
        super().__init__(
            "total_transaction_volume",
            "What is the total transaction volume (sum of all amounts) across all merchants? Round to 2 decimal places.",
            "easy",
        )

    def check(self, answer: Any, api_db: dict, calls_used: int) -> float:
        correct = round(sum(t["amount"] for t in api_db["transactions"]), 2)
        try:
            if abs(float(answer) - correct) < 0.5:
                return 1.0
        except (TypeError, ValueError):
            pass
        return 0.0


# ── Hard tasks ─────────────────────────────────────────────────────────────────

class ResolveOldestDisputeTask(Task):
    def __init__(self):
        super().__init__(
            "hard_resolve_oldest_dispute",
            "Find the first open dispute and resolve it. Return the dispute ID you resolved.",
            "hard",
        )

    def check(self, answer: Any, api_db: dict, calls_used: int) -> float:
        open_disputes = [d for d in api_db["disputes"] if d["status"] in ("open", "resolved")]
        if not open_disputes:
            return 0.0
        target_id = open_disputes[0]["id"]
        resolved = next((d for d in api_db["disputes"] if d["id"] == target_id), None)
        if resolved and resolved["status"] == "resolved" and str(answer) == target_id:
            efficiency = max(0.5, 1.0 - (calls_used - 4) * 0.04)
            return min(1.0, efficiency)
        return 0.0


class FindHighestBalanceMerchantTask(Task):
    def __init__(self):
        super().__init__(
            "find_richest_merchant",
            "Which merchant has the highest balance? Return their merchant ID.",
            "hard",
        )

    def check(self, answer: Any, api_db: dict, calls_used: int) -> float:
        if not api_db["merchants"]:
            return 0.0
        richest = max(api_db["merchants"], key=lambda m: m["balance"])
        if str(answer) == richest["id"]:
            # Must paginate all pages to truly know the richest — penalize if too few calls
            efficiency = max(0.5, 1.0 - (calls_used - 5) * 0.03)
            return min(1.0, efficiency)
        return 0.0


# ── Task sets per scenario ─────────────────────────────────────────────────────

def get_tasks(scenario: str, db: dict) -> list[Task]:
    # All scenarios use the same 3 canonical graders (easy / medium / hard).
    # The scenario only affects the API call limit in the environment.
    return [
        CountMerchantsTask(),           # easy_count_merchants
        CountFailedTransactionsTask(),  # medium_count_failed_transactions
        ResolveOldestDisputeTask(),     # hard_resolve_oldest_dispute
    ]
