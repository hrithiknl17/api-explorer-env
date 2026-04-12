"""
Mock undocumented REST API — a fictional payment/merchant platform.
The agent doesn't know the routes. It must discover them by calling endpoints.
"""
from __future__ import annotations

import random
from typing import Any

# ── Data fixtures ──────────────────────────────────────────────────────────────

def _make_db(seed: int) -> dict:
    rng = random.Random(seed)

    merchants = [
        {"id": f"m_{i:03d}", "name": rng.choice(["Acme Corp", "Zeta Retail", "Nova Foods", "Peak Tech", "Blue Mart"]) + f" #{i}",
         "status": rng.choice(["active", "active", "active", "suspended"]),
         "plan": rng.choice(["starter", "growth", "enterprise"]),
         "balance": round(rng.uniform(100, 50000), 2)}
        for i in range(1, 21)
    ]

    transactions = [
        {"id": f"txn_{i:04d}",
         "merchant_id": rng.choice(merchants)["id"],
         "amount": round(rng.uniform(10, 5000), 2),
         "currency": rng.choice(["INR", "USD", "EUR"]),
         "status": rng.choice(["success", "success", "success", "failed", "pending"]),
         "created_at": f"2024-04-{rng.randint(1,12):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00Z"}
        for i in range(1, 101)
    ]

    disputes = [
        {"id": f"dsp_{i:03d}",
         "transaction_id": rng.choice(transactions)["id"],
         "reason": rng.choice(["item_not_received", "unauthorized_charge", "duplicate", "wrong_amount"]),
         "status": rng.choice(["open", "open", "resolved", "escalated"]),
         "amount": round(rng.uniform(10, 2000), 2)}
        for i in range(1, 16)
    ]

    return {"merchants": merchants, "transactions": transactions, "disputes": disputes}


# ── Route handler ──────────────────────────────────────────────────────────────

class MockAPI:
    """
    Undocumented REST API. Routes are discovered by calling them.
    Returns (status_code, body_dict).
    """

    KNOWN_ROUTES = [
        "GET /merchants",
        "GET /merchants/{id}",
        "GET /merchants/{id}/transactions",
        "GET /transactions",
        "GET /transactions/{id}",
        "GET /disputes",
        "GET /disputes/{id}",
        "POST /disputes/{id}/resolve",
        "GET /summary",
    ]

    def __init__(self, seed: int = 42):
        self.db = _make_db(seed)
        self._calls: int = 0
        self._auth_token: str | None = None

    def call(self, method: str, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, Any]:
        self._calls += 1
        method = method.upper()
        path = path.strip("/")
        parts = path.split("/")

        # ── auth check ─────────────────────────────────────────────────────────
        token = (headers or {}).get("Authorization", "").replace("Bearer ", "")
        if not self._auth_token:
            # First call - seed auth not required, we issue a token on /auth
            pass
        if path == "auth" and method == "POST":
            self._auth_token = "tok_explorer_" + str(random.randint(1000, 9999))
            return 200, {"token": self._auth_token, "message": "Authenticated successfully."}

        # ── routing ────────────────────────────────────────────────────────────
        if parts[0] == "merchants":
            if len(parts) == 1 and method == "GET":
                page = int((body or {}).get("page", 1))
                per_page = 5
                items = self.db["merchants"]
                return 200, {
                    "data": items[(page-1)*per_page : page*per_page],
                    "page": page, "per_page": per_page, "total": len(items)
                }
            if len(parts) == 2 and method == "GET":
                m = next((x for x in self.db["merchants"] if x["id"] == parts[1]), None)
                return (200, m) if m else (404, {"error": "Merchant not found"})
            if len(parts) == 3 and parts[2] == "transactions" and method == "GET":
                txns = [t for t in self.db["transactions"] if t["merchant_id"] == parts[1]]
                return 200, {"data": txns, "total": len(txns)}

        if parts[0] == "transactions":
            if len(parts) == 1 and method == "GET":
                status_filter = (body or {}).get("status")
                items = self.db["transactions"]
                if status_filter:
                    items = [t for t in items if t["status"] == status_filter]
                page = int((body or {}).get("page", 1))
                per_page = 10
                return 200, {
                    "data": items[(page-1)*per_page : page*per_page],
                    "page": page, "per_page": per_page, "total": len(items)
                }
            if len(parts) == 2 and method == "GET":
                t = next((x for x in self.db["transactions"] if x["id"] == parts[1]), None)
                return (200, t) if t else (404, {"error": "Transaction not found"})

        if parts[0] == "disputes":
            if len(parts) == 1 and method == "GET":
                status_filter = (body or {}).get("status")
                items = self.db["disputes"]
                if status_filter:
                    items = [d for d in items if d["status"] == status_filter]
                return 200, {"data": items, "total": len(items)}
            if len(parts) == 2 and method == "GET":
                d = next((x for x in self.db["disputes"] if x["id"] == parts[1]), None)
                return (200, d) if d else (404, {"error": "Dispute not found"})
            if len(parts) == 3 and parts[2] == "resolve" and method == "POST":
                d = next((x for x in self.db["disputes"] if x["id"] == parts[1]), None)
                if not d:
                    return 404, {"error": "Dispute not found"}
                if d["status"] == "resolved":
                    return 409, {"error": "Already resolved"}
                d["status"] = "resolved"
                return 200, {"message": "Dispute resolved", "dispute": d}

        if parts[0] == "summary" and method == "GET":
            txns = self.db["transactions"]
            return 200, {
                "total_transactions": len(txns),
                "total_volume": round(sum(t["amount"] for t in txns), 2),
                "failed_count": sum(1 for t in txns if t["status"] == "failed"),
                "open_disputes": sum(1 for d in self.db["disputes"] if d["status"] == "open"),
            }

        # ── unknown route ──────────────────────────────────────────────────────
        return 404, {"error": f"Cannot {method} /{path}", "hint": "Try exploring available endpoints."}

    @property
    def total_calls(self) -> int:
        return self._calls
