"""
data/store.py — In-memory simulated backend databases.

Provides realistic fake data for:
  - Orders (with items, dates, statuses)
  - Customer accounts (with subscription, balance)
  - Transactions (payments, credits, charges)
  - Knowledge base articles
  - Service status dashboard
  - Support ticket system
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


# ---------------------------------------------------------------------------
# Master DB — mutable across a single episode, reset between episodes
# ---------------------------------------------------------------------------

_ORIGINAL_DB: Dict[str, Any] = {
    # ---- Orders ----------------------------------------------------------------
    "orders": {
        "ORD-1001": {
            "order_id":    "ORD-1001",
            "customer_id": "CUST-001",
            "status":      "delivered",
            "items": [
                {"sku": "HDPH-500", "name": "NoiseBlock Pro Headphones", "qty": 1, "price": 149.99}
            ],
            "total":        149.99,
            "placed_at":    _days_ago(10),
            "delivered_at": _days_ago(7),
            "refund_issued": False,
        },
        "ORD-1002": {
            "order_id":    "ORD-1002",
            "customer_id": "CUST-002",
            "status":      "delivered",
            "items": [
                {"sku": "CABLE-USB", "name": "USB-C Cable 2m", "qty": 2, "price": 12.99}
            ],
            "total":        25.98,
            "placed_at":    _days_ago(45),   # Outside 30-day window!
            "delivered_at": _days_ago(42),
            "refund_issued": False,
        },
        "ORD-1003": {
            "order_id":    "ORD-1003",
            "customer_id": "CUST-003",
            "status":      "delivered",
            "items": [
                {"sku": "WEBCAM-4K", "name": "4K Webcam Ultra", "qty": 1, "price": 89.99}
            ],
            "total":        89.99,
            "placed_at":    _days_ago(5),
            "delivered_at": _days_ago(3),
            "refund_issued": False,
        },
    },

    # ---- Accounts --------------------------------------------------------------
    "accounts": {
        "CUST-001": {
            "customer_id": "CUST-001",
            "name":        "Priya Sharma",
            "email":       "priya.sharma@example.com",
            "plan":        "premium",
            "balance":     0.00,
            "status":      "active",
            "joined_at":   _days_ago(365),
        },
        "CUST-002": {
            "customer_id": "CUST-002",
            "name":        "Marcus Webb",
            "email":       "marcus.webb@example.com",
            "plan":        "basic",
            "balance":     0.00,
            "status":      "active",
            "joined_at":   _days_ago(90),
        },
        "CUST-003": {
            "customer_id": "CUST-003",
            "name":        "Aiko Tanaka",
            "email":       "aiko.tanaka@example.com",
            "plan":        "premium",
            "balance":     0.00,
            "status":      "active",
            "joined_at":   _days_ago(200),
        },
    },

    # ---- Transactions ----------------------------------------------------------
    "transactions": {
        "TXN-5001": {
            "txn_id":      "TXN-5001",
            "customer_id": "CUST-003",
            "type":        "charge",
            "amount":      89.99,
            "description": "Monthly Premium Subscription",
            "date":        _days_ago(2),
            "status":      "settled",
        },
        "TXN-5002": {
            "txn_id":      "TXN-5002",
            "customer_id": "CUST-003",
            "type":        "charge",
            "amount":      89.99,
            "description": "Monthly Premium Subscription",
            "date":        _days_ago(2),   # Same day — duplicate!
            "status":      "settled",
        },
        "TXN-5003": {
            "txn_id":      "TXN-5003",
            "customer_id": "CUST-001",
            "type":        "charge",
            "amount":      149.99,
            "description": "Order ORD-1001",
            "date":        _days_ago(10),
            "status":      "settled",
        },
    },

    # ---- Credits ---------------------------------------------------------------
    "credits": {},   # filled by apply_credit tool

    # ---- Emails sent -----------------------------------------------------------
    "emails_sent": [],

    # ---- Support tickets -------------------------------------------------------
    "support_tickets": {},

    # ---- Service status --------------------------------------------------------
    "services": {
        "authentication": {"status": "degraded",  "error_code": "AUTH-503", "affected_since": _days_ago(0)},
        "payment":        {"status": "operational","error_code": None,        "affected_since": None},
        "streaming":      {"status": "outage",     "error_code": "STREAM-504","affected_since": _days_ago(0)},
        "api":            {"status": "operational","error_code": None,        "affected_since": None},
    },

    # ---- Workarounds applied ---------------------------------------------------
    "workarounds_applied": {},

    # ---- Customers notified ----------------------------------------------------
    "notifications_sent": [],

    # ---- Refunds ---------------------------------------------------------------
    "refunds": {},
}

# Live DB — gets mutated during episode
_DB: Dict[str, Any] = {}


def reset_db() -> None:
    """Deep-copy original into live DB (called at episode reset)."""
    global _DB
    _DB = copy.deepcopy(_ORIGINAL_DB)


def get_db() -> Dict[str, Any]:
    return _DB


# ---------------------------------------------------------------------------
# Knowledge Base articles
# ---------------------------------------------------------------------------

KB_ARTICLES = [
    {
        "id": "KB-001",
        "title": "Refund Policy",
        "content": (
            "Customers may request a full refund within 30 days of delivery. "
            "The item must not have been intentionally damaged by the customer. "
            "Refunds are processed within 3-5 business days back to the original payment method. "
            "To issue a refund: use the process_refund tool with the order_id."
        ),
        "tags": ["refund", "return", "policy", "money back"],
    },
    {
        "id": "KB-002",
        "title": "Billing Dispute Process",
        "content": (
            "If a customer is double-charged, use list_transactions to identify duplicates. "
            "Apply a credit equal to the duplicate amount using apply_credit. "
            "Then send a confirmation email using send_email. "
            "Always verify the account with lookup_account first."
        ),
        "tags": ["billing", "charge", "duplicate", "credit", "dispute"],
    },
    {
        "id": "KB-003",
        "title": "Service Outage Escalation SOP",
        "content": (
            "When a customer reports a service outage: "
            "1. Check get_service_status for affected services. "
            "2. Apply a known workaround using apply_workaround if available. "
            "3. Create an escalation ticket with create_ticket. "
            "4. Notify all affected customers with notify_customers. "
            "5. Respond to the customer with the ticket number and ETA."
        ),
        "tags": ["outage", "technical", "escalation", "service", "down", "SLA"],
    },
    {
        "id": "KB-004",
        "title": "Account Status Codes",
        "content": (
            "active: normal account. "
            "suspended: account on hold, no refunds allowed. "
            "cancelled: account closed. "
            "Always check account status before processing any financial action."
        ),
        "tags": ["account", "status", "suspended", "active"],
    },
]


def search_kb(query: str) -> List[Dict]:
    """Simple keyword search over KB articles."""
    query_lower = query.lower()
    results = []
    for article in KB_ARTICLES:
        score = 0
        for tag in article["tags"]:
            if tag in query_lower or query_lower in tag:
                score += 2
        if query_lower in article["title"].lower():
            score += 3
        if query_lower in article["content"].lower():
            score += 1
        if score > 0:
            results.append({**article, "_score": score})
    results.sort(key=lambda x: x["_score"], reverse=True)
    return results[:3]
