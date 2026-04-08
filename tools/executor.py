"""
tools/executor.py — Simulated external tool implementations.

Each tool:
  - Validates its parameters
  - Reads/writes the in-memory DB
  - Returns a structured result dict  OR  raises ToolError
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from data.store import get_db, search_kb


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Raised when a tool call fails (bad params, not found, policy violation)."""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _days_since(iso_str: str) -> float:
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> Dict[str, Any]:
    """Retrieve order details by order ID."""
    if not order_id:
        raise ToolError("order_id is required")
    db = get_db()
    order = db["orders"].get(order_id)
    if order is None:
        raise ToolError(f"Order '{order_id}' not found")
    return dict(order)


def process_refund(order_id: str, reason: str = "") -> Dict[str, Any]:
    """
    Issue a refund for an order.
    Policy: order must exist, must be delivered, refund_issued=False,
    and must be within 30 days of delivery.
    """
    db = get_db()
    order = db["orders"].get(order_id)
    if order is None:
        raise ToolError(f"Order '{order_id}' not found")
    if order["status"] != "delivered":
        raise ToolError(f"Order '{order_id}' is not in 'delivered' status (status={order['status']})")
    if order["refund_issued"]:
        raise ToolError(f"Refund already issued for order '{order_id}'")

    days = _days_since(order["delivered_at"])
    if days > 30:
        raise ToolError(
            f"Order '{order_id}' is outside the 30-day refund window "
            f"({days:.0f} days since delivery). Refund denied per policy."
        )

    # Check account is active
    customer_id = order["customer_id"]
    account = db["accounts"].get(customer_id, {})
    if account.get("status") == "suspended":
        raise ToolError(f"Account {customer_id} is suspended; cannot process refund")

    # Issue refund
    refund_id = f"REF-{order_id}"
    db["refunds"][refund_id] = {
        "refund_id":   refund_id,
        "order_id":    order_id,
        "customer_id": customer_id,
        "amount":      order["total"],
        "reason":      reason or "Customer request",
        "issued_at":   _now_iso(),
        "status":      "processed",
    }
    db["orders"][order_id]["refund_issued"] = True

    return {
        "success":   True,
        "refund_id": refund_id,
        "amount":    order["total"],
        "message":   f"Refund of ${order['total']:.2f} issued successfully for order {order_id}.",
    }


def lookup_account(customer_id: str) -> Dict[str, Any]:
    """Retrieve account details for a customer."""
    if not customer_id:
        raise ToolError("customer_id is required")
    db = get_db()
    account = db["accounts"].get(customer_id)
    if account is None:
        raise ToolError(f"Account '{customer_id}' not found")
    return dict(account)


def list_transactions(customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """List recent transactions for a customer."""
    if not customer_id:
        raise ToolError("customer_id is required")
    db = get_db()
    txns = [
        t for t in db["transactions"].values()
        if t["customer_id"] == customer_id
    ]
    txns.sort(key=lambda x: x["date"], reverse=True)
    return txns[:limit]


def apply_credit(customer_id: str, amount: float, reason: str = "") -> Dict[str, Any]:
    """Apply a credit to a customer's account."""
    if not customer_id:
        raise ToolError("customer_id is required")
    if amount <= 0:
        raise ToolError(f"Credit amount must be positive, got {amount}")
    db = get_db()
    account = db["accounts"].get(customer_id)
    if account is None:
        raise ToolError(f"Account '{customer_id}' not found")
    if account["status"] == "suspended":
        raise ToolError(f"Cannot apply credit to suspended account {customer_id}")

    credit_id = f"CRD-{customer_id}-{len(db['credits'])+1}"
    db["credits"][credit_id] = {
        "credit_id":   credit_id,
        "customer_id": customer_id,
        "amount":      amount,
        "reason":      reason or "Customer service credit",
        "applied_at":  _now_iso(),
    }
    db["accounts"][customer_id]["balance"] += amount

    return {
        "success":   True,
        "credit_id": credit_id,
        "amount":    amount,
        "new_balance": db["accounts"][customer_id]["balance"],
        "message":   f"Credit of ${amount:.2f} applied to account {customer_id}.",
    }


def send_email(
    customer_id: str,
    subject: str,
    body: str,
) -> Dict[str, Any]:
    """Send an email to a customer."""
    if not customer_id or not subject or not body:
        raise ToolError("customer_id, subject, and body are all required")
    db = get_db()
    account = db["accounts"].get(customer_id)
    if account is None:
        raise ToolError(f"Account '{customer_id}' not found — cannot determine email address")

    email_record = {
        "to":         account["email"],
        "customer_id": customer_id,
        "subject":    subject,
        "body":       body,
        "sent_at":    _now_iso(),
    }
    db["emails_sent"].append(email_record)
    return {
        "success": True,
        "to":      account["email"],
        "message": f"Email sent to {account['email']}.",
    }


def search_kb_tool(query: str) -> List[Dict[str, Any]]:
    """Search the internal knowledge base."""
    if not query:
        raise ToolError("query is required")
    results = search_kb(query)
    if not results:
        return [{"message": "No articles found for that query."}]
    return results


def create_ticket(
    customer_id: str,
    title: str,
    description: str,
    priority: str = "medium",
    category: str = "general",
) -> Dict[str, Any]:
    """Create an internal escalation/support ticket."""
    if not customer_id or not title or not description:
        raise ToolError("customer_id, title, and description are required")
    valid_priorities = {"low", "medium", "high", "urgent"}
    if priority not in valid_priorities:
        raise ToolError(f"priority must be one of {valid_priorities}, got '{priority}'")

    db = get_db()
    ticket_num = len(db["support_tickets"]) + 1
    ticket_id = f"TKT-{ticket_num:04d}"
    db["support_tickets"][ticket_id] = {
        "ticket_id":   ticket_id,
        "customer_id": customer_id,
        "title":       title,
        "description": description,
        "priority":    priority,
        "category":    category,
        "status":      "open",
        "created_at":  _now_iso(),
    }
    return {
        "success":   True,
        "ticket_id": ticket_id,
        "message":   f"Escalation ticket {ticket_id} created with priority '{priority}'.",
        "eta":       "2-4 business hours" if priority == "urgent" else "4-8 business hours",
    }


def get_service_status(service: str = "") -> Dict[str, Any]:
    """Get the operational status of one or all services."""
    db = get_db()
    if service:
        status = db["services"].get(service)
        if status is None:
            raise ToolError(
                f"Service '{service}' not found. "
                f"Available services: {list(db['services'].keys())}"
            )
        return {service: status}
    # Return all
    return dict(db["services"])


def apply_workaround(service: str, workaround_code: str) -> Dict[str, Any]:
    """Apply a known workaround for a degraded or down service."""
    db = get_db()
    service_data = db["services"].get(service)
    if service_data is None:
        raise ToolError(f"Service '{service}' not found")
    if service_data["status"] == "operational":
        return {"success": False, "message": f"Service '{service}' is already operational."}

    # Accept any non-empty workaround_code (realistic: codes come from runbooks)
    if not workaround_code:
        raise ToolError("workaround_code is required")

    db["workarounds_applied"][service] = {
        "service":         service,
        "workaround_code": workaround_code,
        "applied_at":      _now_iso(),
        "previous_status": service_data["status"],
    }
    # Partially restore service
    db["services"][service]["status"] = "degraded" if service_data["status"] == "outage" else "operational"

    return {
        "success":   True,
        "service":   service,
        "new_status": db["services"][service]["status"],
        "message":   f"Workaround applied for {service}. Status changed to '{db['services'][service]['status']}'.",
    }


def notify_customers(service: str, message: str, affected_customer_ids: List[str] = None) -> Dict[str, Any]:
    """Send proactive notifications to customers affected by a service issue."""
    if not service or not message:
        raise ToolError("service and message are required")
    db = get_db()
    if service not in db["services"]:
        raise ToolError(f"Service '{service}' not found")

    # If no specific list given, notify all customers
    if not affected_customer_ids:
        affected_customer_ids = list(db["accounts"].keys())

    sent_to = []
    for cid in affected_customer_ids:
        account = db["accounts"].get(cid)
        if account:
            db["notifications_sent"].append({
                "customer_id": cid,
                "email":       account["email"],
                "service":     service,
                "message":     message,
                "sent_at":     _now_iso(),
            })
            sent_to.append(account["email"])

    return {
        "success":          True,
        "notified_count":   len(sent_to),
        "notified_emails":  sent_to,
        "message":          f"Notified {len(sent_to)} customers about {service} issue.",
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "lookup_order":       lookup_order,
    "process_refund":     process_refund,
    "lookup_account":     lookup_account,
    "list_transactions":  list_transactions,
    "apply_credit":       apply_credit,
    "send_email":         send_email,
    "search_kb":          search_kb_tool,
    "create_ticket":      create_ticket,
    "get_service_status": get_service_status,
    "apply_workaround":   apply_workaround,
    "notify_customers":   notify_customers,
}


def execute_tool(tool_name: str, params: Dict[str, Any]) -> Any:
    """
    Dispatch a tool call.
    Returns the tool result or raises ToolError.
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        raise ToolError(
            f"Unknown tool '{tool_name}'. "
            f"Available tools: {list(TOOL_REGISTRY.keys())}"
        )
    return fn(**params)
