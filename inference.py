"""
inference.py — Baseline inference script for CustomerSupportEnv.

Mandatory environment variables:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.
                 (also accepted: OPENAI_API_KEY, API_KEY)

Usage:
  python inference.py                        # runs all 3 tasks
  python inference.py --task order_refund    # runs one task
  python inference.py --verbose              # extra debug to stderr

Stdout emits structured logs in EXACTLY this format (evaluated by scorer):
  [START] {"task_id": "...", "model": "...", "timestamp": "..."}
  [STEP]  {"task_id": "...", "step": N, "action": "...", "reward": 0.0, "done": false}
  [END]   {"task_id": "...", "score": 0.0, "steps": N, "timestamp": "..."}

All other output goes to stderr only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Local env imports
# ---------------------------------------------------------------------------
from env import CustomerSupportEnv
from models import Action, TaskId, ToolName

# ---------------------------------------------------------------------------
# Credentials — read from env vars per submission spec
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = (
    os.getenv("HF_TOKEN")       # primary — per submission spec
    or os.getenv("API_KEY")     # fallback alias
    or "hf-placeholder"
)
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_STEPS   = 20
TEMPERATURE = 0.0   # deterministic for reproducibility
MAX_TOKENS  = 512
FALLBACK_MESSAGE = (
    "I apologize for the inconvenience. "
    "Our team will follow up with you shortly. "
    "Please don't hesitate to contact us again."
)
# ---------------------------------------------------------------------------
# Structured logging — stdout ONLY, exact format required by scorer
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_start(task_id: str) -> None:
    """[START] line — stdout."""
    record = {"task_id": task_id, "model": MODEL_NAME, "timestamp": _ts()}
    print(f"[START] {json.dumps(record)}", flush=True)


def log_step(task_id: str, step: int, action: str, reward: float, done: bool) -> None:
    """[STEP] line — stdout."""
    # Clamp reward strictly between 0 and 1 (exclusive)
    reward = max(0.01, min(0.99, reward))
    record = {"task_id": task_id, "step": step, "action": action, "reward": reward, "done": done}
    print(f"[STEP] {json.dumps(record)}", flush=True)


def log_end(task_id: str, score: float, steps: int) -> None:
    """[END] line — stdout."""
    # Clamp score strictly between 0 and 1 (exclusive)
    score = max(0.01, min(0.99, score))
    record = {"task_id": task_id, "score": score, "steps": steps, "timestamp": _ts()}
    print(f"[END] {json.dumps(record)}", flush=True)


def dbg(msg: str) -> None:
    """Debug — stderr only, never pollutes structured stdout."""
    print(f"[DBG] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support agent with access to a set of tools.
Your job is to resolve customer issues by calling the appropriate tools
and finally sending a helpful response to the customer.

AVAILABLE TOOLS:
- lookup_order(order_id)
- process_refund(order_id, reason)
- lookup_account(customer_id)
- list_transactions(customer_id, limit)
- apply_credit(customer_id, amount, reason)
- send_email(customer_id, subject, body)
- search_kb(query)
- create_ticket(customer_id, title, description, priority, category)
- get_service_status(service)
- apply_workaround(service, workaround_code)
- notify_customers(service, message, affected_customer_ids)
- respond(message)   <- FINAL reply to customer, ends the episode

RULES:
1. Always verify the customer account and order before financial actions.
2. Follow the 30-day refund policy strictly.
3. For billing disputes: list_transactions first, then apply_credit.
4. For technical issues: get_service_status then apply_workaround then
   create_ticket then notify_customers then respond.
5. Your FINAL action MUST always be respond with a complete, empathetic message.
6. Be efficient: use only the tools you need.

OUTPUT FORMAT — return ONLY a valid JSON object, no prose, no markdown:
{
  "tool": "<tool_name>",
  "params": { "<key>": "<value>" },
  "message": "<text — only when tool is respond>"
}
""").strip()


# ---------------------------------------------------------------------------
# Build user prompt from observation
# ---------------------------------------------------------------------------

def build_user_prompt(obs) -> str:
    history_lines: List[str] = []
    for h in obs.history:
        if h.error:
            status = f"ERROR: {h.error}"
        else:
            result_str = json.dumps(h.result, default=str)
            status = f"OK: {result_str[:300]}"
        history_lines.append(
            f"  Step {h.step}: {h.tool}({json.dumps(h.params)}) -> {status}"
        )

    history_str = "\n".join(history_lines) if history_lines else "  (none yet)"

    return textwrap.dedent(f"""
        === CUSTOMER TICKET ===
        Ticket ID   : {obs.ticket.ticket_id}
        Customer ID : {obs.customer_id}
        Subject     : {obs.ticket.subject}
        Body:
        {obs.ticket.body}

        === CURRENT STATE ===
        Step          : {obs.step_number}
        SLA Remaining : {obs.sla_remaining}s
        Available Tools: {', '.join(obs.available_tools)}

        === ACTION HISTORY ===
        {history_str}

        What is your next action? Respond with JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# Parse model JSON response -> Action
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Optional[Action]:
    if not response_text:
        return None

    text = response_text.strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()

    # Try direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Extract first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    tool_name = str(data.get("tool", "respond")).strip()
    params    = data.get("params", {})
    message   = data.get("message", "") or ""

    valid_tools = {t.value for t in ToolName}
    if tool_name not in valid_tools:
        dbg(f"Unknown tool '{tool_name}' — falling back to respond")
        tool_name = "respond"
        message   = message or FALLBACK_MESSAGE

    if not isinstance(params, dict):
        params = {}

    try:
        return Action(
            tool    = ToolName(tool_name),
            params  = params,
            message = message or None,
        )
    except Exception as exc:
        dbg(f"Action construction error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Run a single episode
# ---------------------------------------------------------------------------

def run_episode(
    client: OpenAI,
    task_id: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    log_start(task_id)
    cumulative_reward = 0.0
    final_score       = 0.01  # Never emit exactly 0.0
    steps_taken       = 0
    last_result       = None
    try:
        env = CustomerSupportEnv(task_id=TaskId(task_id))
        obs = env.reset()
    except Exception as exc:
        dbg(f"Env init failed for {task_id}: {exc}")
        log_end(task_id=task_id, score=0.01, steps=1)
        return {"task_id": task_id, "score": 0.01, "steps": 1,
                "cumulative_reward": 0.01, "grader_breakdown": {}}

    if verbose:
        dbg(f"Ticket: {obs.ticket.subject} | Customer: {obs.customer_id}")

    for step_num in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        # Build LLM messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(obs)},
        ]

        # Call model via OpenAI client
        try:
            completion = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                stream      = False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            dbg(f"Model API error at step {step_num}: {exc}")
            response_text = json.dumps({
                "tool": "respond",
                "params": {},
                "message": FALLBACK_MESSAGE,
            })

        if verbose:
            dbg(f"Step {step_num} response: {response_text[:200]}")

        # Parse action
        action = parse_action(response_text)
        if action is None:
            dbg(f"Step {step_num}: unparseable response — using fallback")
            action = Action(
                tool    = ToolName.RESPOND,
                params  = {},
                message = FALLBACK_MESSAGE,
            )

        # Execute in environment
        result      = env.step(action)
        last_result = result
        steps_taken = result.observation.step_number
        cumulative_reward += result.reward.value

        # Emit [STEP] — exact format, all fields required
        log_step(
            task_id = task_id,
            step    = step_num,
            action  = action.tool.value,
            reward  = round(result.reward.value, 4),
            done    = result.done,
        )

        if verbose:
            tool_err = result.info.get("tool_error", "")
            dbg(
                f"  action={action.tool.value}"
                f" reward={result.reward.value:.4f}"
                f" done={result.done}"
                + (f" err={tool_err}" if tool_err else "")
            )

        obs = result.observation

        if result.done:
            raw = round(result.reward.breakdown.total, 4)
            # Clamp strictly between 0 and 1 (exclusive)
            final_score = max(0.01, min(0.99, raw))
            break

    # Ensure final_score is strictly between 0 and 1 (exclusive)
    final_score = max(0.01, min(0.99, final_score))
    # Emit [END]
    log_end(task_id=task_id, score=final_score, steps=max(steps_taken, 1))

    breakdown: Dict[str, float] = {}
    if last_result is not None:
        bd = last_result.reward.breakdown
        breakdown = {
            "resolution_correct":    round(bd.resolution_correct, 4),
            "policy_compliance":     round(bd.policy_compliance, 4),
            "efficiency":            round(bd.efficiency, 4),
            "customer_satisfaction": round(bd.customer_satisfaction, 4),
        }

    return {
        "task_id":            task_id,
        "score":              final_score,
        "steps":              steps_taken,
        "cumulative_reward":  round(cumulative_reward, 4),
        "grader_breakdown":   breakdown,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_TASKS = [
    "order_refund",
    "account_billing_dispute",
    "technical_escalation",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CustomerSupportEnv — baseline inference"
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=ALL_TASKS + ["all"],
        help="Task(s) to run (default: all)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    dbg(f"API_BASE_URL : {API_BASE_URL}")
    dbg(f"MODEL_NAME   : {MODEL_NAME}")
    dbg(f"HF_TOKEN     : {'set' if API_KEY != 'hf-placeholder' else 'NOT SET'}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]

    results: List[Dict[str, Any]] = []
    for tid in tasks_to_run:
        dbg(f"Starting task: {tid}")
        try:
            r = run_episode(client, tid, verbose=args.verbose)
        except Exception as exc:
            dbg(f"FATAL error in task {tid}: {exc}")
            # Always emit [END] so the validator can parse a score
            log_end(task_id=tid, score=0.01, steps=1)
            r = {"task_id": tid, "score": 0.01, "steps": 1,
                 "cumulative_reward": 0.01, "grader_breakdown": {}}
        results.append(r)
        dbg(f"Finished {tid}: score={r['score']:.4f} steps={r['steps']}")

    # Summary to stderr only
    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    dbg("=" * 52)
    dbg("FINAL SCORE SUMMARY")
    dbg("=" * 52)
    for r in results:
        bar = "█" * int(r["score"] * 20)
        dbg(f"  {r['task_id']:<32} {r['score']:.4f}  {bar}")
    dbg(f"  {'AVERAGE':<32} {avg:.4f}")
    dbg("=" * 52)


if __name__ == "__main__":
    main()