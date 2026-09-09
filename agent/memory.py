"""
Task 8 (Part 2): Persisted Conversation Memory
----------------------------------------------------
Persists conversation history to a JSON file, keyed by thread_id, so that
multi-turn context survives across separate script runs (not just in-memory
within one Python process).
"""

import json
import os
from datetime import datetime, timezone

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_store")
os.makedirs(MEMORY_DIR, exist_ok=True)


def _memory_path(thread_id: str) -> str:
    safe_id = "".join(c for c in thread_id if c.isalnum() or c in ("-", "_"))
    return os.path.join(MEMORY_DIR, f"{safe_id}.json")


def load_history(thread_id: str) -> list:
    """Returns the list of turns for a thread, or [] if none exist yet."""
    path = _memory_path(thread_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_turn(thread_id: str, query: str, response: dict) -> list:
    """Appends one (query, response) turn to the thread's persisted history."""
    history = load_history(thread_id)
    turn = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "response": response,
    }
    history.append(turn)

    path = _memory_path(thread_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


def clear_history(thread_id: str) -> None:
    path = _memory_path(thread_id)
    if os.path.exists(path):
        os.remove(path)


def format_history_as_context(thread_id: str, max_turns: int = 5) -> str:
    """Formats recent turns into a short text block usable as conversation
    context for the agent (e.g. to prepend to a new query)."""
    history = load_history(thread_id)
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        answer = turn["response"].get("answer") or turn["response"].get("status", "")
        lines.append(f"User: {turn['query']}")
        lines.append(f"Agent: {answer}")
    return "\n".join(lines)


if __name__ == "__main__":
    demo_thread = "demo_thread_multiturn"
    fresh_thread = "demo_thread_fresh"

    clear_history(demo_thread)
    clear_history(fresh_thread)

    print("=" * 70)
    print("TRANSCRIPT A: multi-turn conversation on 'demo_thread_multiturn'")
    print("=" * 70)

    append_turn(demo_thread, "What is the interest rate for a home loan?",
                {"type": "policy_answer", "answer": "Home loans: 8.5%-11% p.a."})
    print("Turn 1 added.")

    append_turn(demo_thread, "And what about auto loans?",
                {"type": "policy_answer", "answer": "Auto loans: 9%-14% p.a."})
    print("Turn 2 added.")

    history = load_history(demo_thread)
    print(f"\nPersisted history now has {len(history)} turns:")
    for t in history:
        print(f"  [{t['timestamp']}] Q: {t['query']}")
        print(f"           A: {t['response']['answer']}")

    print("\nFormatted context for next turn:")
    print(format_history_as_context(demo_thread))

    print("\n" + "=" * 70)
    print("TRANSCRIPT B: FRESH conversation on 'demo_thread_fresh' (should be empty)")
    print("=" * 70)

    fresh_history = load_history(fresh_thread)
    print(f"History for '{fresh_thread}' before any turns: {fresh_history}")
    assert fresh_history == [], "Fresh thread should have no prior history"
    print("Confirmed: fresh thread starts with correctly EMPTY history "
          "(state is scoped per thread_id, not shared globally).")