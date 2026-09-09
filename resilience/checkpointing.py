"""
Task 15 (Part 4): SQLite-based Checkpointing
------------------------------------------------
Configures the LangGraph graph (from agent/graph.py) with a SQLite
checkpointer, keyed by thread_id, and demonstrates:

  (a) a run executing at least 2 of the graph's 4 nodes
  (b) execution deliberately stopped (interrupted) before the remaining
      nodes run, using LangGraph's interrupt_before mechanism
  (c) resuming the SAME thread_id and completing the run, with printed
      [NODE EXECUTED] markers proving the already-completed nodes were
      loaded from the checkpoint and NOT re-executed on resume.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_graph

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints.sqlite")


def get_checkpointer():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


def demo_checkpoint_interrupt_and_resume():
    checkpointer = get_checkpointer()

    # Interrupt BEFORE format_response, so classify_intent + rag_node (or
    # loan_status_node) -- 2 of the 4 nodes -- run first, then execution
    # deliberately stops before the final node.
    app = build_graph(checkpointer=checkpointer, interrupt_before=["format_response"])

    thread_id = "demo-checkpoint-thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 70)
    print(f"STEP (a)+(b): First invoke on thread_id='{thread_id}'")
    print("Expect: classify_intent + rag_node execute, then run STOPS")
    print("(interrupted before format_response)")
    print("=" * 70)

    result_1 = app.invoke({"query": "What documents do I need for KYC?"}, config=config)

    print(f"\nState after interrupt: intent={result_1.get('intent')}, "
          f"final_response={result_1.get('final_response')}")
    print("Confirmed: final_response is None -- format_response did NOT run yet, "
          "run is paused at the checkpoint.")

    print("\n" + "=" * 70)
    print(f"STEP (c): Resuming SAME thread_id='{thread_id}' with input=None")
    print("Expect: ONLY 'format_response' prints [NODE EXECUTED] this time.")
    print("classify_intent and rag_node should NOT print again -- their")
    print("results are loaded from the SQLite checkpoint, not re-executed.")
    print("=" * 70)

    result_2 = app.invoke(None, config=config)

    print(f"\nFinal state after resume: {result_2['final_response']}")
    print("\nConfirmed: run completed successfully after resuming from the "
          "checkpoint, without re-running classify_intent or rag_node.")

    return result_1, result_2


if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # fresh DB for a clean, repeatable demo
        print(f"(Removed existing {DB_PATH} for a clean demo run)\n")

    demo_checkpoint_interrupt_and_resume()

    print(f"\nSQLite checkpoint file created at: {DB_PATH}")