"""
Task 7 (Part 2): LangGraph Agent with Conditional Routing
--------------------------------------------------------------
Builds a LangGraph graph with 4 nodes:
    1. classify_intent   - decides whether the query is a policy question
                            (route to RAG) or a loan-status lookup
                            (route to the check_loan_application_status tool)
    2. rag_node           - answers policy questions using Part 1's RAG core
    3. loan_status_node   - answers status lookups using Task 6's tool
    4. format_response    - normalizes either branch's output into one
                             consistent response shape

1 genuine conditional edge: classify_intent -> (rag_node | loan_status_node),
chosen based on query intent detected at runtime.
"""

import re
import sys
import os
from typing import TypedDict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from agent.tools import check_loan_application_status
from rag.indexer import build_and_index_all
from rag.retrieval import calibrate_threshold, grounded_answer

# Record IDs look like LN0001, LN0002, ... (see data/dataset.py)
RECORD_ID_PATTERN = re.compile(r"\bLN\d{4}\b", re.IGNORECASE)

STATUS_KEYWORDS = ["status", "application", "loan application", "my loan process", "check my"]


class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    record_id: Optional[str]
    rag_result: Optional[dict]
    tool_result: Optional[dict]
    final_response: Optional[dict]


# --- Node 1: intent classification ---
def classify_intent(state: AgentState) -> AgentState:
    print("  [NODE EXECUTED] classify_intent")
    query = state["query"]
    match = RECORD_ID_PATTERN.search(query)

    if match:
        state["intent"] = "status_lookup"
        state["record_id"] = match.group(0).upper()
    elif any(kw in query.lower() for kw in STATUS_KEYWORDS):
        state["intent"] = "status_lookup"
        state["record_id"] = None
    else:
        state["intent"] = "policy_question"
        state["record_id"] = None

    return state


# --- Conditional routing function ---
def route_after_classify(state: AgentState) -> str:
    return state["intent"]


# --- Node 2: RAG branch ---
_fixed_collection = None
_calibrated_threshold = None


def _get_rag_resources():
    global _fixed_collection, _calibrated_threshold
    if _fixed_collection is None:
        _fixed_collection, _ = build_and_index_all()
        _calibrated_threshold = calibrate_threshold(_fixed_collection)
    return _fixed_collection, _calibrated_threshold


def rag_node(state: AgentState) -> AgentState:
    print("  [NODE EXECUTED] rag_node")
    collection, threshold = _get_rag_resources()
    result = grounded_answer(collection, state["query"], threshold=threshold)
    state["rag_result"] = result
    return state


# --- Node 3: loan status branch ---
def loan_status_node(state: AgentState) -> AgentState:
    print("  [NODE EXECUTED] loan_status_node")
    record_id = state.get("record_id")

    if not record_id:
        state["tool_result"] = {
            "found": False,
            "error": "No record_id detected in the query. Please provide a "
                     "loan application ID such as LN0001.",
        }
        return state

    result = check_loan_application_status(record_id)
    state["tool_result"] = result
    return state


# --- Node 4: format final response ---
def format_response(state: AgentState) -> AgentState:
    print("  [NODE EXECUTED] format_response")
    if state["intent"] == "policy_question":
        r = state["rag_result"]
        state["final_response"] = {
            "type": "policy_answer",
            "query": state["query"],
            "answer": r["answer"],
            "grounded": r["grounded"],
            "sources": r["sources"],
        }
    else:
        r = state["tool_result"]
        if not r.get("found"):
            state["final_response"] = {
                "type": "status_lookup_error",
                "query": state["query"],
                "answer": r.get("error", f"No application found for {state.get('record_id')}."),
            }
        else:
            state["final_response"] = {
                "type": "status_lookup",
                "query": state["query"],
                "record_id": r["record_id"],
                "status": r["status"],
                "loan_amount_inr": r["loan_amount_inr"],
                "escalation_score": r["escalation_score"],
                "recommend_escalation": r["recommend_escalation"],
            }
    return state


def build_graph(checkpointer=None, interrupt_before=None):
    """
    Builds and compiles the graph. checkpointer and interrupt_before are
    optional (used by resilience/checkpointing.py for Task 15); default
    behavior (no checkpointer, no interrupt) is unchanged from Task 7.
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rag_node", rag_node)
    graph.add_node("loan_status_node", loan_status_node)
    graph.add_node("format_response", format_response)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "policy_question": "rag_node",
            "status_lookup": "loan_status_node",
        },
    )

    graph.add_edge("rag_node", "format_response")
    graph.add_edge("loan_status_node", "format_response")
    graph.add_edge("format_response", END)

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before is not None:
        compile_kwargs["interrupt_before"] = interrupt_before

    return graph.compile(**compile_kwargs)


if __name__ == "__main__":
    app = build_graph()

    print("=" * 70)
    print("DEMO 1: policy question -> should route to rag_node")
    print("=" * 70)
    result1 = app.invoke({"query": "What documents do I need for KYC?"})
    print(result1["final_response"])

    print("\n" + "=" * 70)
    print("DEMO 2: status lookup -> should route to loan_status_node")
    print("=" * 70)
    result2 = app.invoke({"query": "What is the status of application LN0005?"})
    print(result2["final_response"])

    print("\n" + "=" * 70)
    print("DEMO 3: another policy question -> rag_node")
    print("=" * 70)
    result3 = app.invoke({"query": "How is EMI calculated?"})
    print(result3["final_response"])

    print("\n" + "=" * 70)
    print("DEMO 4: another status lookup -> loan_status_node")
    print("=" * 70)
    result4 = app.invoke({"query": "Check status for LN0002"})
    print(result4["final_response"])