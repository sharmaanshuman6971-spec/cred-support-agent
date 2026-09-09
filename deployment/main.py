"""
Task 11 (Part 3): FastAPI Deployment
------------------------------------------
Exposes the agent behind 2 endpoints:
    POST /ask           - ask the agent a policy question or status lookup
    POST /add-document   - add a new document to the knowledge base and
                            re-index it into both ChromaDB collections

Every request is:
  - passed through the Task 10 input guardrail (PII masking + injection check)
  - validated against the Task 9 structured-output schema before being returned
  - logged as one JSON-Lines entry with a trace ID and timing (Task 12)
"""

import sys
import os
import time
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from agent.graph import build_graph
from agent.guardrails import input_guardrail
from agent.schema import validate_agent_response
from agent.memory import append_turn, load_history
from deployment.logging_utils import log_request, new_trace_id

from knowledge_base.documents import KNOWLEDGE_BASE_DOCUMENTS
from rag.indexer import build_and_index_all

app = FastAPI(title="Cred Domain Support Agent", version="1.0.0")

_graph_app = build_graph()


# ---- Pydantic models ----

class AskRequest(BaseModel):
    query: str = Field(..., description="The member's question or status-lookup request.")
    thread_id: str = Field(default="default_thread", description="Conversation thread ID for memory.")


class AskResponse(BaseModel):
    trace_id: str
    thread_id: str
    blocked: bool
    block_reason: Optional[str] = None
    response: Optional[dict] = None
    schema_valid: Optional[bool] = None


class AddDocumentRequest(BaseModel):
    doc_id: str
    topic: str
    title: str
    text: str


class AddDocumentResponse(BaseModel):
    trace_id: str
    added: bool
    total_documents: int
    reindexed: bool


# ---- Endpoints ----

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start_time = time.time()
    trace_id = new_trace_id()

    guard_result = input_guardrail(request.query)

    if guard_result["blocked"]:
        duration_ms = (time.time() - start_time) * 1000
        log_request(
            endpoint="/ask",
            raw_request_text=request.query,
            response_summary={"blocked": True, "reason": "prompt_injection_detected"},
            duration_ms=duration_ms,
            trace_id=trace_id,
        )
        return AskResponse(
            trace_id=trace_id,
            thread_id=request.thread_id,
            blocked=True,
            block_reason=f"Prompt injection detected: '{guard_result['injection_pattern']}'",
            response=None,
            schema_valid=None,
        )

    graph_result = _graph_app.invoke({"query": guard_result["safe_text"]})
    final_response = graph_result["final_response"]

    validation = validate_agent_response(final_response)

    append_turn(request.thread_id, request.query, final_response)

    duration_ms = (time.time() - start_time) * 1000
    log_request(
        endpoint="/ask",
        raw_request_text=request.query,
        response_summary={"type": final_response.get("type"), "schema_valid": validation["valid"]},
        duration_ms=duration_ms,
        trace_id=trace_id,
    )

    return AskResponse(
        trace_id=trace_id,
        thread_id=request.thread_id,
        blocked=False,
        block_reason=None,
        response=final_response,
        schema_valid=validation["valid"],
    )


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest):
    start_time = time.time()
    trace_id = new_trace_id()

    KNOWLEDGE_BASE_DOCUMENTS.append({
        "doc_id": request.doc_id,
        "topic": request.topic,
        "title": request.title,
        "text": request.text,
    })

    # Re-index both ChromaDB collections so the new document is searchable.
    build_and_index_all()

    duration_ms = (time.time() - start_time) * 1000
    log_request(
        endpoint="/add-document",
        raw_request_text=f"doc_id={request.doc_id} title={request.title}",
        response_summary={"added": True, "total_documents": len(KNOWLEDGE_BASE_DOCUMENTS)},
        duration_ms=duration_ms,
        trace_id=trace_id,
    )

    return AddDocumentResponse(
        trace_id=trace_id,
        added=True,
        total_documents=len(KNOWLEDGE_BASE_DOCUMENTS),
        reindexed=True,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deployment.main:app", host="127.0.0.1", port=8000, reload=False)