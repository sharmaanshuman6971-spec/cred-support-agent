# Cred Domain Support Agent (LangGraph)

**Track: Banking & FinTech (Cred)**

A production-minded domain support agent for Cred's lending-operations team.
Answers loan-policy questions via a RAG pipeline built on a hand-authored
knowledge base, looks up loan application status via a designed
escalation-scoring tool, is orchestrated with LangGraph (conditional
routing, persisted memory, structured outputs, guardrails), evaluated with
the RAG triad, exposed via FastAPI with structured logging, and hardened
with SQLite checkpointing, MCP tool exposure, and timeout/retry resilience.

**Everything in this repository runs under `MOCK_LLM` mode: zero API keys,
zero network calls to any LLM provider.** Embeddings (SentenceTransformers,
local), the vector index (ChromaDB, local), and the MCP server (`fastmcp`,
local) all run entirely offline aside from the one-time SentenceTransformers
model download on first run.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

No `.env` file or API key is required anywhere in this project.

---

## Part 1 — Dataset Design & RAG Core

### Task 1: Dataset design choices (for reproducibility)

Run: `python data/dataset.py`

- **Seed:** `42`
- **Record count:** 60 (exceeds the required minimum of 40)
- **Category weights:** roughly equal across all 5 categories (`[0.22, 0.20,
  0.20, 0.19, 0.19]` for Personal/Home/Auto/Education/Business Loan)
- **Status weights:** `[0.15, 0.30, 0.25, 0.15, 0.15]` for
  Submitted/Under Review/Approved/Rejected/Disbursed — skewed toward the
  middle of a realistic operational funnel
- **loan_amount_inr range:** INR 50,000 – INR 50,00,000. Reasoning: spans
  Cred's real product mix from small personal loans to large home loans, so
  one realistic range can serve every category.
- **flagged_for_fraud_review probability:** drawn at `p=0.18` so the
  expected percentage lands near the middle of the required 10–30% band.
- A deterministic top-up pass guarantees every category has ≥3 records and
  every status has ≥1 record (only reassigns category/status labels — never
  touches the fraud flag, which stays purely random per the brief's rule).

**Measured output (seed=42, reproducible):**
- Total records: 60
- Per category: Personal Loan: 10, Home Loan: 10, Auto Loan: 9, Education
  Loan: 12, Business Loan: 19 (all ≥3 ✓)
- Per status: Submitted: 11, Under Review: 18, Approved: 14, Rejected: 7,
  Disbursed: 10 (all ≥1 ✓)
- Flagged for fraud review: **13.3%** (within the required 10–30% band ✓)

### Task 2: Knowledge base

Run: `python knowledge_base/documents.py`

12 documents, one per required topic (loan eligibility, EMI calculation,
credit-card fees, KYC, fraud disputes, account closure, interest rates,
prepayment penalties, minimum balance, credit score factors, joint
accounts, NRI accounts), each 4 sentences, written in original policy
language.

### Task 3: Chunking strategies + indexing

Run: `python rag/indexer.py`

- **Fixed-size-with-overlap:** 220 characters, 50-character overlap →
  **42 chunks**
- **Sentence-based:** 2 sentences per chunk → **24 chunks**
- Both embedded with `all-MiniLM-L6-v2` (local, free), indexed into two
  separate ChromaDB collections (`kb_fixed_size_chunks`,
  `kb_sentence_chunks`). Embeddings are explicitly normalized
  (`normalize_embeddings=True`) so ChromaDB's L2 distance converts correctly
  to a cosine-similarity-equivalent score.

### Task 4: Grounded generation + empirically calibrated threshold

Run: `python rag/retrieval.py`

Under `MOCK_LLM`, "generation" is purely extractive: the answer IS the
retrieved chunk text, concatenated. This makes the answer grounded by
construction. The "I don't know" fallback threshold was calibrated by
measuring real top-1 cosine similarity, not guessed:

**Measured in-scope query similarities (fixed-size collection):**
| Query | Top-1 similarity |
|---|---|
| What documents do I need for KYC? | 0.530 |
| What is the interest rate for a home loan? | 0.787 |
| How is EMI calculated? | 0.578 |
| What happens if my account falls below the minimum balance? | 0.546 |
| How do I dispute a fraudulent transaction on my card? | 0.488 |

**Measured out-of-scope query similarities:**
| Query | Top-1 similarity |
|---|---|
| What is the weather like in Mumbai today? | 0.130 |
| Can you recommend a good recipe for chicken biryani? | 0.101 |

Clusters are cleanly separated (lowest in-scope = 0.488 > highest
out-of-scope = 0.130).

**Chosen threshold: 0.309** (exact midpoint between the two clusters).

Demonstrated on all 5 in-scope queries above (all correctly grounded) plus
1 out-of-scope query (correctly triggers the fallback).

### Task 5: Chunking strategy comparison (Precision@3 / Recall@3)

Run: `python rag/evaluation.py`

Document-level Precision@3/Recall@3 (chunks mapped to parent doc_id,
deduplicated) on the same 5 queries from Task 4:

| Collection | Precision@3 | Recall@3 |
|---|---|---|
| Fixed-size-with-overlap | **0.700** | 1.000 |
| Sentence-based | 0.500 | 1.000 |

**Recommendation: deploy the fixed-size-with-overlap strategy.** Both
strategies achieve perfect recall (the correct document is always retrieved
somewhere in the top-3), but fixed-size chunking achieves meaningfully
higher precision (0.700 vs 0.500) — its 220-character windows retrieve
fewer off-topic parent documents alongside the correct one, likely because
sentence-based chunks are shorter and semantically thinner, causing more
adjacent/unrelated documents to surface in the top-3.

---

## Part 2 — LangGraph Agent

### Task 6: `check_loan_application_status` + escalation score

Run: `python agent/tools.py`


- **FRAUD_WEIGHT = 0.65:** a fraud flag is the dominant signal (direct
  compliance risk).
- **RECENCY_WEIGHT = 0.35:** staleness is secondary but real — an
  application sitting unresolved for a long time creates operational and
  regulatory risk in its own right.
- **Escalation threshold = 0.2917**, derived empirically: the 80th percentile
  of `days_since_created` in the generated dataset (seed=42) is **25 days**.
  `0.35 × (25/30) = 0.2917`. Any flagged application scores ≥0.65 (always above
  threshold, escalates unconditionally); unflagged applications only
  escalate if they are in the oldest 20% of the pipeline.

### Task 7: LangGraph agent (4 nodes, conditional routing)

Run: `python agent/graph.py`

Nodes: `classify_intent` → (`rag_node` | `loan_status_node`) →
`format_response`. The conditional edge routes on a regex match for record
IDs (`LN\d{4}`) or status-related keywords. Demonstrated firing both routes
on different sample queries (2 policy questions → `rag_node`, 2 status
lookups → `loan_status_node`).

### Task 8: Persisted memory (JSON)

Run: `python agent/memory.py`

Conversation history is persisted to `agent/memory_store/{thread_id}.json`.
Demonstrated: a 2-turn conversation on `demo_thread_multiturn` where the
second turn's context builds on the first, and a separate fresh thread
(`demo_thread_fresh`) that correctly starts with empty history.

### Task 9: Structured output schema

Run: `python agent/schema.py`

JSON Schema (via `jsonschema`) with a `oneOf` across three response shapes:
`policy_answer`, `status_lookup`, `status_lookup_error`. Every
`format_response` output is validated against this schema before being
returned by the FastAPI layer (Task 11).

### Task 10: Guardrails

Run: `python agent/guardrails.py`

- **Input — PII masking:** regex-based masking of PAN (`[A-Z]{5}[0-9]{4}
  [A-Z]`), Aadhaar (12 digits, optionally grouped), and bank account numbers
  (9–18 digit runs). Per the brief, applicant name and income figures are
  out of scope for a keyless masker and are never used with real data.
- **Input — prompt injection:** pattern-matches common override attempts
  ("ignore previous instructions", "reveal your system prompt", etc.).
- **Output — groundedness:** refuses to answer (returns the fallback)
  whenever top-1 retrieval similarity falls below the Task 4 calibrated
  threshold (0.309).

All three guardrails demonstrated firing on deliberate test cases.

---

## Part 3 — Evaluation, Observability & FastAPI Deployment

### Task 11: FastAPI deployment

Run: `python deployment/main.py`, then visit `http://127.0.0.1:8000/docs`

Endpoints: `POST /ask`, `POST /add-document`, `GET /health`. All request/
response bodies use Pydantic models. `/ask` runs the input guardrail first
(blocking on detected injection), then the LangGraph agent, then schema
validation, then persists to memory and logs the request.

### Task 12: Structured logging

Every request produces one JSON-Lines entry in `logs/requests.jsonl` with a
`trace_id`, ISO timestamp, endpoint, masked request text, and duration in
ms. The **same** PII-masking guardrail from Task 10 is applied to the
logged text before it's written, so PAN/Aadhaar/bank-account numbers never
touch disk unmasked — verified with an assertion in
`deployment/logging_utils.py`'s demo.

### Task 13: RAG triad evaluation (15 queries)

Run: `python evaluation/rag_triad_eval.py`

15 queries: 1 per required KB topic (12 topics) + 3 out-of-scope/edge-case
queries (exceeds the required minimum of 2). Under `MOCK_LLM`, a
deterministic keyless judge scores:
- **Context relevance:** top-1 retrieval similarity.
- **Groundedness:** same similarity if the agent answered from context;
  1.0 if it correctly declined (an honest refusal makes no unsupported
  claim, so it isn't penalized).
- **Answer relevance:** cosine similarity between the query embedding and
  the final answer text's embedding.

**Measured averages across all 15 queries:**
- Context Relevance: **0.522**
- Groundedness: **0.695**
- Answer Relevance: **0.499**

(Full per-query breakdown is in `transcripts/part3_transcripts.txt`.)

---

## Part 4 — Resilience & Interoperability

### Task 14: MCP server + client

Terminal 1: `python mcp/server.py` (leave running)
Terminal 2: `python mcp/client.py`

`fastmcp` exposes `check_loan_application_status` (with its docstring) as
an MCP tool over HTTP, mounted at `http://127.0.0.1:8000/mcp`. The client
is a fully separate script/process that connects and calls the tool for 3
different record IDs (LN0001, LN0002, LN0005), printing the standardized
`CallToolResult` for each.

### Task 15: SQLite checkpointing

Run: `python resilience/checkpointing.py`

Uses `langgraph-checkpoint-sqlite`'s `SqliteSaver`, keyed by `thread_id`,
with `interrupt_before=["format_response"]`. Demonstrated on thread
`demo-checkpoint-thread-1`:
1. First `invoke()`: `classify_intent` and `rag_node` execute (printed
   `[NODE EXECUTED]` markers), then the run stops before `format_response`
   (`final_response` is `None`).
2. Resuming the **same** `thread_id` with `invoke(None, config=...)`:
   **only** `format_response` prints `[NODE EXECUTED]` — `classify_intent`
   and `rag_node` do NOT re-execute, proving their results were loaded from
   the SQLite checkpoint (`resilience/checkpoints.sqlite`).

### Task 16: Timeouts and retries

Run: `python resilience/timeouts_retries.py`

- **(a) Retry policy:** `RetryPolicy(initial_interval=0.3, backoff_factor=
  2.0, max_interval=5.0, max_attempts=5, jitter=True)` wraps a node that
  fails on attempts 1–2 (simulated transient failure via a counter) and
  succeeds on attempt 3 — recovered within the configured 5 attempts.
- **(b) Per-node timeout:** a node with `timeout=1.0` sleeps 3.0s;
  LangGraph raises a clean `NodeTimeoutError` at ~1.0s elapsed (not a hang).
  Per LangGraph's design, node timeouts require async node functions.
- **(c) Global timeout:** two 1.5s nodes (3.0s total) wrapped in
  `asyncio.wait_for(..., timeout=2.0)` — the whole run is cleanly cancelled
  with `asyncio.TimeoutError` before completion.

---

## Repository Structure