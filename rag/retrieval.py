"""
Task 4 (Part 1): Grounded Generation with Empirically Calibrated Threshold
------------------------------------------------------------------------------
Under MOCK_LLM there is no real language model to judge groundedness, so
retrieval similarity is the only signal available. This module:

1. Measures top-1 cosine similarity for a set of known in-scope queries and
   a set of known out-of-scope queries (run once, printed for the record).
2. Picks an "I don't know" threshold that sits between the two observed
   clusters (NOT a hardcoded tutorial default like 0.5/0.6/0.7).
3. Uses that threshold to decide, for any new query, whether to answer from
   retrieved context or trigger the fallback.

Under MOCK_LLM (no API keys, no network calls to an LLM), "generation" means:
concatenate the retrieved top-k chunk texts into the answer, since that is
the only content-generation mechanism available without a real model. This
is deliberately extractive rather than abstractive -- it is grounded by
construction, because it contains nothing except retrieved text.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.indexer import build_and_index_all, query_collection

# ---- Calibration query sets (Task 4 requirement: >=3 in-scope, >=2 out-of-scope) ----
CALIBRATION_IN_SCOPE_QUERIES = [
    "What documents do I need for KYC?",
    "What is the interest rate for a home loan?",
    "How is EMI calculated?",
    "What happens if my account falls below the minimum balance?",
    "How do I dispute a fraudulent transaction on my card?",
]

CALIBRATION_OUT_OF_SCOPE_QUERIES = [
    "What is the weather like in Mumbai today?",
    "Can you recommend a good recipe for chicken biryani?",
]

FALLBACK_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that "
    "confidently. Could you rephrase, or ask about a Cred policy topic such "
    "as loan eligibility, EMI, KYC, fraud disputes, or account rules?"
)


def measure_top1_similarities(collection, queries: list) -> list:
    """Returns list of (query, top1_similarity) for each query."""
    results = []
    for q in queries:
        hits = query_collection(collection, q, top_k=1)
        top1_sim = hits[0]["similarity"] if hits else 0.0
        results.append((q, top1_sim))
    return results


def calibrate_threshold(collection):
    """
    Empirically measures top-1 similarity for known in-scope and
    out-of-scope queries, prints the results, and returns a threshold set
    at the midpoint between the lowest in-scope score and the highest
    out-of-scope score (falls back to midpoint-of-averages if the clusters
    overlap, so the number is always derived from real observations).
    """
    print("Calibrating 'I don't know' threshold empirically...\n")

    in_scope_results = measure_top1_similarities(collection, CALIBRATION_IN_SCOPE_QUERIES)
    out_scope_results = measure_top1_similarities(collection, CALIBRATION_OUT_OF_SCOPE_QUERIES)

    print("In-scope queries (should retrieve relevant policy content):")
    for q, sim in in_scope_results:
        print(f"  [{sim:.3f}] {q}")

    print("\nOut-of-scope queries (should NOT match anything relevant):")
    for q, sim in out_scope_results:
        print(f"  [{sim:.3f}] {q}")

    in_scope_scores = [sim for _, sim in in_scope_results]
    out_scope_scores = [sim for _, sim in out_scope_results]

    min_in_scope = min(in_scope_scores)
    max_out_scope = max(out_scope_scores)

    if min_in_scope > max_out_scope:
        threshold = (min_in_scope + max_out_scope) / 2
        print(f"\nClusters are cleanly separated (lowest in-scope={min_in_scope:.3f} "
              f"> highest out-of-scope={max_out_scope:.3f}).")
    else:
        avg_in_scope = sum(in_scope_scores) / len(in_scope_scores)
        avg_out_scope = sum(out_scope_scores) / len(out_scope_scores)
        threshold = (avg_in_scope + avg_out_scope) / 2
        print(f"\nWARNING: clusters overlap (lowest in-scope={min_in_scope:.3f} <= "
              f"highest out-of-scope={max_out_scope:.3f}). Using midpoint of "
              f"averages instead: avg_in_scope={avg_in_scope:.3f}, "
              f"avg_out_scope={avg_out_scope:.3f}.")

    print(f"\n==> Calibrated threshold: {threshold:.3f}\n")
    return threshold


def grounded_answer(collection, query: str, threshold: float, top_k: int = 3) -> dict:
    """
    Retrieves top-k chunks and, under MOCK_LLM, "generates" an answer by
    stitching together ONLY the retrieved chunk text. If the top-1 similarity
    is below the calibrated threshold, returns the fallback instead of
    fabricating an answer.
    """
    hits = query_collection(collection, query, top_k=top_k)

    if not hits or hits[0]["similarity"] < threshold:
        return {
            "query": query,
            "answer": FALLBACK_MESSAGE,
            "grounded": False,
            "top1_similarity": hits[0]["similarity"] if hits else 0.0,
            "sources": [],
        }

    context_snippets = [h["text"] for h in hits]
    answer = " ".join(context_snippets)

    return {
        "query": query,
        "answer": answer,
        "grounded": True,
        "top1_similarity": hits[0]["similarity"],
        "sources": [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"]} for h in hits],
    }


if __name__ == "__main__":
    print("Indexing knowledge base (fixed-size + sentence-based collections)...\n")
    fixed_collection, sentence_collection = build_and_index_all()

    threshold = calibrate_threshold(fixed_collection)

    print("=" * 70)
    print("DEMONSTRATION: >=5 in-scope queries + 1 deliberately out-of-scope query")
    print("=" * 70)

    demo_queries = CALIBRATION_IN_SCOPE_QUERIES + [CALIBRATION_OUT_OF_SCOPE_QUERIES[0]]

    for q in demo_queries:
        result = grounded_answer(fixed_collection, q, threshold=threshold)
        print(f"\nQuery: {result['query']}")
        print(f"Top-1 similarity: {result['top1_similarity']:.3f} | Grounded: {result['grounded']}")
        print(f"Answer: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
        if result["sources"]:
            print(f"Sources: {result['sources']}")