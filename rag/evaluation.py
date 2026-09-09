"""
Task 5 (Part 1): Evaluate and Compare Both Chunking Strategies
--------------------------------------------------------------------
Computes Precision@3 and Recall@3 at the DOCUMENT level (chunks mapped back
to their parent doc_id, deduplicated) for the same 5 queries used in Task 4,
separately for the fixed-size collection and the sentence-based collection.

Ground truth (author-defined relevance judgments):
Each evaluation query is written around a single, unambiguous target KB
document -- the one document whose topic directly answers the question.
This keeps precision/recall arithmetic simple and auditable.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.indexer import build_and_index_all, query_collection

# Query -> set of relevant doc_ids (ground truth, author-defined)
EVAL_QUERIES = [
    {"query": "What documents do I need for KYC?", "relevant_doc_ids": {"KB004"}},
    {"query": "What is the interest rate for a home loan?", "relevant_doc_ids": {"KB007"}},
    {"query": "How is EMI calculated?", "relevant_doc_ids": {"KB002"}},
    {"query": "What happens if my account falls below the minimum balance?", "relevant_doc_ids": {"KB009"}},
    {"query": "How do I dispute a fraudulent transaction on my card?", "relevant_doc_ids": {"KB005"}},
]

TOP_K = 3


def precision_recall_at_k(collection, query: str, relevant_doc_ids: set, k: int = TOP_K):
    """
    Retrieves top-k chunks, maps them back to parent doc_ids, dedups, then
    computes Precision@k and Recall@k at the document level.
    """
    hits = query_collection(collection, query, top_k=k)

    seen = []
    for h in hits:
        if h["doc_id"] not in seen:
            seen.append(h["doc_id"])

    retrieved_doc_ids = set(seen)
    relevant_retrieved = retrieved_doc_ids & relevant_doc_ids

    precision = len(relevant_retrieved) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
    recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0.0

    return {
        "query": query,
        "retrieved_doc_ids": list(retrieved_doc_ids),
        "relevant_doc_ids": list(relevant_doc_ids),
        "relevant_retrieved": list(relevant_retrieved),
        "precision_at_k": precision,
        "recall_at_k": recall,
    }


def evaluate_collection(collection, collection_label: str):
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {collection_label}")
    print(f"{'=' * 70}")

    all_results = []
    for item in EVAL_QUERIES:
        result = precision_recall_at_k(collection, item["query"], item["relevant_doc_ids"])
        all_results.append(result)

        print(f"\nQuery: {result['query']}")
        print(f"  Retrieved (deduped) doc_ids: {result['retrieved_doc_ids']}")
        print(f"  Relevant doc_ids (ground truth): {result['relevant_doc_ids']}")
        print(f"  Relevant retrieved: {result['relevant_retrieved']}")
        print(f"  Precision@3 = {len(result['relevant_retrieved'])}/{len(result['retrieved_doc_ids'])} "
              f"= {result['precision_at_k']:.3f}")
        print(f"  Recall@3 = {len(result['relevant_retrieved'])}/{len(result['relevant_doc_ids'])} "
              f"= {result['recall_at_k']:.3f}")

    avg_precision = sum(r["precision_at_k"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall_at_k"] for r in all_results) / len(all_results)

    print(f"\n{collection_label} AVERAGES -> Precision@3: {avg_precision:.3f} | Recall@3: {avg_recall:.3f}")

    return {"label": collection_label, "avg_precision": avg_precision, "avg_recall": avg_recall, "results": all_results}


if __name__ == "__main__":
    print("Indexing knowledge base (fixed-size + sentence-based collections)...")
    fixed_collection, sentence_collection = build_and_index_all()

    fixed_summary = evaluate_collection(fixed_collection, "FIXED-SIZE-WITH-OVERLAP collection")
    sentence_summary = evaluate_collection(sentence_collection, "SENTENCE-BASED collection")

    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"Fixed-size    -> Precision@3: {fixed_summary['avg_precision']:.3f} | Recall@3: {fixed_summary['avg_recall']:.3f}")
    print(f"Sentence-based -> Precision@3: {sentence_summary['avg_precision']:.3f} | Recall@3: {sentence_summary['avg_recall']:.3f}")

    if fixed_summary["avg_precision"] + fixed_summary["avg_recall"] >= \
       sentence_summary["avg_precision"] + sentence_summary["avg_recall"]:
        recommended = "fixed-size-with-overlap"
    else:
        recommended = "sentence-based"

    print(f"\nRecommendation: deploy the '{recommended}' strategy, based on the "
          f"precision/recall numbers measured above.")