"""
Task 13 (Part 3): RAG Triad Evaluation at Scale (15 queries, MOCK_LLM)
------------------------------------------------------------------------------
Scores context relevance, groundedness, and answer relevance for 15 test
queries -- at least 1 per required KB topic (12 topics), plus 3 deliberately
out-of-scope / edge-case queries (exceeds the required minimum of 2).

Under MOCK_LLM there is no real LLM to act as judge, so this module
implements a DETERMINISTIC, keyless judge built entirely from signals we
already have (embedding similarity), documented explicitly so the scoring
method is transparent and reproducible:

  - context_relevance : top-1 cosine similarity between the query and the
                         best-matching retrieved chunk. Measures whether the
                         retrieval step found relevant material at all.

  - groundedness       : if the agent answered from context (grounded=True),
                          this equals the same top-1 similarity, because our
                          MOCK_LLM "generation" is purely extractive -- the
                          answer IS the retrieved text, so its groundedness
                          is bounded by how well that text matches the query.
                          If the agent instead triggered the "I don't know"
                          fallback (grounded=False), we score groundedness as
                          1.0: the agent made no unsupported claims, which is
                          the CORRECT behavior for an ungrounded question, so
                          it should not be penalized for refusing honestly.

  - answer_relevance   : cosine similarity between the embedding of the
                          FINAL ANSWER TEXT and the embedding of the query.
                          Measures whether the answer actually addresses
                          what was asked, independent of retrieval quality.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rag.indexer import build_and_index_all, get_embedding_model
from rag.retrieval import calibrate_threshold, grounded_answer

# 15 test queries: >=1 per required KB topic (12 topics) + >=2 out-of-scope/edge-case (we use 3)
TEST_QUERIES = [
    {"query": "What are the eligibility criteria for a personal loan?", "topic": "loan_eligibility_criteria"},
    {"query": "How is EMI calculated?", "topic": "emi_calculation_rules"},
    {"query": "What is the annual fee structure for credit cards?", "topic": "credit_card_fee_structure"},
    {"query": "What documents do I need for KYC?", "topic": "kyc_document_requirements"},
    {"query": "How do I dispute a fraudulent transaction on my card?", "topic": "fraud_dispute_resolution_process"},
    {"query": "What is the process to close my account?", "topic": "account_closure_process"},
    {"query": "What is the interest rate for a home loan?", "topic": "interest_rate_slabs"},
    {"query": "Is there a penalty for prepaying my loan early?", "topic": "prepayment_penalty_rules"},
    {"query": "What happens if my account falls below the minimum balance?", "topic": "minimum_balance_requirements"},
    {"query": "What factors affect my credit score?", "topic": "credit_score_impact_factors"},
    {"query": "What are the rules for opening a joint account?", "topic": "joint_account_rules"},
    {"query": "Am I eligible to open an NRI account?", "topic": "nri_account_eligibility"},
    {"query": "What is the weather like in Mumbai today?", "topic": "OUT_OF_SCOPE"},
    {"query": "Can you recommend a good recipe for chicken biryani?", "topic": "OUT_OF_SCOPE"},
    {"query": "asdkj qlwkejasd 12345 random gibberish", "topic": "EDGE_CASE_GIBBERISH"},
]


def _cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def judge_context_relevance(top1_similarity: float) -> float:
    return round(max(0.0, min(top1_similarity, 1.0)), 3)


def judge_groundedness(grounded: bool, top1_similarity: float) -> float:
    if grounded:
        return round(max(0.0, min(top1_similarity, 1.0)), 3)
    return 1.0  # honest, correctly-declined answer is fully "grounded" (makes no unsupported claim)


def judge_answer_relevance(query: str, answer: str, model) -> float:
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    answer_emb = model.encode([answer], normalize_embeddings=True)[0]
    sim = _cosine_similarity(query_emb, answer_emb)
    return round(max(0.0, min(sim, 1.0)), 3)


def evaluate_all(collection, threshold: float):
    model = get_embedding_model()
    results = []

    for item in TEST_QUERIES:
        query = item["query"]
        response = grounded_answer(collection, query, threshold=threshold)

        context_rel = judge_context_relevance(response["top1_similarity"])
        grounded_score = judge_groundedness(response["grounded"], response["top1_similarity"])
        answer_rel = judge_answer_relevance(query, response["answer"], model)

        results.append({
            "query": query,
            "topic": item["topic"],
            "grounded": response["grounded"],
            "context_relevance": context_rel,
            "groundedness": grounded_score,
            "answer_relevance": answer_rel,
        })

    return results


def print_report(results):
    print(f"{'Query':<55} {'Ctx.Rel':>8} {'Ground':>8} {'Ans.Rel':>8}")
    print("-" * 82)
    for r in results:
        q_display = (r["query"][:52] + "...") if len(r["query"]) > 52 else r["query"]
        print(f"{q_display:<55} {r['context_relevance']:>8.3f} {r['groundedness']:>8.3f} {r['answer_relevance']:>8.3f}")

    avg_context = sum(r["context_relevance"] for r in results) / len(results)
    avg_ground = sum(r["groundedness"] for r in results) / len(results)
    avg_answer = sum(r["answer_relevance"] for r in results) / len(results)

    print("-" * 82)
    print(f"{'AVERAGES':<55} {avg_context:>8.3f} {avg_ground:>8.3f} {avg_answer:>8.3f}")

    return {"avg_context_relevance": avg_context, "avg_groundedness": avg_ground, "avg_answer_relevance": avg_answer}


if __name__ == "__main__":
    print("Indexing knowledge base...")
    fixed_collection, _ = build_and_index_all()
    threshold = calibrate_threshold(fixed_collection)

    print(f"\nRunning RAG triad evaluation on {len(TEST_QUERIES)} queries "
          f"(covers all 12 required topics + 3 out-of-scope/edge-case queries)...\n")

    results = evaluate_all(fixed_collection, threshold)
    averages = print_report(results)

    print(f"\nFinal averages across all {len(TEST_QUERIES)} queries:")
    print(f"  Context Relevance : {averages['avg_context_relevance']:.3f}")
    print(f"  Groundedness      : {averages['avg_groundedness']:.3f}")
    print(f"  Answer Relevance  : {averages['avg_answer_relevance']:.3f}")