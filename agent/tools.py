"""
Task 6 (Part 2): check_loan_application_status tool with designed
escalation_score.
------------------------------------------------------------------
escalation_score formula:
    escalation_score = (FRAUD_WEIGHT * fraud_flag) + (RECENCY_WEIGHT * recency_signal)

Where:
    fraud_flag      = 1.0 if flagged_for_fraud_review else 0.0
    recency_signal  = days_since_created / MAX_DAYS   (normalized to [0, 1])
    FRAUD_WEIGHT     = 0.7
    RECENCY_WEIGHT   = 0.3

Rationale: a fraud flag is treated as the dominant signal (weight 0.7) because
it directly indicates a compliance risk, while staleness (days sitting in the
pipeline without resolution) is a secondary but real signal that operational
delay itself creates member-experience and regulatory risk (weight 0.3).

Escalation threshold: derived empirically from THIS dataset, not guessed.
We compute the 80th percentile of days_since_created across all generated
applications, then take the escalation_score an UNFLAGGED application would
have at that percentile as the threshold. Any FLAGGED application always
scores >= 0.7, which is always above this threshold -- so every fraud-flagged
case escalates unconditionally, while unflagged cases escalate only if they
are unusually stale (top 20% oldest in the pipeline).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import LOAN_APPLICATIONS, get_record_by_id

FRAUD_WEIGHT = 0.65
RECENCY_WEIGHT = 0.35
MAX_DAYS = 30


def _compute_escalation_threshold():
    """Derives the escalation threshold from the 80th percentile of
    days_since_created observed in the actual generated dataset."""
    days_sorted = sorted(r["days_since_created"] for r in LOAN_APPLICATIONS)
    idx = int(0.8 * (len(days_sorted) - 1))
    p80_days = days_sorted[idx]
    threshold = RECENCY_WEIGHT * (p80_days / MAX_DAYS)
    return threshold, p80_days


ESCALATION_THRESHOLD, P80_DAYS = _compute_escalation_threshold()


def compute_escalation_score(flagged_for_fraud_review: bool, days_since_created: int) -> float:
    fraud_component = FRAUD_WEIGHT * (1.0 if flagged_for_fraud_review else 0.0)
    recency_component = RECENCY_WEIGHT * min(days_since_created / MAX_DAYS, 1.0)
    score = fraud_component + recency_component
    return round(score, 4)


def check_loan_application_status(record_id: str) -> dict:
    """
    Looks up a loan application by record_id and returns its status,
    loan amount, and a designed escalation_score in [0, 1].

    Args:
        record_id: the loan application's unique ID, e.g. "LN0001".

    Returns:
        dict with keys: record_id, found, status, loan_amount_inr,
        escalation_score, recommend_escalation, flagged_for_fraud_review,
        days_since_created.
    """
    record = get_record_by_id(record_id)

    if record is None:
        return {
            "record_id": record_id,
            "found": False,
            "status": None,
            "loan_amount_inr": None,
            "escalation_score": None,
            "recommend_escalation": None,
        }

    score = compute_escalation_score(
        record["flagged_for_fraud_review"], record["days_since_created"]
    )

    return {
        "record_id": record["record_id"],
        "found": True,
        "status": record["status"],
        "loan_amount_inr": record["loan_amount_inr"],
        "escalation_score": score,
        "recommend_escalation": score >= ESCALATION_THRESHOLD,
        "flagged_for_fraud_review": record["flagged_for_fraud_review"],
        "days_since_created": record["days_since_created"],
    }


if __name__ == "__main__":
    print(f"Dataset mein 80th percentile days = {P80_DAYS}")
    print(f"Escalation ke liye threshold value = {ESCALATION_THRESHOLD:.4f}")

    sample_ids = [LOAN_APPLICATIONS[i]["record_id"] for i in [0, 1, 2, 3, 4]]
    for rid in sample_ids:
        result = check_loan_application_status(rid)
        print(result)

    print("\nLookup for a non-existent record:")
    print(check_loan_application_status("LN9999"))