"""
Task 1 (Part 1): Loan Application Dataset Generator
-----------------------------------------------------
Generates a seeded, deterministic list of loan applications for the
Cred Domain Support Agent capstone.

Design choices (documented in README.md too):
- SEED = 42 for full reproducibility.
- NUM_RECORDS = 60 (comfortably above the required 40).
- CATEGORY_WEIGHTS: roughly equal weighting across all 5 categories so each
  category naturally clears the "at least 3 records" bar even with randomness.
- STATUS_WEIGHTS: skewed slightly toward 'Under Review' and 'Approved' to
  mimic a realistic operational funnel (most applications are mid-pipeline),
  while still guaranteeing every status appears at least once.
- loan_amount_inr range: INR 50,000 to INR 50,00,000 (50 thousand to 50 lakh).
  Reasoning: this spans Cred's real product mix from small personal loans
  to large home loans, so a single realistic range can serve every category.
- flagged_for_fraud_review: drawn with p=0.18 so the expected flagged
  percentage lands near the middle of the required 10-30% band.
"""

import random

SEED = 42
NUM_RECORDS = 60

CATEGORIES = ["Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Business Loan"]
STATUSES = ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]

CATEGORY_WEIGHTS = [0.22, 0.20, 0.20, 0.19, 0.19]
STATUS_WEIGHTS = [0.15, 0.30, 0.25, 0.15, 0.15]

FRAUD_FLAG_PROBABILITY = 0.18

MIN_LOAN_AMOUNT_INR = 50_000
MAX_LOAN_AMOUNT_INR = 5_000_000


def generate_dataset(seed: int = SEED, num_records: int = NUM_RECORDS):
    rng = random.Random(seed)

    records = []
    for i in range(1, num_records + 1):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        loan_amount_inr = rng.randint(MIN_LOAN_AMOUNT_INR, MAX_LOAN_AMOUNT_INR)
        # round to nearest 1000 for realism
        loan_amount_inr = round(loan_amount_inr / 1000) * 1000
        days_since_created = rng.randint(0, 30)
        flagged_for_fraud_review = rng.random() < FRAUD_FLAG_PROBABILITY

        record = {
            "record_id": f"LN{i:04d}",
            "category": category,
            "status": status,
            "loan_amount_inr": loan_amount_inr,
            "days_since_created": days_since_created,
            "flagged_for_fraud_review": flagged_for_fraud_review,
        }
        records.append(record)

    # Deterministic top-up pass: guarantees every category has >=3 records
    # and every status has >=1 record. This only reassigns category/status
    # labels (never touches the fraud flag, which must stay purely random
    # per the brief's rule against hand-tuning that specific field).
    _ensure_category_coverage(records, rng)
    _ensure_status_coverage(records, rng)

    return records


def _ensure_category_coverage(records, rng, min_count=3):
    from collections import Counter
    counts = Counter(r["category"] for r in records)
    for cat in CATEGORIES:
        while counts[cat] < min_count:
            donor_cat = counts.most_common(1)[0][0]
            for r in records:
                if r["category"] == donor_cat:
                    r["category"] = cat
                    counts[donor_cat] -= 1
                    counts[cat] += 1
                    break


def _ensure_status_coverage(records, rng, min_count=1):
    from collections import Counter
    counts = Counter(r["status"] for r in records)
    for st in STATUSES:
        while counts[st] < min_count:
            donor_st = counts.most_common(1)[0][0]
            for r in records:
                if r["status"] == donor_st:
                    r["status"] = st
                    counts[donor_st] -= 1
                    counts[st] += 1
                    break


LOAN_APPLICATIONS = generate_dataset()


def report_dataset_stats(records):
    from collections import Counter

    cat_counts = Counter(r["category"] for r in records)
    status_counts = Counter(r["status"] for r in records)
    flagged_count = sum(1 for r in records if r["flagged_for_fraud_review"])
    flagged_pct = (flagged_count / len(records)) * 100

    print(f"Total records: {len(records)}")
    print("\nCount per category:")
    for cat in CATEGORIES:
        print(f"  {cat}: {cat_counts[cat]}")

    print("\nCount per status:")
    for st in STATUSES:
        print(f"  {st}: {status_counts[st]}")

    print(f"\nFlagged for fraud review: {flagged_count}/{len(records)} = {flagged_pct:.1f}%")
    assert 10 <= flagged_pct <= 30, "flagged_for_fraud_review percentage out of required 10-30% band"
    for cat in CATEGORIES:
        assert cat_counts[cat] >= 3, f"{cat} has fewer than 3 records"
    for st in STATUSES:
        assert status_counts[st] >= 1, f"{st} has zero records"
    print("\nAll structural thresholds satisfied.")


def get_record_by_id(record_id: str):
    """Look up a single loan application record by its record_id."""
    for r in LOAN_APPLICATIONS:
        if r["record_id"] == record_id:
            return r
    return None


if __name__ == "__main__":
    report_dataset_stats(LOAN_APPLICATIONS)
    print("\nSample records:")
    for r in LOAN_APPLICATIONS[:3]:
        print(r)