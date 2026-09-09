"""
Task 2 (Part 1): Knowledge Base Documents
-------------------------------------------
12+ policy documents covering every required topic for the Cred Domain
Support Agent. Each document is 2-5 sentences, written in plain policy
language, and will be chunked + embedded in rag/chunking.py and
rag/indexer.py.
"""

KNOWLEDGE_BASE_DOCUMENTS = [
    {
        "doc_id": "KB001",
        "topic": "loan_eligibility_criteria",
        "title": "Loan Eligibility Criteria by Loan Type",
        "text": (
            "Personal loan applicants must be salaried or self-employed individuals "
            "aged 21 to 58 with a minimum monthly income of INR 25,000. Home loan "
            "eligibility requires a stable income history of at least two years and "
            "a loan-to-value ratio not exceeding 80% of the property's assessed "
            "value. Auto loan applicants need a minimum credit score of 650, while "
            "education loans require either a co-applicant or collateral for amounts "
            "above INR 750,000. Business loans require at least two years of "
            "audited financial statements and a positive operating cash flow in the "
            "most recent fiscal year."
        ),
    },
    {
        "doc_id": "KB002",
        "topic": "emi_calculation_rules",
        "title": "EMI Calculation Rules",
        "text": (
            "The Equated Monthly Installment is calculated using the reducing "
            "balance method, where interest is charged only on the outstanding "
            "principal for each period. The standard formula applied is "
            "EMI = P x r x (1+r)^n / ((1+r)^n - 1), where P is principal, r is the "
            "monthly interest rate, and n is the tenure in months. EMIs are debited "
            "on the 5th of every month, and any change to the repayment date must "
            "be requested at least 10 working days in advance. Partial prepayments "
            "reduce either the tenure or the EMI amount, depending on the "
            "borrower's stated preference at the time of prepayment."
        ),
    },
    {
        "doc_id": "KB003",
        "topic": "credit_card_fee_structure",
        "title": "Credit Card Fee Structure",
        "text": (
            "Annual fees range from INR 500 for entry-level cards to INR 10,000 for "
            "premium cards, and are typically waived if annual spend crosses a "
            "specified threshold. A late payment fee of up to INR 1,300 applies "
            "when the minimum amount due is not paid by the due date, in addition "
            "to applicable finance charges of up to 3.5% per month on the "
            "outstanding balance. Cash withdrawal via credit card attracts a fee of "
            "2.5% of the withdrawn amount, subject to a minimum of INR 500. "
            "Foreign currency transactions carry a currency conversion markup of "
            "3.5% on the transaction value."
        ),
    },
    {
        "doc_id": "KB004",
        "topic": "kyc_document_requirements",
        "title": "KYC Document Requirements",
        "text": (
            "Every applicant must submit one valid government-issued photo "
            "identity proof, such as a PAN card, Aadhaar card, passport, or "
            "voter ID. Address proof documents accepted include utility bills "
            "not older than three months, a rental agreement, or the Aadhaar "
            "card if the address is current. Self-employed applicants must "
            "additionally provide proof of business registration and the last "
            "two years of income tax returns. All submitted documents undergo "
            "verification against the source database within two working days "
            "before the application can move to underwriting."
        ),
    },
    {
        "doc_id": "KB005",
        "topic": "fraud_dispute_resolution_process",
        "title": "Fraud Dispute Resolution Process",
        "text": (
            "Members who notice an unauthorized transaction must report it "
            "within three days of the transaction date to be eligible for "
            "zero-liability protection. Once reported, the disputed amount is "
            "provisionally credited back within five working days pending "
            "investigation. The investigation team reviews transaction logs, "
            "device fingerprints, and merchant confirmation, and communicates a "
            "final resolution within 45 days as per regulatory guidelines. If "
            "fraud is confirmed, the provisional credit is made permanent and the "
            "affected card or account is blocked and reissued."
        ),
    },
    {
        "doc_id": "KB006",
        "topic": "account_closure_process",
        "title": "Account Closure Process",
        "text": (
            "Account closure requests can be submitted through the app, by "
            "written letter, or by visiting a branch, and require the account to "
            "have a zero outstanding balance. Any pending EMIs, card dues, or "
            "linked standing instructions must be cleared or transferred before "
            "closure can be processed. Closure requests are typically completed "
            "within 7 working days, after which a closure confirmation is sent to "
            "the registered email and phone number. Accounts with a fraud flag or "
            "an active dispute cannot be closed until the matter is resolved."
        ),
    },
    {
        "doc_id": "KB007",
        "topic": "interest_rate_slabs",
        "title": "Interest Rate Slabs",
        "text": (
            "Personal loan interest rates range from 10.5% to 24% per annum "
            "depending on the applicant's credit score and income stability. "
            "Home loans are offered at 8.5% to 11% per annum, with lower rates "
            "reserved for applicants with a credit score above 750. Auto loans "
            "carry rates between 9% and 14%, while business loans range from "
            "11% to 18% based on the business's financial track record. Rates "
            "are reviewed quarterly and existing floating-rate borrowers are "
            "notified of any revision at least 15 days in advance."
        ),
    },
    {
        "doc_id": "KB008",
        "topic": "prepayment_penalty_rules",
        "title": "Prepayment Penalty Rules",
        "text": (
            "Floating-rate personal and home loans carry no prepayment penalty "
            "for individual borrowers, in line with regulatory guidance. "
            "Fixed-rate loans attract a prepayment penalty of up to 2% of the "
            "outstanding principal if repaid within the first 12 months, "
            "reducing to 1% thereafter. Business loans, regardless of rate type, "
            "carry a standard prepayment charge of 2% on the prepaid amount. "
            "Partial prepayments are permitted a maximum of four times per "
            "financial year without additional documentation."
        ),
    },
    {
        "doc_id": "KB009",
        "topic": "minimum_balance_requirements",
        "title": "Minimum Balance Requirements",
        "text": (
            "Regular savings accounts must maintain an average monthly balance "
            "of INR 10,000 in metro cities and INR 5,000 in non-metro locations. "
            "Falling below the required balance attracts a penalty ranging from "
            "INR 100 to INR 600 depending on the shortfall percentage. Salary "
            "accounts and accounts linked to an active home loan are exempt from "
            "minimum balance requirements for as long as the linkage remains "
            "active. Senior citizen accounts carry a reduced minimum balance "
            "requirement of INR 2,500 across all locations."
        ),
    },
    {
        "doc_id": "KB010",
        "topic": "credit_score_impact_factors",
        "title": "Credit Score Impact Factors",
        "text": (
            "Payment history is the single largest factor affecting credit "
            "score, and even one missed payment can lower a score by 50 to 100 "
            "points. Credit utilization, meaning the proportion of available "
            "credit limit currently in use, should ideally stay below 30% to "
            "maintain a healthy score. The length of credit history, the mix of "
            "credit types held, and the number of recent hard inquiries also "
            "contribute meaningfully to the overall score calculation. Closing "
            "old credit accounts can shorten credit history length and may "
            "cause a temporary dip in the score."
        ),
    },
    {
        "doc_id": "KB011",
        "topic": "joint_account_rules",
        "title": "Joint Account Rules",
        "text": (
            "Joint accounts can be opened by up to four individuals, with the "
            "operating mode chosen as either 'either or survivor', 'jointly', or "
            "'former or survivor' at the time of account opening. All KYC "
            "documents must be submitted independently by each joint holder "
            "before the account is activated. For joint loan applications, the "
            "credit assessment considers the combined income and credit history "
            "of all applicants, and all joint holders are equally liable for "
            "repayment regardless of who initiated the loan. Changing the "
            "operating mode after account opening requires consent from all "
            "existing holders."
        ),
    },
    {
        "doc_id": "KB012",
        "topic": "nri_account_eligibility",
        "title": "NRI Account Eligibility",
        "text": (
            "Non-Resident Indians are eligible to open NRE or NRO accounts by "
            "submitting a valid passport, visa or work permit, and proof of "
            "overseas address in addition to standard KYC documents. NRE "
            "accounts hold foreign earnings and are fully repatriable, while NRO "
            "accounts are meant for income earned within India and carry "
            "restrictions on repatriation. NRI loan applications are assessed "
            "against a modified eligibility framework that accounts for foreign "
            "income stability and currency risk. A Power of Attorney holder in "
            "India may be authorized to operate the account on the NRI's behalf, "
            "subject to submission of a notarized POA document."
        ),
    },
]


def get_document_by_id(doc_id: str):
    for doc in KNOWLEDGE_BASE_DOCUMENTS:
        if doc["doc_id"] == doc_id:
            return doc
    return None


if __name__ == "__main__":
    print(f"Total KB documents: {len(KNOWLEDGE_BASE_DOCUMENTS)}")
    for doc in KNOWLEDGE_BASE_DOCUMENTS:
        sentence_count = doc["text"].count(". ") + 1
        print(f"  {doc['doc_id']} | {doc['topic']} | ~{sentence_count} sentences")