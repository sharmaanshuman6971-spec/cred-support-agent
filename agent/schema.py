"""
Task 9 (Part 2): Structured Output Schema
------------------------------------------------
Defines a JSON Schema that every agent response (the `final_response` dict
produced by agent/graph.py's format_response node) must conform to, and
validates responses against it in code using the `jsonschema` library.

Two response shapes are supported (policy answers vs status lookups), so the
schema uses oneOf to allow either valid shape while still rejecting anything
malformed.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jsonschema import validate, ValidationError

POLICY_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "policy_answer"},
        "query": {"type": "string"},
        "answer": {"type": "string"},
        "grounded": {"type": "boolean"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "chunk_id": {"type": "string"},
                },
                "required": ["doc_id", "chunk_id"],
            },
        },
    },
    "required": ["type", "query", "answer", "grounded", "sources"],
    "additionalProperties": False,
}

STATUS_LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "status_lookup"},
        "query": {"type": "string"},
        "record_id": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"],
        },
        "loan_amount_inr": {"type": "number"},
        "escalation_score": {"type": "number", "minimum": 0, "maximum": 1},
        "recommend_escalation": {"type": "boolean"},
    },
    "required": ["type", "query", "record_id", "status", "loan_amount_inr",
                 "escalation_score", "recommend_escalation"],
    "additionalProperties": False,
}

STATUS_LOOKUP_ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "status_lookup_error"},
        "query": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["type", "query", "answer"],
    "additionalProperties": False,
}

AGENT_RESPONSE_SCHEMA = {
    "oneOf": [POLICY_ANSWER_SCHEMA, STATUS_LOOKUP_SCHEMA, STATUS_LOOKUP_ERROR_SCHEMA],
}


def validate_agent_response(response: dict) -> dict:
    """
    Validates a response dict against AGENT_RESPONSE_SCHEMA.
    Returns {"valid": True} on success, or {"valid": False, "error": "..."}.
    """
    try:
        validate(instance=response, schema=AGENT_RESPONSE_SCHEMA)
        return {"valid": True, "error": None}
    except ValidationError as e:
        return {"valid": False, "error": str(e.message)}


if __name__ == "__main__":
    print("Test 1: valid policy_answer response")
    valid_policy = {
        "type": "policy_answer",
        "query": "What is the interest rate for a home loan?",
        "answer": "Home loans: 8.5%-11% p.a.",
        "grounded": True,
        "sources": [{"doc_id": "KB007", "chunk_id": "KB007_fixed_0"}],
    }
    print(validate_agent_response(valid_policy))

    print("\nTest 2: valid status_lookup response")
    valid_status = {
        "type": "status_lookup",
        "query": "Check status for LN0002",
        "record_id": "LN0002",
        "status": "Approved",
        "loan_amount_inr": 4625000,
        "escalation_score": 0.02,
        "recommend_escalation": False,
    }
    print(validate_agent_response(valid_status))

    print("\nTest 3: INVALID response (missing required field 'answer')")
    invalid_response = {
        "type": "policy_answer",
        "query": "What is the interest rate for a home loan?",
        "grounded": True,
        "sources": [],
    }
    print(validate_agent_response(invalid_response))

    print("\nTest 4: INVALID response (bad status enum value)")
    invalid_status = {
        "type": "status_lookup",
        "query": "Check status for LN0002",
        "record_id": "LN0002",
        "status": "InvalidStatusValue",
        "loan_amount_inr": 4625000,
        "escalation_score": 0.02,
        "recommend_escalation": False,
    }
    print(validate_agent_response(invalid_status))