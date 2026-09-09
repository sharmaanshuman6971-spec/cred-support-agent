"""
Task 12 (Part 3): Structured Logging
------------------------------------------
Logs every request as one JSON-Lines entry with a trace ID and timing
information. The logged request text is masked using the SAME PII masking
guardrail from Task 10 before it's written to disk, so a fixed-format PII
field (PAN/Aadhaar/bank account) never reaches the log file in the clear.
"""

import json
import os
import uuid
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.guardrails import mask_pii

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "requests.jsonl")


def new_trace_id() -> str:
    return str(uuid.uuid4())


def log_request(endpoint: str, raw_request_text: str, response_summary: dict,
                 duration_ms: float, trace_id: str = None) -> dict:
    """
    Writes one JSON-Lines log entry. raw_request_text is masked with the
    Task 10 PII guardrail BEFORE being included in the entry, guaranteeing
    fixed-format PII never touches disk unmasked.
    """
    trace_id = trace_id or new_trace_id()
    masked = mask_pii(raw_request_text)

    entry = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "request_text_masked": masked["masked"],
        "pii_was_masked": masked["pii_found"],
        "response_summary": response_summary,
        "duration_ms": round(duration_ms, 2),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def read_all_logs() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    print("Demo: logging a request that contains raw PII")
    trace_id = new_trace_id()
    entry = log_request(
        endpoint="/ask",
        raw_request_text="What is my loan status? My PAN is ABCDE1234F.",
        response_summary={"type": "status_lookup", "status": "Approved"},
        duration_ms=142.7,
        trace_id=trace_id,
    )
    print(entry)

    print("\nVerifying the raw PAN never landed in the log file unmasked...")
    logs = read_all_logs()
    last_entry = logs[-1]
    assert "ABCDE1234F" not in last_entry["request_text_masked"], "PII leaked into log!"
    print("Confirmed: PAN is NOT present in the logged (masked) request text.")
    print(f"\nTotal log entries so far: {len(logs)}")