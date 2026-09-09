"""
Task 14 (Part 4): MCP Server
------------------------------------
Wraps check_loan_application_status (from agent/tools.py) as an MCP tool and
launches a private MCP server locally using fastmcp, on HTTP transport.

fastmcp's HTTP transport mounts at /mcp by default, so the client must point
at http://127.0.0.1:8000/mcp -- NOT the bare host:port root.

Run this in its own terminal and leave it running, then run mcp/client.py
in a SEPARATE terminal/process to connect to it.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from agent.tools import check_loan_application_status as _check_loan_application_status

mcp = FastMCP("Cred Loan Status Server")


@mcp.tool()
def check_loan_application_status(record_id: str) -> dict:
    """
    Look up a Cred loan application by its record_id and return its status,
    loan amount in INR, and a computed escalation_score in [0, 1] that
    combines fraud-review flagging with how long the application has been
    sitting in the pipeline.

    Args:
        record_id: The loan application's unique ID, e.g. "LN0001".

    Returns:
        A dict with keys: record_id, found, status, loan_amount_inr,
        escalation_score, recommend_escalation, flagged_for_fraud_review,
        days_since_created.
    """
    return _check_loan_application_status(record_id)


if __name__ == "__main__":
    print("Starting MCP server on http://127.0.0.1:8000/mcp ...")
    print("Leave this running, then run 'python mcp/client.py' in a separate terminal.")
    mcp.run(transport="http", host="127.0.0.1", port=8000)