"""
Task 14 (Part 4): MCP Client
------------------------------------
A SEPARATE process/script from the LangGraph agent, which connects to the
MCP server (mcp/server.py) over HTTP and calls check_loan_application_status
for at least 2 different record IDs, printing the standardized MCP response
for each.

IMPORTANT: Start the server first ('python mcp/server.py' in one terminal),
THEN run this client in a separate terminal.
"""

import asyncio
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"

TEST_RECORD_IDS = ["LN0001", "LN0002", "LN0005"]


async def call_tool_for_record(client: Client, record_id: str):
    result = await client.call_tool(
        "check_loan_application_status", {"record_id": record_id}
    )
    print(f"\n--- MCP tool call: record_id={record_id} ---")
    print(f"Standardized MCP response: {result}")
    return result


async def main():
    print(f"Connecting to MCP server at {SERVER_URL} ...")
    async with Client(SERVER_URL) as client:
        print("Connected. Listing available tools...")
        tools = await client.list_tools()
        print(f"Tools exposed by server: {[t.name for t in tools]}")

        for record_id in TEST_RECORD_IDS:
            await call_tool_for_record(client, record_id)

    print(f"\nSuccessfully called the tool via MCP for {len(TEST_RECORD_IDS)} different record IDs.")


if __name__ == "__main__":
    asyncio.run(main())