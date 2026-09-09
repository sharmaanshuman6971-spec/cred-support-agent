"""
Task 16 (Part 4): Timeouts and Retries
------------------------------------------
Demonstrates three resilience mechanisms using LangGraph's native support:

  (a) RETRY POLICY: an exponential-backoff retry policy (max_attempts,
      initial_interval, max_interval, jitter) wraps a node that simulates a
      transient failure -- it fails the first 2 calls via a counter, then
      succeeds on the 3rd -- and demonstrates recovery within the configured
      attempts.

  (b) PER-NODE TIMEOUT: a node timeout (in seconds) is attached to a node
      whose simulated work exceeds it, demonstrating a clean, typed error
      (NodeTimeoutError) rather than a hang.

  (c) GLOBAL TIMEOUT: the whole graph invocation is wrapped in
      asyncio.wait_for(), demonstrating the entire run being cancelled when
      its total simulated execution time exceeds the global budget.

NOTE: LangGraph only supports per-node timeouts on ASYNC node functions
(sync functions cannot be safely cancelled mid-execution in-process), so all
demo nodes here are defined as async and run via app.ainvoke().
"""

import asyncio
import sys
import os
from typing import TypedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy


class DemoState(TypedDict):
    result: str


# =====================================================================
# (a) RETRY POLICY: recovers from a simulated transient failure
# =====================================================================

_flaky_attempt_counter = {"count": 0}


def flaky_node(state: DemoState) -> DemoState:
    _flaky_attempt_counter["count"] += 1
    attempt = _flaky_attempt_counter["count"]
    print(f"    flaky_node attempt #{attempt}")
    if attempt < 3:
        raise ConnectionError(f"Simulated transient failure on attempt #{attempt}")
    state["result"] = "success after retries"
    return state


RETRY_POLICY = RetryPolicy(
    initial_interval=0.3,
    backoff_factor=2.0,
    max_interval=5.0,
    max_attempts=5,
    jitter=True,
)


def demo_retry_policy():
    print("=" * 70)
    print("(a) RETRY POLICY: recovering from a simulated transient failure")
    print(f"    Config: initial_interval=0.3s, backoff_factor=2.0, "
          f"max_interval=5.0s, max_attempts=5, jitter=True")
    print("=" * 70)

    _flaky_attempt_counter["count"] = 0  # reset for a repeatable demo

    graph = StateGraph(DemoState)
    graph.add_node("flaky_node", flaky_node, retry_policy=RETRY_POLICY)
    graph.set_entry_point("flaky_node")
    graph.add_edge("flaky_node", END)
    app = graph.compile()

    result = app.invoke({"result": ""})
    print(f"\nFinal result: {result}")
    print(f"Total attempts made: {_flaky_attempt_counter['count']} "
          f"(recovered within max_attempts=5)")


# =====================================================================
# (b) PER-NODE TIMEOUT: fires a clean error, not a hang
# =====================================================================

async def slow_async_node(state: DemoState) -> DemoState:
    await asyncio.sleep(3.0)  # deliberately exceeds the node's 1.0s timeout
    state["result"] = "should never reach here"
    return state


async def demo_node_timeout():
    print("\n" + "=" * 70)
    print("(b) PER-NODE TIMEOUT: node sleeps 3.0s, node timeout=1.0s")
    print("=" * 70)

    graph = StateGraph(DemoState)
    graph.add_node("slow_async_node", slow_async_node, timeout=1.0)
    graph.set_entry_point("slow_async_node")
    graph.add_edge("slow_async_node", END)
    app = graph.compile()

    try:
        result = await app.ainvoke({"result": ""})
        print(f"UNEXPECTED: node completed without timing out: {result}")
    except Exception as e:
        print(f"\nCLEAN TIMEOUT ERROR fired (not a hang): {type(e).__name__}: {e}")


# =====================================================================
# (c) GLOBAL TIMEOUT: whole graph run cancelled on total-time overrun
# =====================================================================

async def slow_step_1(state: DemoState) -> DemoState:
    await asyncio.sleep(1.5)
    state["result"] = "step1 done"
    return state


async def slow_step_2(state: DemoState) -> DemoState:
    await asyncio.sleep(1.5)
    state["result"] = "step2 done"
    return state


GLOBAL_TIMEOUT_SECONDS = 2.0


async def demo_global_timeout():
    print("\n" + "=" * 70)
    print(f"(c) GLOBAL TIMEOUT: 2 nodes x 1.5s each = 3.0s total, "
          f"global budget={GLOBAL_TIMEOUT_SECONDS}s")
    print("=" * 70)

    graph = StateGraph(DemoState)
    graph.add_node("slow_step_1", slow_step_1)
    graph.add_node("slow_step_2", slow_step_2)
    graph.set_entry_point("slow_step_1")
    graph.add_edge("slow_step_1", "slow_step_2")
    graph.add_edge("slow_step_2", END)
    app = graph.compile()

    try:
        result = await asyncio.wait_for(app.ainvoke({"result": ""}), timeout=GLOBAL_TIMEOUT_SECONDS)
        print(f"UNEXPECTED: run completed within budget: {result}")
    except asyncio.TimeoutError:
        print(f"\nGLOBAL TIMEOUT fired cleanly: total run time exceeded "
              f"{GLOBAL_TIMEOUT_SECONDS}s and the whole run was cancelled.")


async def main():
    demo_retry_policy()
    await demo_node_timeout()
    await demo_global_timeout()


if __name__ == "__main__":
    asyncio.run(main())