from __future__ import annotations

import pytest

from pocketeval.flow import Edge, FunctionNode, State, make_flow, run_flow


@pytest.mark.asyncio
async def test_flow_branches_and_shares_state() -> None:
    async def decide(state: State) -> str:
        return "answer" if state["prompt"] == "hello" else "skip"

    async def answer(state: State) -> None:
        state["response"] = "world"

    async def skip(state: State) -> None:
        state["response"] = "n/a"

    start = FunctionNode(decide)
    flow = make_flow(
        start,
        [Edge(start, FunctionNode(answer), "answer"), Edge(start, FunctionNode(skip), "skip")],
    )
    assert (await run_flow(flow, {"prompt": "hello"}))["response"] == "world"
    assert (await run_flow(flow, {"prompt": "other"}))["response"] == "n/a"
