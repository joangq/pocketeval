"""Tiny helpers for constructing async PocketFlow graphs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from dataclasses import dataclass

from pocketflow import AsyncFlow, AsyncNode

State = MutableMapping[str, object]
Step = Callable[[State], Awaitable[str | None]]


class FunctionNode(AsyncNode):
    """Use an async function as a PocketFlow node.

    Return an action string to select a named edge, or None for the default edge.
    """

    def __init__(self, function: Step) -> None:
        super().__init__()
        self.function = function

    async def prep_async(self, shared: State) -> State:
        return shared

    async def exec_async(self, prepared: State) -> str | None:
        return await self.function(prepared)

    async def post_async(self, shared: State, prepared: State, result: str | None) -> str | None:
        return result


@dataclass(frozen=True, slots=True)
class Edge:
    source: AsyncNode
    target: AsyncNode
    action: str = "default"


def make_flow(start: AsyncNode, edges: Iterable[Edge] = ()) -> AsyncFlow:
    """Connect nodes and return a new PocketFlow graph."""
    for edge in edges:
        edge.source.next(edge.target, edge.action)
    return AsyncFlow(start=start)


async def run_flow(flow: AsyncFlow, state: State) -> State:
    """Run an async flow and return its shared state."""
    await flow.run_async(state)
    return state
