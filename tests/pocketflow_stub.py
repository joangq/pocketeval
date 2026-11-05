"""Offline-only PocketFlow shim. Production always uses the real dependency."""

from __future__ import annotations

from collections.abc import MutableMapping

State = MutableMapping[str, object]


class AsyncNode:
    def __init__(self, max_retries: int = 1, wait: int = 0) -> None:
        self.successors: dict[str, AsyncNode] = {}

    def next(self, node: AsyncNode, action: str = "default") -> AsyncNode:
        self.successors[action] = node
        return node

    async def prep_async(self, shared: State) -> State:
        return shared

    async def exec_async(self, prepared: State) -> str | None:
        return None

    async def post_async(self, shared: State, prepared: State, result: str | None) -> str | None:
        return result

    async def _run_async(self, shared: State) -> str | None:
        prepared = await self.prep_async(shared)
        result = await self.exec_async(prepared)
        return await self.post_async(shared, prepared, result)


class AsyncFlow:
    def __init__(self, start: AsyncNode) -> None:
        self.start_node = start

    async def run_async(self, shared: State) -> None:
        node: AsyncNode | None = self.start_node
        while node is not None:
            action = await node._run_async(shared)
            node = node.successors.get(action or "default")
