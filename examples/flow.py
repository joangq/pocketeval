"""PocketFlow example with injectable model."""

from __future__ import annotations

import asyncio

from pocketeval import Message, Model, make_model
from pocketeval.flow import FunctionNode, State, make_flow, run_flow


def build_flow(model: Model):
    async def generate(state: State) -> None:
        response = await model.complete([Message("user", str(state["prompt"]))])
        state["answer"] = response.text

    return make_flow(FunctionNode(generate))


async def main() -> None:
    model = make_model("openai/gpt-4o-mini")
    response = await run_flow(build_flow(model), {"prompt": "Explain PocketFlow."})
    print(response["answer"])


if __name__ == "__main__":
    asyncio.run(main())
