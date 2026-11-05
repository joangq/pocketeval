"""Real model evaluation example (requires LiteLLM and provider credentials)."""

from __future__ import annotations

import asyncio

from pocketeval import make_model
from pocketeval.evals import Case, ModelJudge, ModelTarget, contains, evaluate, rubric


async def main() -> None:
    model = make_model("openai/gpt-4o-mini")
    judge = ModelJudge(make_model("openai/gpt-4o-mini", response_format={"type": "json_object"}))
    report = await evaluate(
        [
            Case(
                "france",
                "Name the capital of France.",
                [contains("Paris"), rubric(judge, "Is the answer factually correct?")],
            )
        ],
        ModelTarget(model),
    )
    print(report)
    report.assert_passed()


if __name__ == "__main__":
    asyncio.run(main())
