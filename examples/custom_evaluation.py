"""Offline example: replace FakeTarget and CustomCheck with your own implementations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pocketeval.evals import Case, Observation, Verdict, evaluate


@dataclass(frozen=True)
class CustomCheck:
    name: str = "min-words"

    def evaluate(self, case: Case, observation: Observation) -> Verdict:
        words = len(observation.text.split())
        return Verdict(float(words), words >= 2, f"words={words}")


async def main() -> None:
    report = await evaluate(
        [Case("greeting", "Say hello", [CustomCheck()])],
        lambda prompt: f"Hello there: {prompt}",
    )
    print(report)
    report.assert_passed()


if __name__ == "__main__":
    asyncio.run(main())
