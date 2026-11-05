from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from pocketeval.evals import (
    Case,
    Grade,
    ModelJudge,
    ModelTarget,
    Observation,
    Output,
    Verdict,
    contains,
    cost,
    equals,
    evaluate,
    latency,
    levenshtein,
    relevance,
    rouge_l,
    rouge_n,
    rubric,
    similarity,
)
from pocketeval.model import Completion


@dataclass
class FakeJudge:
    score: float = 0.9

    async def grade(self, prompt: str, answer: str, criterion: str) -> Grade:
        return Grade(self.score, f"checked: {criterion}")


@dataclass
class FakeModel:
    text: str
    cost: float | None = None

    async def complete(self, messages: object) -> Completion:
        return Completion(self.text, self.cost)


@pytest.mark.asyncio
async def test_builtin_checks() -> None:
    checks = [
        equals("hello world"),
        contains("WORLD", ignore_case=True),
        similarity("hello world", threshold=1),
        rouge_n("hello world", threshold=1),
        rouge_n("hello world", n=2, threshold=1),
        rouge_l("hello world", threshold=1),
        levenshtein("hello world"),
        latency(max_ms=1_000),
        cost(max_usd=0.001),
    ]
    report = await evaluate(
        [Case("hello", "hello", checks)], lambda _: Output("hello world", 0.0002)
    )
    assert report.passed
    assert len(report.results) == 9
    report.assert_passed()


@pytest.mark.asyncio
async def test_checks_report_real_failure_values() -> None:
    report = await evaluate(
        [Case("text", "x", [levenshtein("kitten", max_distance=2), cost(max_usd=0.1)])],
        lambda _: Output("sitting", 0.2),
    )
    assert not report.passed
    assert report.results[0].value == 3
    assert report.results[1].value == 0.2
    with pytest.raises(AssertionError, match="text / levenshtein"):
        report.assert_passed()


@pytest.mark.asyncio
async def test_missing_and_invalid_cost_fail() -> None:
    case = Case("cost", "hi", [cost(max_usd=0.01)])
    for amount in (None, -1, float("nan"), float("inf")):
        report = await evaluate([case], lambda _: Output("hi", amount))
        assert not report.passed
        assert "Cost unavailable" in report.results[0].reason


@pytest.mark.asyncio
async def test_rubric_accepts_arbitrary_judge() -> None:
    judge = FakeJudge(0.91)
    report = await evaluate(
        [Case("judge", "hello", [rubric(judge, "clear"), relevance(judge)])],
        lambda _: "hello",
    )
    assert report.passed
    assert report.results[0].value == 0.91


@pytest.mark.asyncio
async def test_invalid_judge_score_fails_closed() -> None:
    for score in (float("nan"), -1.0, 2.0, True):
        report = await evaluate(
            [Case("judge", "x", [rubric(FakeJudge(score), "correct")])], lambda _: "x"
        )
        assert not report.passed
        assert "score" in report.results[0].reason


@pytest.mark.asyncio
async def test_one_target_failure_does_not_abort_suite() -> None:
    def target(prompt: str) -> str:
        if prompt == "bad":
            raise RuntimeError("unavailable")
        return "good"

    report = await evaluate(
        [Case("bad", "bad", [equals("good")]), Case("good", "good", [equals("good")])],
        target,
    )
    assert not report.results[0].passed
    assert report.results[1].passed
    assert report.results[0].elapsed_ms is None


@pytest.mark.asyncio
async def test_async_target_and_measurement() -> None:
    async def target(_: str) -> str:
        await asyncio.sleep(0.01)
        return "done"

    report = await evaluate([Case("slow", "hello", [latency(max_ms=1)])], target)
    assert not report.passed
    assert report.results[0].value is not None
    assert report.results[0].value >= 1


@dataclass(frozen=True)
class CustomCheck:
    """No subclassing and no framework imports beyond the public contracts."""

    name: str = "word-count"

    def evaluate(self, case: Case, observation: Observation) -> Verdict:
        words = len(observation.text.split())
        return Verdict(float(words), words >= 2, f"words={words}")


@pytest.mark.asyncio
async def test_custom_protocol_implementation_and_empty_suite() -> None:
    report = await evaluate([Case("custom", "prompt", [CustomCheck()])], lambda _: "two words")
    assert report.passed
    assert not (await evaluate([], lambda _: "hi")).passed
    with pytest.raises(AssertionError, match="No checks ran"):
        (await evaluate([], lambda _: "hi")).assert_passed()
    with pytest.raises(ValueError, match="no checks"):
        await evaluate([Case("empty", "prompt", [])], lambda _: "hi")


@pytest.mark.asyncio
async def test_model_target_and_model_judge() -> None:
    target = ModelTarget(FakeModel("answer", 0.002), system_prompt="Be concise")
    report = await evaluate([Case("target", "prompt", [equals("answer")])], target)
    assert report.passed
    assert report.results[0].cost_usd == 0.002

    judge = ModelJudge(FakeModel('{"score": 0.8, "reason": "correct"}'))
    grade = await judge.grade("prompt", "answer", "criterion")
    assert grade == Grade(0.8, "correct")


@pytest.mark.asyncio
async def test_model_judge_rejects_invalid_scores_and_payloads() -> None:
    for text in ('{"score": true}', '{"score": 2}', '{"score": "0.8"}', "[]"):
        with pytest.raises(ValueError):
            await ModelJudge(FakeModel(text)).grade("x", "y", "z")


def test_checks_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        contains("")
    with pytest.raises(ValueError):
        rouge_n("text", n=0)
    with pytest.raises(ValueError):
        similarity("text", threshold=float("nan"))
    with pytest.raises(ValueError):
        rubric(FakeJudge(), " ")


@pytest.mark.asyncio
async def test_failing_check_does_not_hide_next_check() -> None:
    class BrokenCheck:
        name = "broken"

        def evaluate(self, case: Case, observation: Observation) -> Verdict:
            raise RuntimeError("broken criterion")

    report = await evaluate(
        [Case("two checks", "x", [BrokenCheck(), equals("x")])], lambda _: "x"
    )
    assert not report.results[0].passed
    assert report.results[1].passed
