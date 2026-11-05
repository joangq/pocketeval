"""Built-in checks; custom checks only need to satisfy the public Check protocol."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from difflib import SequenceMatcher

from pocketeval.evals.core import Case, Grade, Judge, Observation, Verdict

CheckFunction = Callable[[Case, Observation], Verdict | Awaitable[Verdict]]


@dataclass(frozen=True, slots=True)
class FunctionCheck:
    name: str
    function: CheckFunction
    detail: str = ""

    def evaluate(self, case: Case, observation: Observation) -> Verdict | Awaitable[Verdict]:
        return self.function(case, observation)


def _score_threshold(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Threshold must be a finite number between 0 and 1")


def _minimum(name: str, measure: Callable[[Observation], float], threshold: float) -> FunctionCheck:
    _score_threshold(threshold)

    def check(_case: Case, observation: Observation) -> Verdict:
        value = measure(observation)
        if not math.isfinite(value):
            raise ValueError("Metric returned a non-finite value")
        return Verdict(value, value >= threshold, f"score={value:.3f}, min={threshold:.3f}")

    return FunctionCheck(name, check)


def equals(expected: str) -> FunctionCheck:
    return _minimum("equals", lambda obs: float(obs.text == expected), 1.0)


def contains(expected: str, *, ignore_case: bool = False) -> FunctionCheck:
    if not expected:
        raise ValueError("Expected substring cannot be empty")

    def measure(obs: Observation) -> float:
        text, needle = obs.text, expected
        if ignore_case:
            text, needle = text.casefold(), needle.casefold()
        return float(needle in text)

    return _minimum("contains", measure, 1.0)


def similarity(expected: str, *, threshold: float = 0.8) -> FunctionCheck:
    """Case-insensitive character-level SequenceMatcher ratio."""
    return _minimum(
        "similarity",
        lambda obs: SequenceMatcher(None, expected.casefold(), obs.text.casefold()).ratio(),
        threshold,
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _f1(overlap: int, reference: int, candidate: int) -> float:
    total = reference + candidate
    if total == 0:
        return 0
    return 2 * overlap / total


def rouge_n(expected: str, *, n: int = 1, threshold: float = 0.75) -> FunctionCheck:
    """Word n-gram overlap F1, including repeated n-grams."""
    if n < 1:
        raise ValueError("n must be positive")
    order = n

    def grams(text: str) -> Counter[tuple[str, ...]]:
        words = _tokens(text)
        ngrams: list[tuple[str, ...]] = []
        for start in range(len(words) - order + 1):
            ngrams.append(tuple(words[start : start + order]))
        return Counter(ngrams)

    reference = grams(expected)

    def measure(obs: Observation) -> float:
        candidate = grams(obs.text)
        overlap = sum((reference & candidate).values())
        return _f1(overlap, sum(reference.values()), sum(candidate.values()))

    return _minimum(f"rouge-{n}", measure, threshold)


def rouge_l(expected: str, *, threshold: float = 0.75) -> FunctionCheck:
    """Word-level longest-common-subsequence F1."""
    reference = _tokens(expected)

    def measure(obs: Observation) -> float:
        candidate = _tokens(obs.text)
        previous = [0 for _ in range(len(candidate) + 1)]
        for word in reference:
            current = [0]
            for index, other in enumerate(candidate, start=1):
                current.append(
                    previous[index - 1] + 1
                    if word == other
                    else max(previous[index], current[-1])
                )
            previous = current
        return _f1(previous[-1], len(reference), len(candidate))

    return _minimum("rouge-l", measure, threshold)


def levenshtein(expected: str, *, max_distance: int = 0) -> FunctionCheck:
    if max_distance < 0:
        raise ValueError("max_distance cannot be negative")

    def distance(candidate: str) -> int:
        left, right = expected, candidate
        if len(left) < len(right):
            left, right = right, left
        previous = list(range(len(right) + 1))
        for row, char in enumerate(left, start=1):
            current = [row]
            for column, other in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + (char != other),
                    )
                )
            previous = current
        return previous[-1]

    def check(_case: Case, obs: Observation) -> Verdict:
        value = float(distance(obs.text))
        return Verdict(value, value <= max_distance, f"distance={value:.0f}, max={max_distance}")

    return FunctionCheck("levenshtein", check)


def latency(*, max_ms: float) -> FunctionCheck:
    if not math.isfinite(max_ms) or max_ms < 0:
        raise ValueError("max_ms must be finite and non-negative")

    def check(_case: Case, obs: Observation) -> Verdict:
        elapsed = obs.latency_ms
        return Verdict(elapsed, elapsed <= max_ms, f"latency={elapsed:.2f}ms, max={max_ms}ms")

    return FunctionCheck("latency", check)


def cost(*, max_usd: float) -> FunctionCheck:
    if not math.isfinite(max_usd) or max_usd < 0:
        raise ValueError("max_usd must be finite and non-negative")

    def check(_case: Case, obs: Observation) -> Verdict:
        value = obs.cost_usd
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Cost unavailable or invalid; supply Output(cost_usd=...)")
        return Verdict(value, value <= max_usd, f"cost=${value:.6f}, max=${max_usd:.6f}")

    return FunctionCheck("cost", check)


def rubric(judge: Judge, criterion: str, *, threshold: float = 0.7) -> FunctionCheck:
    if not criterion.strip():
        raise ValueError("Criterion must not be empty")
    _score_threshold(threshold)

    async def check(case: Case, obs: Observation) -> Verdict:
        grade: Grade = await judge.grade(case.prompt, obs.text, criterion)
        score = grade.score
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Judge score must be a finite number between 0 and 1")
        return Verdict(float(score), score >= threshold, grade.reason)

    return FunctionCheck("rubric", check, detail=criterion)


def relevance(judge: Judge, *, threshold: float = 0.7) -> FunctionCheck:
    criterion = "Rate whether the answer directly and sufficiently addresses the user's question."
    check = rubric(judge, criterion, threshold=threshold)
    return FunctionCheck("relevance", check.evaluate, detail=criterion)
