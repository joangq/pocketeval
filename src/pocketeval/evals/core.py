"""Dependency-free evaluation contracts and runner.

    Protocols are the extension points: Target, Check, and Judge.
    No provider, test runner, persistence, or specific metric is required.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class Output:
    text: str
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    text: str
    latency_ms: float
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class Grade:
    score: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Verdict:
    value: float
    passed: bool
    reason: str = ""


TargetOutput: TypeAlias = str | Output


class Target(Protocol):
    """Anything that accepts a prompt and returns a string or Output."""

    def __call__(self, prompt: str) -> TargetOutput | Awaitable[TargetOutput]: ...


class Judge(Protocol):
    """Independent grader implementation, used only by judge-based checks."""

    async def grade(self, prompt: str, answer: str, criterion: str) -> Grade: ...


class Check(Protocol):
    """Independent criterion. Define a name and an evaluate method."""

    @property
    def name(self) -> str: ...

    def evaluate(self, case: Case, observation: Observation) -> Verdict | Awaitable[Verdict]: ...


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    prompt: str
    checks: Sequence[Check]


@dataclass(frozen=True, slots=True)
class Result:
    case: str
    check: str
    passed: bool
    value: float | None
    reason: str = ""
    answer: str = ""
    elapsed_ms: float | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class Report:
    results: tuple[Result, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def assert_passed(self) -> None:
        if self.passed:
            return
        failures: list[str] = []
        for entry in self.results:
            if entry.passed:
                continue
            detail = entry.reason if entry.reason else str(entry.value)
            failures.append(f"{entry.case} / {entry.check}: {detail}")
        lines = failures if failures else ["No checks ran"]
        raise AssertionError("\n".join(["Evaluation failures:", *lines]))


async def _invoke_target(case: Case, target: Target) -> Observation:
    started = perf_counter()
    response = target(case.prompt)
    if inspect.isawaitable(response):
        response = await response
    elapsed = (perf_counter() - started) * 1000
    output = Output(response) if isinstance(response, str) else response
    if not isinstance(output, Output) or not isinstance(output.text, str):
        raise TypeError("Target must return str or Output")
    return Observation(output.text, elapsed, output.cost_usd)


async def _resolve_verdict(
    case: Case,
    check: Check,
    observation: Observation,
) -> Verdict:
    verdict = check.evaluate(case, observation)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    if not isinstance(verdict, Verdict):
        raise TypeError("Check must return Verdict")
    return verdict


async def _run_check(case: Case, check: Check, observation: Observation) -> Result:
    try:
        verdict = await _resolve_verdict(case, check, observation)
    except Exception as exc:  # noqa: BLE001 - a failed check must not hide other checks
        message = f"{type(exc).__name__}: {exc}"
        return _result(case, check, None, message, observation)
    return _result(case, check, verdict, "", observation)


async def evaluate(cases: Iterable[Case], target: Target) -> Report:
    """Execute each target once per case. Isolate provider and check failures."""
    results: list[Result] = []
    for case in cases:
        if not case.checks:
            raise ValueError(f"Case {case.name!r} has no checks")
        try:
            observation = await _invoke_target(case, target)
        except Exception as exc:  # noqa: BLE001 - one failed target must not abort a suite
            message = f"Target {type(exc).__name__}: {exc}"
            results.extend(
                _result(case, check, None, message, None) for check in case.checks
            )
            continue
        for check in case.checks:
            results.append(await _run_check(case, check, observation))
    return Report(tuple(results))


def _result(
    case: Case,
    check: Check,
    verdict: Verdict | None,
    error: str,
    observation: Observation | None,
) -> Result:
    passed = verdict.passed if verdict else False
    score = verdict.value if verdict else None
    reason = verdict.reason if verdict else error
    answer = observation.text if observation else ""
    elapsed_ms = observation.latency_ms if observation else None
    cost_usd = observation.cost_usd if observation else None
    return Result(
        case=case.name,
        check=check.name,
        passed=passed,
        value=score,
        reason=reason,
        answer=answer,
        elapsed_ms=elapsed_ms,
        cost_usd=cost_usd,
    )
