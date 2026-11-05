"""Optional LiteLLM adapter. Importing pocketeval does not import LiteLLM."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

from pocketeval.model import Completion, Message, TokenUsage


class _ResponseMessage(Protocol):
    content: str | None


class _Choice(Protocol):
    message: _ResponseMessage


class _Response(Protocol):
    choices: Sequence[_Choice]
    usage: object


class LiteLLMModel:
    def __init__(self, name: str, **options: object) -> None:
        if not name.strip():
            raise ValueError("Model name must not be empty")
        self.name = name
        self.options = options

    async def complete(self, messages: Sequence[Message]) -> Completion:
        try:
            provider = import_module("litellm")
        except ModuleNotFoundError as exc:
            message = "Install pocketeval-pocketflow[litellm] to use make_model"
            raise RuntimeError(message) from exc

        complete = getattr(provider, "acompletion")
        call = cast(Callable[..., Awaitable[_Response]], complete)
        response = await call(
            model=self.name,
            messages=[{"role": message.role, "content": message.content} for message in messages],
            **self.options,
        )
        if not response.choices:
            raise ValueError("Model returned no choices")
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Model returned no text")
        return Completion(content, _completion_cost(provider, response), _usage(response))


def _usage(response: _Response) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    return TokenUsage(input_tokens, output_tokens)


def _completion_cost(provider: object, response: _Response) -> float | None:
    try:
        cost_fn = cast(Callable[..., object], getattr(provider, "completion_cost"))
        value = cost_fn(completion_response=response)
    except Exception:  # noqa: BLE001 - provider accounting is optional and can fail independently
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    cost = float(value)
    return cost if math.isfinite(cost) and cost >= 0 else None
