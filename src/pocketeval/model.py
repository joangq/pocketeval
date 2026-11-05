"""Small, provider-independent chat model interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    cost_usd: float | None = None
    usage: TokenUsage | None = None


class Model(Protocol):
    """Implement this protocol to use any LLM provider."""

    async def complete(self, messages: Sequence[Message]) -> Completion: ...


def make_model(name: str, **options: object) -> Model:
    """Create the optional LiteLLM adapter; other models implement ``Model``."""
    from pocketeval.providers.litellm import LiteLLMModel

    return LiteLLMModel(name, **options)
