from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pocketeval import Message, TokenUsage, make_model
from pocketeval.providers.litellm import LiteLLMModel


@pytest.mark.asyncio
async def test_litellm_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    async def acompletion(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(acompletion=acompletion, completion_cost=lambda **_: 0.001),
    )
    model = make_model("example-model", temperature=0.2)
    completion = await model.complete([Message("user", "hello")])
    assert completion.text == "hello"
    assert completion.usage == TokenUsage(5, 2)
    assert completion.usage.total_tokens == 7
    assert completion.cost_usd == 0.001
    assert calls == [
        {
            "model": "example-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
        }
    ]


@pytest.mark.asyncio
async def test_missing_cost_is_not_recorded_as_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    async def acompletion(**_: object) -> object:
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    def fail(**_: object) -> float:
        raise RuntimeError("unknown model pricing")

    monkeypatch.setitem(
        sys.modules, "litellm", SimpleNamespace(acompletion=acompletion, completion_cost=fail)
    )
    result = await LiteLLMModel("test").complete([Message("user", "x")])
    assert result.cost_usd is None
    assert result.usage is None


def test_blank_model_name_rejected() -> None:
    with pytest.raises(ValueError, match="name"):
        make_model(" ")
