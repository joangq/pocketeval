# pocketeval

A small, typed Python library extracted from an application that used LiteLLM, PocketFlow, and an evaluation runner. The extracted code is domain-independent: it contains **no API endpoints, business prompts, chat storage, or application configuration**.

## Install

Python 3.11+:

```bash
pip install -e .                # Provider-free evaluation core
pip install -e '.[flow]'       # PocketFlow utilities
pip install -e '.[litellm]'    # LiteLLM-backed models
pip install -e '.[all]'        # Both optional adapters
```

The evaluation core has **zero runtime dependencies**. PocketFlow and LiteLLM are separate, optional extras. Only `pocketeval.flow` requires the PocketFlow extra.

## Model interface

Any implementation satisfying `Model` can be used, without subclassing:

```python
from pocketeval import Completion, Message, Model, make_model

model: Model = make_model("openai/gpt-4o-mini", temperature=0)
answer = await model.complete([Message("user", "Say hello")])
print(answer.text, answer.cost_usd, answer.usage)
```

`make_model` constructs the optional LiteLLM adapter. To use another backend, implement `async complete(messages: Sequence[Message]) -> Completion`.

## PocketFlow helpers

```python
from pocketeval.flow import Edge, FunctionNode, State, make_flow, run_flow

async def validate(state: State) -> str:
    return "valid" if state["prompt"] else "invalid"

async def answer(state: State) -> None:
    state["answer"] = "Hello"

async def invalid(state: State) -> None:
    state["answer"] = "Empty prompt"

start = FunctionNode(validate)
flow = make_flow(start, [
    Edge(start, FunctionNode(answer), "valid"),
    Edge(start, FunctionNode(invalid), "invalid"),
])
state = await run_flow(flow, {"prompt": "hi"})
print(state["answer"])
```

`FunctionNode` is a small adapter over PocketFlow's `AsyncNode`; existing PocketFlow node subclasses work with `Edge` and `make_flow` directly. A step returns an edge name or `None` for the default transition. Each call can construct its own graph to avoid mutating shared node connections.

## Pluggable evaluation

The evaluation runner knows only three **public, structural `Protocol` interfaces**:

| Contract | Method | Purpose |
|---|---|---|
| `Target` | `__call__(prompt)` → `str \| Output` or an awaitable | Model, application, or flow under test |
| `Check` | `name` and `evaluate(case, observation)` → `Verdict` or awaitable | Independent, user-defined assertion |
| `Judge` | `async grade(prompt, answer, criterion)` → `Grade` | Optional provider-independent semantic grader |

A target is executed once per case. Its wall-clock latency is measured; supplied cost is preserved (missing cost is **not** silently treated as zero). Checks are independent, and individual target/check failures appear as failed results rather than aborting the suite. `Report.assert_passed()` integrates with pytest, CI, or plain scripts.

```python
from pocketeval.evals import Case, ModelTarget, contains, cost, evaluate, latency

report = await evaluate(
    [Case("hello", "Say hello", [contains("hello", ignore_case=True), latency(max_ms=5000)])],
    ModelTarget(model),
)
report.assert_passed()
```

Custom check with no inheritance or registration:

```python
from dataclasses import dataclass
from pocketeval.evals import Case, Observation, Verdict

@dataclass(frozen=True)
class WordCount:
    name: str = "word-count"

    def evaluate(self, case: Case, observation: Observation) -> Verdict:
        count = len(observation.text.split())
        return Verdict(float(count), count >= 5, f"words={count}")
```

You can supply any `Judge` implementation to `rubric(judge, criterion)` and `relevance(judge)`. `ModelJudge` adapts any `Model` into a judge. For JSON-enforcing LiteLLM providers, set `response_format={"type": "json_object"}` when creating the judge model. Run evaluations against flows using a custom target function that creates a state, calls `run_flow(...)`, and returns `Output`.

Built-in checks: `equals`, `contains`, `similarity`, `rouge_n`, `rouge_l`, `levenshtein`, `latency`, `cost`, `rubric`, `relevance`. Names and return values come from public dataclasses; there are no plugin registries, global state, or hidden dependencies.

## Quality checks

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
ruff format --check .
ty check src tests examples
```

The source application's test-only QA corpus, mocked API search, and model-specific prompts are deliberately excluded. Its generic metrics, timing, judge, and PocketFlow patterns are retained in provider-neutral form.
