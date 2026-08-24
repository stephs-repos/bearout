"""Tests for the Anthropic adapter's model and sampling contract.

The published error rates are a property of one judge running at one
temperature.  The adapter's job is to make that the shipped default, so an
out-of-the-box run and the README describe the same configuration.  These
tests pin that, plus the reason it cannot simply forward ``temperature``
unconditionally: newer Claude models reject sampling parameters with a 400.

The ``anthropic`` package is an optional extra and is not installed here, so
a stub is injected before the adapter imports it.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="SUPPORTED")],
            model=kwargs["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=42, output_tokens=1, cache_read_input_tokens=7),
        )


class FakeAsyncAnthropic:
    def __init__(self, **_: Any) -> None:
        self.messages = FakeMessages()


@pytest.fixture(autouse=True)
def _stub_anthropic_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("anthropic")
    module.AsyncAnthropic = FakeAsyncAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


def build(**kwargs: Any):
    from bearout.adapters.anthropic import AnthropicChat

    return AnthropicChat(api_key="test-key", **kwargs)


async def call(chat, temperature: float | None = 0.0) -> dict[str, Any]:
    return await chat.chat(
        system_prompt="sys", user_message="SENTENCE:\nx", temperature=temperature
    )


class TestMeasuredDefault:
    def test_default_model_is_the_measured_judge(self) -> None:
        from bearout.adapters.anthropic import DEFAULT_MODEL, MEASURED_MODEL

        # If these drift apart, a default run stops being the configuration the
        # published rates describe, and the README quietly becomes wrong.
        assert DEFAULT_MODEL == MEASURED_MODEL

    def test_measured_model_is_pinned_to_a_dated_snapshot(self) -> None:
        from bearout.adapters.anthropic import MEASURED_MODEL

        # An alias can be repointed underneath the measurement; a snapshot cannot.
        assert MEASURED_MODEL.startswith("claude-sonnet-4-5-")
        assert MEASURED_MODEL != "claude-sonnet-4-5"


class TestTemperatureForwarding:
    @pytest.mark.asyncio
    async def test_measured_judge_receives_temperature(self) -> None:
        chat = build()
        await call(chat, temperature=0.0)
        assert chat._client.messages.calls[0]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_newer_models_do_not_receive_temperature(self) -> None:
        # Claude 4.6-generation models and later return a 400 for sampling
        # parameters, so forwarding one would break the swappable-judge promise.
        chat = build(model="claude-opus-5")
        await call(chat, temperature=0.0)
        assert "temperature" not in chat._client.messages.calls[0]

    @pytest.mark.asyncio
    async def test_forwarding_can_be_forced_for_another_older_model(self) -> None:
        chat = build(model="claude-sonnet-4-5", forward_temperature=True)
        await call(chat, temperature=0.0)
        assert chat._client.messages.calls[0]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_forwarding_can_be_disabled_for_the_measured_judge(self) -> None:
        chat = build(forward_temperature=False)
        await call(chat, temperature=0.0)
        assert "temperature" not in chat._client.messages.calls[0]

    @pytest.mark.asyncio
    async def test_omitted_temperature_is_never_invented(self) -> None:
        chat = build()
        await call(chat, temperature=None)
        assert "temperature" not in chat._client.messages.calls[0]


class TestRequestShape:
    @pytest.mark.asyncio
    async def test_sources_ride_under_a_cache_breakpoint(self) -> None:
        chat = build()
        await chat.chat(
            system_prompt="sys",
            user_message="SENTENCE:\nx",
            cache_prefix="SOURCES:\nbig block\n\n",
        )
        content = chat._client.messages.calls[0]["messages"][0]["content"]
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert content[0]["text"].startswith("SOURCES:")
        assert content[1]["text"] == "SENTENCE:\nx"

    @pytest.mark.asyncio
    async def test_usage_is_surfaced_for_cost_accounting(self) -> None:
        chat = build()
        result = await call(chat)
        assert result["prompt_tokens"] == 42
        assert result["completion_tokens"] == 1
        assert result["cache_read_input_tokens"] == 7
        assert result["content"] == "SUPPORTED"
