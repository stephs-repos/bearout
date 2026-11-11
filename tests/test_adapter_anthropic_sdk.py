"""Wire-level tests of the Anthropic adapter against the real, installed SDK.

``test_adapter_anthropic.py`` pins the adapter's own logic against a stub.
This file checks what that stub cannot: that the request the installed
``anthropic`` package actually puts on the wire is the measured
configuration, and that its response objects parse into the adapter's dict.
The HTTP layer is a mock transport, so nothing leaves the machine.

Skipped when the ``anthropic`` extra is not installed.  CI runs it at both
ends of the supported range (see the ``sdk-compat`` job).
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")

SDK_MAJOR = int(anthropic.__version__.split(".")[0])
# SDK 1.0 moved its HTTP layer from httpx to the httpx2 fork.
httpx = importlib.import_module("httpx2" if SDK_MAJOR >= 1 else "httpx")

pytestmark = pytest.mark.filterwarnings(
    # SDK >= 1.11 warns on every call that the measured judge is nearing
    # end-of-life.  That is the model's lifecycle, not adapter behaviour.
    "ignore:The model 'claude-sonnet-4-5:DeprecationWarning"
)

SOURCES = "The library lends books for three weeks."


def message_json(model: str, text: str = "SUPPORTED") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 42,
            "output_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 7,
        },
    }


class Recorder:
    """A mock transport handler that records every request it is sent."""

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.status = 200
        self.text = "SUPPORTED"
        self.raise_connect_error = False

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        if self.raise_connect_error:
            raise httpx.ConnectError("refused", request=request)
        if self.status != 200:
            # retry-after-ms keeps the retry tests fast on every SDK version.
            return httpx.Response(self.status, headers={"retry-after-ms": "1"}, json={})
        model = json.loads(request.content)["model"]
        return httpx.Response(200, json=message_json(model, self.text))

    def body(self, i: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Route every AsyncAnthropic the adapter builds through a Recorder."""
    recorder = Recorder()
    real_client = anthropic.AsyncAnthropic

    class RecordingAsyncAnthropic(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            transport = httpx.MockTransport(recorder)
            kwargs.setdefault("http_client", anthropic.DefaultAsyncHttpxClient(transport=transport))
            super().__init__(**kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", RecordingAsyncAnthropic)
    return recorder


def build(**kwargs: Any):
    from bearout.adapters.anthropic import AnthropicChat

    return AnthropicChat(api_key="test-key", **kwargs)


def sdk_default_api_version() -> str:
    return anthropic.AsyncAnthropic(api_key="k").default_headers["anthropic-version"]


class TestMeasuredRequestOnTheWire:
    @pytest.mark.asyncio
    async def test_measured_judge_sends_temperature_zero_under_2023_06_01(self, wire) -> None:
        from bearout.adapters.anthropic import MEASURED_MODEL

        await build().chat(system_prompt="sys", user_message="SENTENCE:\nx", temperature=0.0)
        body = wire.body()
        assert body["model"] == MEASURED_MODEL
        assert body["temperature"] == 0.0
        # Exactly one version header, whatever casing the SDK uses for its own.
        assert wire.requests[0].headers.get_list("anthropic-version") == ["2023-06-01"]

    @pytest.mark.asyncio
    async def test_newer_model_sends_no_temperature_and_keeps_sdk_version(self, wire) -> None:
        await build(model="claude-opus-5").chat(
            system_prompt="sys", user_message="SENTENCE:\nx", temperature=0.0
        )
        assert "temperature" not in wire.body()
        assert wire.requests[0].headers.get_list("anthropic-version") == [sdk_default_api_version()]

    @pytest.mark.asyncio
    async def test_request_shape_matches_the_measured_prompt_layout(self, wire) -> None:
        await build().chat(
            system_prompt="sys",
            user_message="SENTENCE:\nx",
            temperature=0.0,
            cache_prefix="SOURCES:\nbig block\n\n",
        )
        body = wire.body()
        assert body["system"] == "sys"
        assert body["max_tokens"] == 8192
        [message] = body["messages"]
        assert message["role"] == "user"
        assert message["content"] == [
            {
                "type": "text",
                "text": "SOURCES:\nbig block\n\n",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "SENTENCE:\nx"},
        ]

    @pytest.mark.asyncio
    async def test_api_key_is_sent(self, wire) -> None:
        # SDK 2.x moved the key from X-Api-Key to Authorization: Bearer; the
        # API accepts either, so only check that the key went out.
        await build().chat(system_prompt="sys", user_message="x")
        headers = wire.requests[0].headers
        assert headers.get("x-api-key") == "test-key" or (
            headers.get("authorization") == "Bearer test-key"
        )


class TestResponseParsing:
    @pytest.mark.asyncio
    async def test_sdk_response_objects_map_to_the_adapter_dict(self, wire) -> None:
        from bearout.adapters.anthropic import MEASURED_MODEL

        result = await build().chat(system_prompt="sys", user_message="x", temperature=0.0)
        assert result == {
            "content": "SUPPORTED",
            "model": MEASURED_MODEL,
            "stop_reason": "end_turn",
            "prompt_tokens": 42,
            "completion_tokens": 1,
            "cache_read_input_tokens": 7,
        }


class TestRetryAndTimeoutDefaults:
    """The adapter relies on SDK defaults; pin that they did not move."""

    def test_client_defaults(self, wire) -> None:
        client = build()._client
        assert client.max_retries == 2
        assert client.timeout.read == 600
        assert client.timeout.connect == 5.0

    @pytest.mark.asyncio
    async def test_overloaded_status_is_retried_twice_then_raised(self, wire) -> None:
        wire.status = 529
        with pytest.raises(anthropic.APIStatusError):
            await build().chat(system_prompt="sys", user_message="x")
        assert len(wire.requests) == 3

    @pytest.mark.asyncio
    async def test_bad_request_is_not_retried(self, wire) -> None:
        wire.status = 400
        with pytest.raises(anthropic.BadRequestError):
            await build().chat(system_prompt="sys", user_message="x")
        assert len(wire.requests) == 1

    @pytest.mark.asyncio
    async def test_connection_errors_are_retried_twice_then_raised(self, wire) -> None:
        wire.raise_connect_error = True
        with pytest.raises(anthropic.APIConnectionError):
            await build().chat(system_prompt="sys", user_message="x")
        assert len(wire.requests) == 3


class TestGateEndToEnd:
    """validate_grounding fails closed on any adapter error, so an SDK
    incompatibility would surface only as every answer abstaining."""

    @pytest.mark.asyncio
    async def test_supported_answer_passes_through_the_real_sdk(self, wire) -> None:
        from bearout import validate_grounding

        outcome = await validate_grounding(SOURCES, SOURCES, llm=build())
        assert outcome.grounded is True
        assert outcome.answer == SOURCES
        assert wire.body()["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_unsupported_verdict_still_abstains(self, wire) -> None:
        from bearout import validate_grounding

        wire.text = "UNSUPPORTED"
        outcome = await validate_grounding(SOURCES, SOURCES, llm=build())
        assert outcome.grounded is False
        assert outcome.answer == ""
