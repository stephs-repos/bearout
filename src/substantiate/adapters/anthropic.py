"""Reference ChatLLM adapter for the Anthropic API.

Requires the ``anthropic`` extra: ``pip install substantiate[anthropic]``.

Design notes:

- ``temperature`` from the gate is accepted and ignored — current Claude
  models reject sampling parameters outright (the gate's determinism
  intent is carried by its strict one-word-answer prompts instead).
- ``cache_prefix`` (the sources block) is placed under a prompt-cache
  breakpoint, so the fan-out of per-claim verifier calls re-reads the
  same sources at cache-read rates instead of re-billing them per claim.
- Thinking is left at the model default (adaptive on current models).
"""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL = "claude-opus-5"


class AnthropicChat:
    """ChatLLM implementation backed by the Anthropic Messages API.

    Credentials resolve the same way the SDK does (``ANTHROPIC_API_KEY``,
    ``ant auth login`` profile, etc.) when ``api_key`` is omitted.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 8192,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The Anthropic adapter needs the 'anthropic' package: "
                "pip install substantiate[anthropic]"
            ) from exc
        self._client = AsyncAnthropic() if api_key is None else AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def chat(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,  # noqa: ARG002 — see module docstring
        cache_prefix: str | None = None,
    ) -> dict[str, Any]:
        if cache_prefix:
            content: Any = [
                {
                    "type": "text",
                    "text": cache_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": user_message},
            ]
        else:
            content = user_message
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return {
            "content": text,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        }


__all__ = ["AnthropicChat", "DEFAULT_MODEL"]
