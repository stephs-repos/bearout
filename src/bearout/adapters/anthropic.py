"""Reference ChatLLM adapter for the Anthropic API.

Requires the ``anthropic`` extra: ``pip install bearout[anthropic]``.

Design notes:

- The default model is :data:`MEASURED_MODEL`, the judge the published
  error rates were measured with.  An out-of-the-box run is therefore the
  configuration those numbers describe.  Pass ``model=`` for anything
  else, and re-measure before relying on the published rates: a different
  judge is a different operating point.
- ``temperature`` is forwarded only when the target model accepts it.
  Newer Claude generations reject sampling parameters outright, while the
  measured judge predates that change and was run at temperature 0 — so
  the parameter has to actually reach the API for the default
  configuration to reproduce the measurement.
- ``cache_prefix`` (the sources block) is placed under a prompt-cache
  breakpoint, so the fan-out of per-sentence verifier calls re-reads the
  same sources at cache-read rates instead of re-billing them per call.
- Thinking is left at the model default.
"""

from __future__ import annotations

from typing import Any

# The judge the published error rates were measured with.  Pinned to the dated
# snapshot rather than the ``claude-sonnet-4-5`` alias, because an alias can be
# repointed and a judge that moves underneath you silently invalidates every
# number in the README.  If this snapshot is retired, that is a signal to
# re-measure, not to quietly bump the string.
MEASURED_MODEL = "claude-sonnet-4-5-20250929"

# Ship the configuration the published rates actually describe.
DEFAULT_MODEL = MEASURED_MODEL


class AnthropicChat:
    """ChatLLM implementation backed by the Anthropic Messages API.

    Credentials resolve the same way the SDK does (``ANTHROPIC_API_KEY``,
    ``ant auth login`` profile, etc.) when ``api_key`` is omitted.

    ``forward_temperature`` defaults to True only for :data:`MEASURED_MODEL`,
    whose measurement ran at temperature 0.  Set it True explicitly for another
    older model that accepts sampling parameters; leave it False for newer
    generations, which reject them outright.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 8192,
        forward_temperature: bool | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The Anthropic adapter needs the 'anthropic' package: "
                "pip install bearout[anthropic]"
            ) from exc
        self._client = AsyncAnthropic() if api_key is None else AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._forward_temperature = (
            model == MEASURED_MODEL if forward_temperature is None else forward_temperature
        )

    async def chat(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
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

        sampling: dict[str, Any] = {}
        if self._forward_temperature and temperature is not None:
            sampling["temperature"] = temperature

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
            **sampling,
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


__all__ = ["DEFAULT_MODEL", "MEASURED_MODEL", "AnthropicChat"]
