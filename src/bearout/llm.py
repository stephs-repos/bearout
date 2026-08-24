"""The minimal LLM interface the gate depends on.

Any model backend works: implement ``chat`` (an async method returning a
dict with at least a ``content`` string) and pass the object in.  The
protocol is deliberately tiny — telemetry, retries, model selection, and
cost accounting are the adapter's concern, not the gate's.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatLLM(Protocol):
    """A chat-shaped LLM call.

    Implementations must return a dict with at least:
        content: str — the raw text response

    ``cache_prefix`` is an optional large, stable block (e.g. the
    retrieved sources) that the adapter may place under a prompt-cache
    breakpoint so repeated per-claim calls against the same sources are
    cheap.  Adapters that don't support caching may simply prepend it to
    the user message.

    ``temperature`` is advisory: the gate asks for determinism, but
    newer Claude models reject sampling parameters entirely, so
    adapters are free to ignore it when the target model would refuse it.
    """

    async def chat(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        cache_prefix: str | None = None,
    ) -> dict[str, Any]: ...


__all__ = ["ChatLLM"]
