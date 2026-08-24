"""The grounding gate against a real judge, with cost and latency measured.

Two scenarios, both run against the live Anthropic API:

  A. A mostly-sound answer with one fabricated detail.  The gate removes
     the fabrication and ships the rest.
  B. An answer where most of the content is unsupported.  The gate
     refuses to ship it at all rather than hand back a remnant.

The sources here are **illustrative, not real law** — they exist to make
the gate's decision legible in a few lines.  For verification against a
genuine authoritative source, see ``03_fidelity_elaws.py``.

Every verifier call is metered, so the run ends with a practical figure:
what a fail-closed gate costs per answer, given that it makes one model
call per sentence.

**This does not measure accuracy.**  It shows the gate's behavior on a
handful of sentences.  The published false-pass and false-strip rates
come from a sealed, expert-labelled 474-item set that is deliberately not
shipped in this repository — training against the exam destroys the only
measure you have.  See the README for the methodology and the results.

Usage::

    export ANTHROPIC_API_KEY=...
    uv run python examples/02_gate_live.py
    uv run python examples/02_gate_live.py --model claude-sonnet-4-5
    uv run python examples/02_gate_live.py --save   # write the transcript

Cost: a few cents per run (7 verifier calls on short prompts).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bearout import GateOutcome, validate_grounding
from bearout.adapters.anthropic import DEFAULT_MODEL, AnthropicChat

# List prices in USD per million tokens, as published 2026-06-24. Only models
# whose pricing is pinned here get a cost estimate; anything else reports
# tokens and latency only, because a stale price is worse than no price.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

SOURCES = """\
[1] Reporting a defect (illustrative excerpt)
A homeowner reports a defect by submitting a written form to the warranty
provider. The submission must identify the defect and the affected part of
the home.

[2] After you submit (illustrative excerpt)
The warranty provider acknowledges the submission and may arrange an
inspection. Homeowners should retain a copy of anything they submit.
"""

# Scenario A: two sentences the sources support, one fabricated dollar figure.
# The fabrication is fluent, plausible, and arrives wrapped in correct material
# — which is exactly what makes it dangerous and what the gate is for.
ANSWER_ONE_BAD_SENTENCE = (
    "You must report the defect in writing to the warranty provider. "
    "The provider will then pay you $100,000. "
    "Keep a copy of your submission."
)

# Scenario B: only the first sentence is supported. Shipping the survivor alone
# would read as a complete answer while having lost everything that qualified it.
ANSWER_MOSTLY_UNSUPPORTED = (
    "You report a defect by submitting a written form. "
    "The provider must respond within 14 days or the claim is automatically approved. "
    "You are entitled to alternative accommodation for the duration of any repair. "
    "Unresolved claims proceed directly to binding arbitration."
)


@dataclass
class Meter:
    """A ChatLLM that forwards to a real adapter and counts what it costs.

    The gate deliberately keeps telemetry out of its own code path — its
    module docstring puts cost accounting on the adapter's side of the
    line.  Wrapping rather than instrumenting is how you get the numbers
    back without touching the code the published rates describe.
    """

    inner: Any
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    seconds: float = 0.0
    verdicts: list[dict[str, Any]] = field(default_factory=list)

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        started = time.perf_counter()
        result = await self.inner.chat(**kwargs)
        self.seconds += time.perf_counter() - started

        self.calls += 1
        self.prompt_tokens += result.get("prompt_tokens", 0) or 0
        self.completion_tokens += result.get("completion_tokens", 0) or 0
        self.cache_read_tokens += result.get("cache_read_input_tokens", 0) or 0
        self.verdicts.append(
            {
                "sentence": kwargs.get("user_message", "").removeprefix("SENTENCE:\n"),
                "raw_reply": (result.get("content") or "").strip(),
            }
        )
        return result

    def cost_usd(self, model: str) -> float | None:
        price = PRICING_PER_MTOK.get(model)
        if price is None:
            return None
        per_in, per_out = price
        # Cache reads bill at roughly a tenth of the input rate.
        billed_in = self.prompt_tokens + self.cache_read_tokens * 0.1
        return (billed_in * per_in + self.completion_tokens * per_out) / 1_000_000


def show(title: str, answer: str, outcome: GateOutcome) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    print(f"Draft ({len(outcome.verdicts)} sentences):\n  {answer}\n")

    print("Per-sentence verdicts:")
    for sentence, supported in outcome.verdicts:
        mark = "keep  " if supported else "STRIP "
        print(f"  [{mark}] {sentence}")

    print(f"\ngrounded = {outcome.grounded}")
    if outcome.grounded:
        print(f"Shipped:\n  {outcome.answer}")
    else:
        print("Shipped:\n  (nothing — too little survived; the caller must abstain)")
    if outcome.removed:
        print("\nSuppressed, and recorded:")
        for sentence in outcome.removed:
            print(f"  - {sentence}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--save", action="store_true", help="write the run transcript to disk")
    args = parser.parse_args()

    meter = Meter(inner=AnthropicChat(model=args.model))

    print(f"Judge model: {args.model}")
    print(
        "Note: the error rates published in the README were measured with a "
        "different judge.\nA different model is a different operating point — "
        "re-measure before relying on it."
    )

    outcome_a = await validate_grounding(
        answer=ANSWER_ONE_BAD_SENTENCE, sources_block=SOURCES, llm=meter
    )
    show("A — one fabricated sentence among sound ones", ANSWER_ONE_BAD_SENTENCE, outcome_a)

    outcome_b = await validate_grounding(
        answer=ANSWER_MOSTLY_UNSUPPORTED, sources_block=SOURCES, llm=meter
    )
    show("B — most of the answer is unsupported", ANSWER_MOSTLY_UNSUPPORTED, outcome_b)

    cost = meter.cost_usd(args.model)
    cost_text = f"${cost:.4f}" if cost is not None else "n/a (no pinned price for this model)"
    print(f"\n{'=' * 68}\nCost and latency\n{'=' * 68}")
    print(
        f"  verifier calls   {meter.calls}\n"
        f"  input tokens     {meter.prompt_tokens} "
        f"(+{meter.cache_read_tokens} cache reads)\n"
        f"  output tokens    {meter.completion_tokens}\n"
        f"  wall time        {meter.seconds:.1f}s (sequential — the gate defaults to "
        f"concurrency=1)\n"
        f"  estimated cost   {cost_text}"
    )

    if args.save:
        path = Path(__file__).resolve().parent / "transcripts" / "02_gate_live.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "sources": SOURCES,
                    "scenarios": {
                        "a_one_bad_sentence": {
                            "answer": ANSWER_ONE_BAD_SENTENCE,
                            "grounded": outcome_a.grounded,
                            "verdicts": [
                                {"sentence": s, "supported": ok} for s, ok in outcome_a.verdicts
                            ],
                        },
                        "b_mostly_unsupported": {
                            "answer": ANSWER_MOSTLY_UNSUPPORTED,
                            "grounded": outcome_b.grounded,
                            "verdicts": [
                                {"sentence": s, "supported": ok} for s, ok in outcome_b.verdicts
                            ],
                        },
                    },
                    "usage": {
                        "calls": meter.calls,
                        "input_tokens": meter.prompt_tokens,
                        "cache_read_tokens": meter.cache_read_tokens,
                        "output_tokens": meter.completion_tokens,
                        "seconds": round(meter.seconds, 2),
                        "estimated_cost_usd": cost,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nTranscript written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
