"""The gate's decision logic, with no API key and no network.

The judge here is a **stub**: it returns a fixed verdict per sentence from
a table written below.  It demonstrates what the gate does with verdicts —
not how accurate a real judge is.  For that, the measured error rates are
in the README, and ``02_gate_live.py`` runs against a real model.

Four scenarios, each isolating one property:

  A. Strip and ship      — one unsupported sentence removed, rest survives
  B. Abstain             — too little survives, so nothing ships
  C. Citation-aware split— why sentence splitting is a correctness concern
  D. Fail closed         — the judge breaks, and the sentence is suppressed

D is the safety contract, and it is the one worth reading twice: a gate
that opens when it breaks is not a gate.

Usage::

    uv run python examples/01_gate_offline.py
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from typing import Any

from bearout import GateOutcome, split_sentences, validate_grounding

SOURCES = """\
[1] Reporting a defect (illustrative excerpt)
A homeowner reports a defect by submitting a written form to the warranty
provider. The submission must identify the defect and the affected part of
the home.

[2] After you submit (illustrative excerpt)
The warranty provider acknowledges the submission and may arrange an
inspection. Homeowners should retain a copy of anything they submit.
"""

# The stub judge's answer key. Every sentence the examples below feed to the
# gate appears here with the verdict a correct judge would return. Anything
# absent is treated as unsupported, which keeps even the stub fail-closed.
SUPPORTED_SENTENCES = {
    "You must report the defect in writing to the warranty provider.",
    "Keep a copy of your submission.",
    "You report a defect by submitting a written form.",
}


class ScriptedJudge:
    """A ChatLLM stub that looks up a fixed verdict. Deterministic, free.

    Implementing ``chat`` is the entire contract — the gate depends on the
    ``ChatLLM`` protocol, never on a vendor SDK, which is what lets a
    trained detector, a local model, or this three-line stub all sit in
    the same slot.
    """

    def __init__(self, *, fail_with: str | None = None) -> None:
        self.fail_with = fail_with

    async def chat(self, *, user_message: str, **_: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise RuntimeError(self.fail_with)
        sentence = user_message.removeprefix("SENTENCE:\n").strip()
        verdict = "SUPPORTED" if sentence in SUPPORTED_SENTENCES else "UNSUPPORTED"
        return {"content": verdict}


def show(title: str, answer: str, outcome: GateOutcome) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    print(f"Draft:\n  {answer}\n")
    print("Per-sentence verdicts:")
    for sentence, supported in outcome.verdicts:
        print(f"  [{'keep  ' if supported else 'STRIP '}] {sentence}")
    print(f"\ngrounded = {outcome.grounded}")
    print(f"Shipped:\n  {outcome.answer or '(nothing — the caller must abstain)'}")


async def scenario_a() -> None:
    answer = (
        "You must report the defect in writing to the warranty provider. "
        "The provider will then pay you $100,000. "
        "Keep a copy of your submission."
    )
    outcome = await validate_grounding(answer=answer, sources_block=SOURCES, llm=ScriptedJudge())
    show("A — strip the fabrication, ship the rest", answer, outcome)
    print(
        "\nThe dollar figure appears in no source. It is fluent, plausible, and\n"
        "surrounded by correct material — which is what makes it the dangerous\n"
        "kind of wrong. Two of three sentences survive, above the keep ratio."
    )


async def scenario_b() -> None:
    answer = (
        "You report a defect by submitting a written form. "
        "The provider must respond within 14 days or the claim is automatically approved. "
        "You are entitled to alternative accommodation during any repair. "
        "Unresolved claims proceed directly to binding arbitration."
    )
    outcome = await validate_grounding(answer=answer, sources_block=SOURCES, llm=ScriptedJudge())
    show("B — too little survives, so nothing ships", answer, outcome)
    print(
        "\nOne sentence of four is supported, below the keep ratio, so the whole\n"
        "answer is withheld. Shipping the survivor alone would read as complete\n"
        "while having lost every condition that qualified it — worse than silence."
    )


async def scenario_c() -> None:
    text = (
        "Report the defect under Reg. 892 s. 4.4(2). "
        "The provider publishes guidance on this "
        "(see its 'Problem with your appliances? Who you gonna call?' page). "
        "Keep a copy."
    )
    naive = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    aware = split_sentences(text)

    print(f"\n{'=' * 68}\nC — sentence splitting is a correctness concern\n{'=' * 68}")
    print(f"Text:\n  {text}\n")
    print(f"Naive split on [.!?] — {len(naive)} fragments:")
    for fragment in naive:
        print(f"  · {fragment}")
    print(f"\nCitation-aware split — {len(aware)} sentences:")
    for sentence in aware:
        print(f"  · {sentence}")
    print(
        "\nThe naive split shatters the citation into fragments like '892 s.' — which\n"
        "any honest judge then rejects, because a fragment asserts nothing the source\n"
        "supports. It also breaks mid-parenthetical on the question marks inside a\n"
        "quoted document title. So a well-sourced, citation-heavy answer over-abstains,\n"
        "and the false-strip rate you measure is really a bug in your tokenizer. One\n"
        "observed case lost 18 of 25 'sentences' this way.\n\n"
        "This is why the splitter carries an abbreviation list and tracks paren depth:\n"
        "not tidiness, but a measurable term in the published error rate."
    )


async def scenario_d() -> None:
    answer = (
        "You must report the defect in writing to the warranty provider. "
        "Keep a copy of your submission."
    )
    outcome = await validate_grounding(
        answer=answer,
        sources_block=SOURCES,
        llm=ScriptedJudge(fail_with="connection reset by peer"),
    )
    show("D — the judge breaks, and the gate closes", answer, outcome)
    print(
        "\nBoth sentences are genuinely supported, but the judge raised. Unverifiable\n"
        "is treated as unsupported, so nothing ships. A judge error, an API timeout,\n"
        "and an unparseable reply are all the same answer: no.\n"
        "This is the direction the failure has to run. A gate that opens when it\n"
        "breaks provides no guarantee at all — it only provides one on the days\n"
        "nothing goes wrong, which are not the days that matter."
    )


async def main() -> int:
    # Route the gate's own logging to stdout so the suppression records appear
    # inline. Every strip is logged; nothing fails silently.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="  [gate] %(message)s")

    print("Judge: scripted stub (deterministic, no API key, no network).")
    print("It shows what the gate does with verdicts — not how accurate a judge is.")
    await scenario_a()
    await scenario_b()
    await scenario_c()
    await scenario_d()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
