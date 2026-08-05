"""Sentence-level grounding gate: verify → strip → abstain, fail-closed.

Every sentence of a generated answer is checked against the retrieved
sources by a strict judge.  Unsupported sentences are removed.  If too
little of the answer survives, the whole answer is withheld rather than
shipped gutted.

**This is the code the published error rates describe.**  See the
"Measured error rates" section of the README: on a sealed 474-item
expert-labeled fixture, this gate false-strips ~20% of supported
sentences and false-passes ~2% of unsupported ones.  The prompt below is
the exact string under measurement — changing a word changes the
operating point and invalidates the numbers.

Fail-closed everywhere: a judge error, an API failure, or an
unparseable verdict all count as UNSUPPORTED.  The only way a sentence
ships is an affirmative SUPPORTED.

An alternative design — decomposing each sentence into atomic claims and
verifying those — was preregistered, measured, and **rejected**: it
roughly doubled the false-strip rate.  The code, the preregistration,
and the run artifacts are in ``experiments/claim-decomposition/``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from substantiate.llm import ChatLLM

logger = logging.getLogger(__name__)

# Abstain when fewer than this fraction of sentences survive, rather than
# shipping a gutted answer.
MIN_KEEP_RATIO = 0.5

# The exact prompt under measurement.  Do not reword: the published
# false-strip / false-pass rates are properties of this string.
GROUNDING_VERIFIER_SYSTEM_PROMPT = """You are a strict groundedness checker for a consumer legal-information tool. You are given SOURCES and one SENTENCE an assistant wrote.

A sentence is SUPPORTED only if EVERY factual claim, number, date, qualifier, and condition in it is directly stated in, or directly entailed by, the SOURCES. Rewording and paraphrase are fine — but the assistant must not ADD anything the sources do not contain.

Answer UNSUPPORTED if the sentence adds any detail not in the sources — for example a timing qualifier ("at any time", "immediately", "within 30 days"), a number, a guarantee, a cause, or a recommendation the sources do not make. When in doubt, answer UNSUPPORTED.

Reply with exactly one word: SUPPORTED or UNSUPPORTED."""


# ---------------------------------------------------------------------------
# Sentence splitting
#
# Naive splitting on [.!?] is not merely imprecise here — it is a direct
# cause of false strips.  Two failure modes, both observed in production:
#
#   1. Statutory citations shatter.  "Reg. 892 s. 4.4(2)" becomes fragments
#      like "892 s.", which the judge then correctly rejects as unsupported
#      — so a well-retrieved, citation-heavy answer over-abstains.  One
#      measured case stripped 18 of 25 "sentences" this way.
#   2. Terminal punctuation inside a parenthetical citation.  A document
#      title carried into parentheses — "(Tarion's 'Problem with your
#      appliances? Who you gonna call?' page)" — splits mid-citation.
#
# Hence: an abbreviation blocklist, a single-letter-abbreviation guard
# (s. c. v. O.), and paren-depth tracking that suppresses any boundary
# while inside an open '('.
# ---------------------------------------------------------------------------

_NO_SPLIT_ABBREVS = (
    "Reg",
    "reg",
    "No",
    "ss",
    "Ss",
    "Art",
    "art",
    "Cl",
    "cl",
    "Para",
    "para",
    "Sched",
    "subs",
    "Subs",
    "vs",
    "Inc",
    "Ltd",
    "approx",
    "Approx",
    "pp",
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])"  # keep the terminator with the sentence (don't consume it)
    r"(?<!\b[A-Za-z]\.)"  # not after a single-letter abbreviation (s. c. v. O.)
    + "".join(rf"(?<!{re.escape(_a)}\.)" for _a in _NO_SPLIT_ABBREVS)  # Reg. No. art. ...
    + r"\s+(?=[\"“'(A-Z])"  # only at a real boundary (next char starts a sentence)
)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, never breaking inside a citation.

    Boundaries falling inside an open parenthetical are suppressed by
    tracking paren depth; abbreviations and single-letter forms are
    protected by the split regex.  See the module comment for why this
    matters to the measured error rates.
    """
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    scanned = 0
    depth = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        boundary = m.start()
        depth += text.count("(", scanned, boundary) - text.count(")", scanned, boundary)
        scanned = boundary
        if depth <= 0:
            depth = 0  # clamp: a stray ')' must not make later boundaries "inside"
            segment = text[start:boundary].strip()
            if segment:
                out.append(segment)
            start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass
class GateOutcome:
    """What the gate decided.

    ``answer`` is the surviving text, or ``""`` when ``grounded`` is False
    (the caller must abstain — never ship the original in that case).
    ``removed`` and ``verdicts`` exist so a suppression is always
    explainable after the fact.
    """

    answer: str
    grounded: bool
    removed: list[str] = field(default_factory=list)
    verdicts: list[tuple[str, bool]] = field(default_factory=list)


async def verify_sentence(sentence: str, sources_block: str, verifier: ChatLLM) -> bool:
    """True if the sentence is grounded in the sources.  Fails CLOSED.

    The sources ride in ``cache_prefix``: they are identical across every
    sentence of an answer, so an adapter that supports prompt caching pays
    for them once instead of once per sentence.
    """
    try:
        result = await verifier.chat(
            system_prompt=GROUNDING_VERIFIER_SYSTEM_PROMPT,
            user_message=f"SENTENCE:\n{sentence}",
            cache_prefix=f"SOURCES:\n{sources_block}\n\n",
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed: unverifiable == unsupported
        logger.warning("substantiate: verifier failed (%s); treating sentence as unsupported", exc)
        return False
    return (result.get("content") or "").strip().upper().startswith("SUPPORTED")


async def validate_grounding(
    answer: str,
    sources_block: str,
    *,
    llm: ChatLLM,
    min_keep_ratio: float = MIN_KEEP_RATIO,
    concurrency: int = 1,
) -> GateOutcome:
    """Verify each sentence of ``answer`` against ``sources_block``.

    Unsupported sentences are dropped.  If the surviving fraction falls
    below ``min_keep_ratio`` the answer is withheld entirely
    (``grounded=False``, ``answer=""``) — a half-answer that has lost its
    qualifiers is more dangerous than no answer.

    ``concurrency`` defaults to 1 (sequential), which is how the published
    rates were measured.  Sentences are judged independently and share no
    state, so raising it changes latency and cost, not verdicts.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return GateOutcome(answer.strip(), True)

    if concurrency > 1:
        sem = asyncio.Semaphore(concurrency)

        async def _one(s: str) -> bool:
            async with sem:
                return await verify_sentence(s, sources_block, llm)

        supported_flags = list(await asyncio.gather(*(_one(s) for s in sentences)))
    else:
        supported_flags = [await verify_sentence(s, sources_block, llm) for s in sentences]

    verdicts = list(zip(sentences, supported_flags, strict=True))
    kept = [s for s, ok in verdicts if ok]
    removed = [s for s, ok in verdicts if not ok]

    grounded = bool(kept) and (len(kept) / len(sentences)) >= min_keep_ratio
    if removed:
        logger.info(
            "substantiate: removed %d/%d sentence(s) as unsupported",
            len(removed),
            len(sentences),
        )
    return GateOutcome(
        answer=" ".join(kept).strip() if grounded else "",
        grounded=grounded,
        removed=removed,
        verdicts=verdicts,
    )


__all__ = [
    "GROUNDING_VERIFIER_SYSTEM_PROMPT",
    "MIN_KEEP_RATIO",
    "GateOutcome",
    "split_sentences",
    "validate_grounding",
    "verify_sentence",
]
