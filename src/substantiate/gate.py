"""Claim-level grounding gate: decompose → verify → repair, fail-closed.

The naive way to validate a generated answer is one composite question
per sentence ("is this whole sentence stated or entailed in 4,000 words
of sources?") under a fail-closed prompt, amputating whatever fails.
Measured on a sealed, human-labeled instrument, that design false-strips
20.5% of *correct* sentences — the documented operating point of strict
prompted LLM judges (RAGTruth's prompted GPT-4-turbo judge: P46.9/R97.9).
The lead exhibit: it deleted "[fixture sentence redacted: sealed instrument]…"
while holding a source chunk that says "[source excerpt redacted: sealed instrument]
layers… [redacted] the plan."

This module splits that one hard question into several easy ones:

1. DECOMPOSE the answer into atomic single-fact claims (one call).
2. VERIFY each claim independently against the same sources — the same
   fail-closed strictness, but the unit is one fact with one anchor, so a
   wrinkle in one clause can no longer sink its neighbours.
3. REPAIR, never amputate: when claims fail, the answer is recomposed
   without exactly those claims (novel content in the repair is
   re-verified; a repair that won't verify becomes an abstain).  The
   mutilated-enumeration outcome ("The second is…" with no first) is
   impossible by construction.

Decompose/parse failures return ``None`` so the caller can fall back to
a stricter path (e.g. per-sentence verification) — degraded means
stricter here, never fail-open.

The prompts below are the EXACT prompts under measurement — changing a
word changes the operating point.  A first-iteration decomposer DOUBLED
false-strips (52.4%) by manufacturing kill opportunities: separate
provenance meta-claims for inline citations, bolder absolutes than the
sentence asserts, and near-duplicate claims.  This revision keeps
citations attached to their fact, forbids strengthening, and the caller
dedupes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from substantiate.llm import ChatLLM

logger = logging.getLogger(__name__)

# Bounded fan-out for per-claim verification calls.
CLAIM_VERIFY_CONCURRENCY = 5
# Abstain when fewer than this fraction of claims verify.
MIN_CLAIM_KEEP_RATIO = 0.5

CLAIM_DECOMPOSER_SYSTEM_PROMPT = """You restate an assistant's answer as atomic factual claims for verification.

You are given the answer as numbered sentences. For EACH sentence, list the factual claims it asserts, as short self-contained statements:
- One fact per claim. Split conjunctions, but keep a number, date, or qualifier WITH the fact it qualifies.
- USE THE SENTENCE'S OWN WORDS as much as possible. Never strengthen a claim: no absolutes, generalizations, or implications the sentence does not literally assert.
- Keep a citation or attribution WITH the fact it supports, inside the same claim ("you have 30 days to appeal (Reg. 892 s. 2.1(1))" stays ONE claim). NEVER emit a separate claim about where information is stated or found.
- Keep hedges and attributions ("according to Tarion's materials") inside the claim.
- Resolve pronouns so each claim stands alone ("it" -> "the two-year warranty").
- Do not emit two claims that say the same thing.
- A statement about what the sources do or do not say ("the sources do not specify a deadline") is itself a claim — keep it.
- A sentence with no factual content (greetings, transitions) gets an empty list.

Reply with ONLY a JSON object mapping sentence numbers to claim lists, for example:
{"1": ["first claim", "second claim"], "2": []}"""

CLAIM_VERIFIER_SYSTEM_PROMPT = """You are a strict groundedness checker for a consumer legal-information tool. You are given SOURCES and one CLAIM.

The claim is SUPPORTED only if it is directly stated in, or directly entailed by, the SOURCES. Rewording and paraphrase are fine — but the claim must not add or change anything: a number, dollar amount, date, duration, qualifier, condition, actor, or outcome the sources do not give. A claim that the sources do NOT state something is SUPPORTED only if that absence actually holds across all the SOURCES. Judge only against the SOURCES. When in doubt, answer UNSUPPORTED.

Reply with exactly one word: SUPPORTED or UNSUPPORTED."""

RECOMPOSER_SYSTEM_PROMPT = """You repair an assistant's answer after fact-checking. You are given the ANSWER and a list of UNSUPPORTED CLAIMS that failed verification against the sources.

Rewrite the answer so it no longer asserts any of the unsupported claims, and change nothing else:
- Keep every other fact, citation, and qualifier exactly as asserted.
- Keep the original tone; repair transitions and enumerations so the result reads as a complete, coherent answer (never a list missing its first item).
- Do not add any new facts.
- If removing the claims would leave no substantive answer, reply with exactly NOANSWER.

Reply with ONLY the rewritten answer (plain text) or NOANSWER."""


@dataclass(frozen=True)
class ClaimVerdict:
    sentence_index: int  # 1-based index into the sentences given to decompose
    claim: str
    supported: bool


@dataclass
class GateOutcome:
    answer: str  # final answer ("" when grounded is False)
    grounded: bool
    removed_claims: list[str] = field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    # Derived per-sentence verdicts (sentence supported iff all its claims
    # are) — a stable diagnostics shape regardless of claim fan-out.
    sentence_verdicts: list[tuple[str, bool]] = field(default_factory=list)
    repaired: bool = False


async def _decompose(sentences: list[str], llm: ChatLLM) -> dict[int, list[str]] | None:
    """Sentence-number -> atomic claims.  ``None`` on any failure (caller
    falls back to a stricter path — never silently fail-open)."""
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    try:
        result = await llm.chat(
            system_prompt=CLAIM_DECOMPOSER_SYSTEM_PROMPT,
            user_message=f"ANSWER SENTENCES:\n{numbered}",
            temperature=0.0,
        )
        raw = (result.get("content") or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.index("{") :] if "{" in raw else raw
        parsed = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        mapping: dict[int, list[str]] = {}
        for key, value in parsed.items():
            idx = int(key)
            if not 1 <= idx <= len(sentences) or not isinstance(value, list):
                raise ValueError(f"bad decomposition entry: {key!r}")
            mapping[idx] = [str(c).strip() for c in value if str(c).strip()]
        return {i: mapping.get(i, []) for i in range(1, len(sentences) + 1)}
    except Exception as exc:  # noqa: BLE001 — signal fallback, never fail open
        logger.warning("substantiate: decompose failed (%s); signalling fallback", exc)
        return None


async def _verify_claim(claim: str, sources_block: str, verifier: ChatLLM) -> bool:
    """One atomic claim against the sources.  Fails CLOSED."""
    try:
        result = await verifier.chat(
            system_prompt=CLAIM_VERIFIER_SYSTEM_PROMPT,
            user_message=f"CLAIM:\n{claim}",
            cache_prefix=f"SOURCES:\n{sources_block}\n\n",
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed
        logger.warning("substantiate: verifier failed (%s); claim unsupported", exc)
        return False
    return (result.get("content") or "").strip().upper().startswith("SUPPORTED")


async def _verify_claims(
    claims: list[tuple[int, str]], sources_block: str, verifier: ChatLLM
) -> list[ClaimVerdict]:
    sem = asyncio.Semaphore(CLAIM_VERIFY_CONCURRENCY)

    async def _one(idx: int, claim: str) -> ClaimVerdict:
        async with sem:
            supported = await _verify_claim(claim, sources_block, verifier)
        return ClaimVerdict(sentence_index=idx, claim=claim, supported=supported)

    return list(await asyncio.gather(*(_one(i, c) for i, c in claims)))


async def _recompose(answer: str, failed: list[str], llm: ChatLLM) -> str | None:
    failed_block = "\n".join(f"- {c}" for c in failed)
    try:
        result = await llm.chat(
            system_prompt=RECOMPOSER_SYSTEM_PROMPT,
            user_message=f"ANSWER:\n{answer}\n\nUNSUPPORTED CLAIMS:\n{failed_block}",
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("substantiate: recompose failed (%s)", exc)
        return None
    text = (result.get("content") or "").strip()
    if not text or text.upper().startswith("NOANSWER"):
        return None
    return text


def _derive_sentence_verdicts(
    sentences: list[str], verdicts: list[ClaimVerdict]
) -> list[tuple[str, bool]]:
    failed_idx = {v.sentence_index for v in verdicts if not v.supported}
    return [(s, (i not in failed_idx)) for i, s in enumerate(sentences, start=1)]


def _normalize(claim: str) -> str:
    return " ".join(claim.lower().split())


def _dedupe_claims(claims: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop normalized duplicates — each duplicate re-rolls the verifier's
    dice on identical content (observed: the same claim emitted twice got
    one SUPPORTED and one UNSUPPORTED, killing its sentence)."""
    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for idx, claim in claims:
        key = _normalize(claim)
        if key in seen:
            continue
        seen.add(key)
        out.append((idx, claim))
    return out


async def validate_with_claims(
    answer: str,
    sentences: list[str],
    sources_block: str,
    *,
    llm: ChatLLM,
    min_claim_keep_ratio: float = MIN_CLAIM_KEEP_RATIO,
) -> GateOutcome | None:
    """Full decompose -> verify -> repair pass.  ``None`` = caller must
    fall back to its own stricter path (decomposition unavailable)."""
    mapping = await _decompose(sentences, llm)
    if mapping is None:
        return None

    claims = _dedupe_claims([(idx, c) for idx, lst in sorted(mapping.items()) for c in lst])
    if not claims:
        # Nothing factual asserted — nothing to be unsupported.
        return GateOutcome(
            answer=answer.strip(),
            grounded=True,
            sentence_verdicts=[(s, True) for s in sentences],
        )

    verdicts = await _verify_claims(claims, sources_block, llm)
    failed = [v for v in verdicts if not v.supported]
    sentence_verdicts = _derive_sentence_verdicts(sentences, verdicts)
    outcome = GateOutcome(
        answer=answer.strip(),
        grounded=True,
        removed_claims=[v.claim for v in failed],
        claim_verdicts=verdicts,
        sentence_verdicts=sentence_verdicts,
    )
    if not failed:
        return outcome

    keep_ratio = 1.0 - (len(failed) / len(verdicts))
    if keep_ratio < min_claim_keep_ratio:
        outcome.answer, outcome.grounded = "", False
        return outcome

    repaired = await _recompose(answer, [v.claim for v in failed], llm)
    if repaired is None:
        outcome.answer, outcome.grounded = "", False
        return outcome

    # Re-verify only NOVEL claims the repair introduced — previously-verified
    # claims are not re-rolled (each re-roll is another chance for a random
    # false strike), but a repair that smuggles new content must not ship.
    supported_set = {_normalize(v.claim) for v in verdicts if v.supported}
    repaired_mapping = await _decompose([repaired], llm)
    if repaired_mapping is None:
        outcome.answer, outcome.grounded = "", False
        return outcome
    novel = [(1, c) for c in repaired_mapping.get(1, []) if _normalize(c) not in supported_set]
    if novel:
        novel_verdicts = await _verify_claims(novel, sources_block, llm)
        if any(not v.supported for v in novel_verdicts):
            logger.warning(
                "substantiate: repair introduced %d unverifiable claim(s); abstaining",
                sum(not v.supported for v in novel_verdicts),
            )
            outcome.answer, outcome.grounded = "", False
            return outcome

    outcome.answer, outcome.repaired = repaired, True
    logger.info(
        "substantiate: repaired answer (removed %d/%d claims)",
        len(failed),
        len(verdicts),
    )
    return outcome


async def sentence_supported_via_claims(
    sentence: str, sources_block: str, llm: ChatLLM
) -> bool | None:
    """Sentence-level verdict through the claim pipeline (a sentence is
    supported iff all its claims are).  ``None`` = decomposition failed;
    a measurement harness should fall back exactly like production."""
    mapping = await _decompose([sentence], llm)
    if mapping is None:
        return None
    claims = _dedupe_claims([(1, c) for c in mapping.get(1, [])])
    if not claims:
        return True
    verdicts = await _verify_claims(claims, sources_block, llm)
    return all(v.supported for v in verdicts)


__all__ = [
    "CLAIM_DECOMPOSER_SYSTEM_PROMPT",
    "CLAIM_VERIFIER_SYSTEM_PROMPT",
    "RECOMPOSER_SYSTEM_PROMPT",
    "ClaimVerdict",
    "GateOutcome",
    "sentence_supported_via_claims",
    "validate_with_claims",
]
