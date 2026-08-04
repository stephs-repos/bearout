"""Gate pipeline logic with a stub LLM.

No network: the stub routes on the system prompt (decompose / verify /
recompose) and returns canned outputs. The anchor scenario: a correct
multi-source enumeration must survive intact, and a repair must never
ship novel unverified content.
"""

from __future__ import annotations

import pytest

from substantiate.gate import (
    CLAIM_DECOMPOSER_SYSTEM_PROMPT,
    CLAIM_VERIFIER_SYSTEM_PROMPT,
    RECOMPOSER_SYSTEM_PROMPT,
    sentence_supported_via_claims,
    validate_with_claims,
)

ANSWER = (
    "[fixture sentence redacted: sealed instrument]. "
    "The first is a one-year warranty covering workmanship and materials. "
    "The second is a two-year warranty covering water penetration."
)
SENTENCES = [
    "[fixture sentence redacted: sealed instrument].",
    "The first is a one-year warranty covering workmanship and materials.",
    "The second is a two-year warranty covering water penetration.",
]
SOURCES = "[source excerpt redacted: sealed instrument]... (stub sources)"

DECOMPOSITION = """{
  "1": ["[redacted claim]", "[redacted claim]"],
  "2": ["There is a one-year warranty", "The one-year warranty covers workmanship and materials"],
  "3": ["There is a two-year warranty", "The two-year warranty covers water penetration"]
}"""


class StubLLM:
    """Routes on system prompt; records calls for assertions."""

    def __init__(
        self,
        decompose_responses: list[str] | str = DECOMPOSITION,
        verify_map: dict[str, str] | None = None,
        recompose_response: str = "",
    ):
        self.decompose_responses = (
            [decompose_responses]
            if isinstance(decompose_responses, str)
            else list(decompose_responses)
        )
        self.verify_map = verify_map or {}
        self.recompose_response = recompose_response
        self.verified_claims: list[str] = []
        self.recompose_calls: list[str] = []

    async def chat(self, *, system_prompt, user_message, temperature=0.0, cache_prefix=None):
        if system_prompt == CLAIM_DECOMPOSER_SYSTEM_PROMPT:
            resp = self.decompose_responses.pop(0) if self.decompose_responses else "{}"
            return {"content": resp}
        if system_prompt == CLAIM_VERIFIER_SYSTEM_PROMPT:
            claim = user_message.rsplit("CLAIM:\n", 1)[1].strip()
            self.verified_claims.append(claim)
            return {"content": self.verify_map.get(claim, "SUPPORTED")}
        if system_prompt == RECOMPOSER_SYSTEM_PROMPT:
            self.recompose_calls.append(user_message)
            return {"content": self.recompose_response}
        raise AssertionError(f"unexpected system prompt: {system_prompt[:60]}")


@pytest.mark.asyncio
async def test_all_claims_pass_returns_original_untouched():
    """A correct enumeration survives with ZERO distortion."""
    llm = StubLLM()
    outcome = await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm)
    assert outcome is not None
    assert outcome.grounded is True
    assert outcome.answer == ANSWER
    assert outcome.repaired is False
    assert outcome.removed_claims == []
    assert all(ok for _, ok in outcome.sentence_verdicts)
    assert len(llm.verified_claims) == 6


@pytest.mark.asyncio
async def test_failed_claim_triggers_repair_not_amputation():
    repaired_text = (
        "[fixture sentence redacted: sealed instrument]. "
        "The first is a one-year warranty. "
        "The second is a two-year warranty covering water penetration."
    )
    repaired_decomposition = """{
      "1": ["[redacted claim]", "[redacted claim]",
            "There is a one-year warranty", "There is a two-year warranty",
            "The two-year warranty covers water penetration"]
    }"""
    llm = StubLLM(
        decompose_responses=[DECOMPOSITION, repaired_decomposition],
        verify_map={"The one-year warranty covers workmanship and materials": "UNSUPPORTED"},
        recompose_response=repaired_text,
    )
    outcome = await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm)
    assert outcome is not None
    assert outcome.grounded is True
    assert outcome.repaired is True
    assert outcome.answer == repaired_text
    assert outcome.removed_claims == ["The one-year warranty covers workmanship and materials"]
    # The flagged sentence shows unsupported in the derived verdicts...
    assert outcome.sentence_verdicts[1][1] is False
    # ...and the repair's claims were NOT re-rolled (all matched the
    # previously-supported set), so no extra verify calls happened.
    assert len(llm.verified_claims) == 6


@pytest.mark.asyncio
async def test_repair_with_novel_unsupported_claim_abstains():
    """A repair that smuggles new content must not ship."""
    repaired_decomposition = """{
      "1": ["[redacted claim]", "The deposit limit is $100,000"]
    }"""
    llm = StubLLM(
        decompose_responses=[DECOMPOSITION, repaired_decomposition],
        verify_map={
            "The one-year warranty covers workmanship and materials": "UNSUPPORTED",
            "The deposit limit is $100,000": "UNSUPPORTED",
        },
        recompose_response="Rewritten answer asserting the deposit limit is $100,000.",
    )
    outcome = await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm)
    assert outcome is not None
    assert outcome.grounded is False
    assert outcome.answer == ""


@pytest.mark.asyncio
async def test_majority_failed_claims_abstains_without_repair():
    verify_map = {
        "[redacted claim]": "UNSUPPORTED",
        "[redacted claim]": "UNSUPPORTED",
        "There is a one-year warranty": "UNSUPPORTED",
        "The one-year warranty covers workmanship and materials": "UNSUPPORTED",
    }
    llm = StubLLM(verify_map=verify_map)
    outcome = await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm)
    assert outcome is not None
    assert outcome.grounded is False
    assert outcome.answer == ""
    assert llm.recompose_calls == []  # below keep-ratio: no repair attempt


@pytest.mark.asyncio
async def test_decompose_failure_signals_fallback():
    llm = StubLLM(decompose_responses="this is not json at all")
    assert await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm) is None


@pytest.mark.asyncio
async def test_recompose_noanswer_abstains():
    llm = StubLLM(
        verify_map={"The one-year warranty covers workmanship and materials": "UNSUPPORTED"},
        recompose_response="NOANSWER",
    )
    outcome = await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm)
    assert outcome is not None
    assert outcome.grounded is False


@pytest.mark.asyncio
async def test_no_factual_claims_keeps_answer():
    llm = StubLLM(decompose_responses='{"1": [], "2": [], "3": []}')
    outcome = await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm)
    assert outcome is not None
    assert outcome.grounded is True
    assert outcome.answer == ANSWER
    assert llm.verified_claims == []


@pytest.mark.asyncio
async def test_sentence_helper_maps_all_claims_to_verdict():
    llm = StubLLM(
        decompose_responses='{"1": ["claim a", "claim b"]}',
        verify_map={"claim b": "UNSUPPORTED"},
    )
    assert await sentence_supported_via_claims("s", SOURCES, llm) is False
    llm2 = StubLLM(decompose_responses='{"1": ["claim a"]}')
    assert await sentence_supported_via_claims("s", SOURCES, llm2) is True
    llm3 = StubLLM(decompose_responses="garbage")
    assert await sentence_supported_via_claims("s", SOURCES, llm3) is None


@pytest.mark.asyncio
async def test_verifier_receives_sources_as_cache_prefix():
    """The sources block rides the cache_prefix channel so adapters can
    put it under a prompt-cache breakpoint across the per-claim fan-out."""

    class RecordingStub(StubLLM):
        def __init__(self):
            super().__init__()
            self.cache_prefixes: list[str | None] = []

        async def chat(self, *, system_prompt, user_message, temperature=0.0, cache_prefix=None):
            if system_prompt == CLAIM_VERIFIER_SYSTEM_PROMPT:
                self.cache_prefixes.append(cache_prefix)
            return await super().chat(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                cache_prefix=cache_prefix,
            )

    llm = RecordingStub()
    outcome = await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm)
    assert outcome is not None
    assert llm.cache_prefixes and all(SOURCES in (p or "") for p in llm.cache_prefixes)
