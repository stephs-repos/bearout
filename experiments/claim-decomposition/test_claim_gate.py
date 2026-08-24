"""Gate pipeline logic with a stub LLM.

No network: the stub routes on the system prompt (decompose / verify /
recompose) and returns canned outputs. The anchor scenario: a correct
multi-source enumeration must survive intact, and a repair must never
ship novel unverified content.
"""

from __future__ import annotations

import pytest

from claim_gate import (
    CLAIM_DECOMPOSER_SYSTEM_PROMPT,
    CLAIM_VERIFIER_SYSTEM_PROMPT,
    RECOMPOSER_SYSTEM_PROMPT,
    sentence_supported_via_claims,
    validate_with_claims,
)

ANSWER = (
    "The library lends books for three weeks. "
    "Renewals are allowed twice. "
    "Late fees are ten cents per day."
)
SENTENCES = [
    "The library lends books for three weeks.",
    "Renewals are allowed twice.",
    "Late fees are ten cents per day.",
]
SOURCES = "Loans run three weeks and may be renewed twice; the late fee is ten cents per day."

DECOMPOSITION = """{
  "1": ["The library lends books", "The loan period is three weeks"],
  "2": ["Renewals are allowed", "Up to two renewals are allowed"],
  "3": ["Late fees apply", "The late fee is ten cents per day"]
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
        "The library lends books for three weeks. "
        "Renewals are allowed. "
        "Late fees are ten cents per day."
    )
    repaired_decomposition = """{
      "1": ["The library lends books", "The loan period is three weeks",
            "Renewals are allowed", "Late fees apply",
            "The late fee is ten cents per day"]
    }"""
    llm = StubLLM(
        decompose_responses=[DECOMPOSITION, repaired_decomposition],
        verify_map={"Up to two renewals are allowed": "UNSUPPORTED"},
        recompose_response=repaired_text,
    )
    outcome = await validate_with_claims(ANSWER, SENTENCES, SOURCES, llm=llm)
    assert outcome is not None
    assert outcome.grounded is True
    assert outcome.repaired is True
    assert outcome.answer == repaired_text
    assert outcome.removed_claims == ["Up to two renewals are allowed"]
    # The flagged sentence shows unsupported in the derived verdicts...
    assert outcome.sentence_verdicts[1][1] is False
    # ...and the repair's claims were NOT re-rolled (all matched the
    # previously-supported set), so no extra verify calls happened.
    assert len(llm.verified_claims) == 6


@pytest.mark.asyncio
async def test_repair_with_novel_unsupported_claim_abstains():
    """A repair that smuggles new content must not ship."""
    repaired_decomposition = """{
      "1": ["The loan period is three weeks", "The deposit limit is $100,000"]
    }"""
    llm = StubLLM(
        decompose_responses=[DECOMPOSITION, repaired_decomposition],
        verify_map={
            "Up to two renewals are allowed": "UNSUPPORTED",
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
        "The library lends books": "UNSUPPORTED",
        "The loan period is three weeks": "UNSUPPORTED",
        "Renewals are allowed": "UNSUPPORTED",
        "Up to two renewals are allowed": "UNSUPPORTED",
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
        verify_map={"Up to two renewals are allowed": "UNSUPPORTED"},
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
