"""The guarantees the published numbers rest on, pinned as tests.

Three things must never regress silently:

1. Fail-closed on anything that is not an affirmative SUPPORTED — an empty,
   missing, malformed, or hedged verdict is treated as unsupported.
2. The abstention threshold at its exact boundary.
3. The strings the error rates are properties of: the verifier prompt and
   the measured judge.  A failure here means "re-measure", not "update the
   pin".
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata

import pytest

import bearout
from bearout.adapters.anthropic import MEASURED_MODEL
from bearout.gate import (
    GROUNDING_VERIFIER_SYSTEM_PROMPT,
    MIN_KEEP_RATIO,
    validate_grounding,
    verify_sentence,
)

SOURCES = "Loans run three weeks and may be renewed twice."


class CannedJudge:
    """Returns the same raw reply for every sentence."""

    def __init__(self, reply):
        self.reply = reply

    async def chat(self, *, system_prompt, user_message, temperature=0.0, cache_prefix=None):
        return self.reply


class ScriptedJudge:
    """Returns verdicts from a list, in call order, with optional per-call delay."""

    def __init__(self, verdicts: list[str], delays: list[float] | None = None):
        self.verdicts = list(verdicts)
        self.delays = list(delays or [])

    async def chat(self, *, system_prompt, user_message, temperature=0.0, cache_prefix=None):
        verdict = self.verdicts.pop(0)
        if self.delays:
            await asyncio.sleep(self.delays.pop(0))
        return {"content": verdict}


# --------------------------------------------------------------------------
# 1. Fail closed on anything that is not an affirmative pass
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        {"content": ""},
        {"content": None},
        {},
        {"content": "MAYBE"},
        {"content": "I think it is supported."},
        {"content": "The sentence is supported by source [1]."},
        {"content": "UNSUPPORTED"},
        {"content": "UNSUPPORTED — adds a date"},
        {"content": "unsupported"},
        {"content": "NOT SUPPORTED"},
        {"content": " \n\t"},
    ],
    ids=[
        "empty",
        "none",
        "missing-key",
        "hedge-word",
        "prose-hedge",
        "prose-verdict-not-first",
        "explicit-unsupported",
        "unsupported-with-reason",
        "lowercase-unsupported",
        "not-supported",
        "whitespace",
    ],
)
async def test_non_affirmative_reply_is_unsupported(reply):
    assert await verify_sentence("Any sentence.", SOURCES, CannedJudge(reply)) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        {"content": "SUPPORTED"},
        {"content": "supported"},
        {"content": "  SUPPORTED\n"},
        {"content": "SUPPORTED — source [1] states it directly."},
    ],
    ids=["exact", "lowercase", "padded", "with-reason"],
)
async def test_affirmative_reply_is_supported(reply):
    assert await verify_sentence("Any sentence.", SOURCES, CannedJudge(reply)) is True


@pytest.mark.asyncio
async def test_judge_exception_is_unsupported():
    class Raises:
        async def chat(self, **kwargs):
            raise ConnectionError("network down")

    assert await verify_sentence("Any sentence.", SOURCES, Raises()) is False


@pytest.mark.asyncio
async def test_judge_exception_under_concurrency_still_fails_closed():
    class Raises:
        async def chat(self, **kwargs):
            raise RuntimeError("boom")

    out = await validate_grounding("One. Two. Three. Four.", SOURCES, llm=Raises(), concurrency=4)
    assert out.grounded is False
    assert out.answer == ""
    assert [ok for _, ok in out.verdicts] == [False] * 4


# --------------------------------------------------------------------------
# 2. The abstention threshold at its boundary
# --------------------------------------------------------------------------

FOUR = "Alpha one. Beta two. Gamma three. Delta four."


@pytest.mark.asyncio
async def test_exactly_at_keep_ratio_ships():
    judge = ScriptedJudge(["SUPPORTED", "UNSUPPORTED", "SUPPORTED", "UNSUPPORTED"])
    out = await validate_grounding(FOUR, SOURCES, llm=judge, min_keep_ratio=0.5)
    assert out.grounded is True
    assert out.answer == "Alpha one. Gamma three."


@pytest.mark.asyncio
async def test_just_below_keep_ratio_abstains():
    judge = ScriptedJudge(["SUPPORTED", "UNSUPPORTED", "SUPPORTED", "UNSUPPORTED"])
    out = await validate_grounding(FOUR, SOURCES, llm=judge, min_keep_ratio=0.51)
    assert out.grounded is False
    assert out.answer == ""


@pytest.mark.asyncio
async def test_nothing_kept_abstains_even_at_zero_ratio():
    """A ratio of 0.0 must not ship an empty answer as 'grounded'."""
    judge = ScriptedJudge(["UNSUPPORTED"])
    out = await validate_grounding("Only sentence.", SOURCES, llm=judge, min_keep_ratio=0.0)
    assert out.grounded is False
    assert out.answer == ""
    assert out.removed == ["Only sentence."]


@pytest.mark.asyncio
async def test_default_keep_ratio_is_the_measured_one():
    assert MIN_KEEP_RATIO == 0.5


# --------------------------------------------------------------------------
# 3. Verdicts stay paired with their sentences under concurrency
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_verdicts_stay_paired_with_sentences():
    """The first sentence finishes last; its verdict must still land on it."""
    judge = ScriptedJudge(
        ["UNSUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"],
        delays=[0.05, 0.0, 0.0, 0.0],
    )
    out = await validate_grounding(FOUR, SOURCES, llm=judge, concurrency=4)
    assert out.verdicts[0] == ("Alpha one.", False)
    assert out.removed == ["Alpha one."]
    assert out.answer == "Beta two. Gamma three. Delta four."


# --------------------------------------------------------------------------
# 4. The strings the published rates are properties of
# --------------------------------------------------------------------------


def test_verifier_prompt_is_the_measured_one():
    """If this fails, the operating point changed: re-measure, do not re-pin."""
    digest = hashlib.sha256(GROUNDING_VERIFIER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == "a678f8e11284840205ac07a4a0dc9c14dafdae60ef4cf19f1e1ab0cf626f94b5"


def test_measured_model_is_the_dated_snapshot():
    """Not a prefix check: a different dated snapshot is a different judge."""
    assert MEASURED_MODEL == "claude-sonnet-4-5-20250929"


def test_runtime_version_matches_package_metadata():
    assert bearout.__version__ == importlib.metadata.version("bearout")
