"""Sentence-level grounding gate — pipeline logic with a stub judge.

No network: the stub returns canned verdicts keyed by sentence. The
splitter tests use the real citation shapes that motivated it.
"""

from __future__ import annotations

import pytest

from bearout.gate import (
    GROUNDING_VERIFIER_SYSTEM_PROMPT,
    split_sentences,
    validate_grounding,
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


class StubJudge:
    """Returns UNSUPPORTED for sentences in ``unsupported``; records calls."""

    def __init__(self, unsupported: set[str] | None = None, raises: bool = False):
        self.unsupported = unsupported or set()
        self.raises = raises
        self.seen: list[str] = []
        self.cache_prefixes: list[str | None] = []

    async def chat(self, *, system_prompt, user_message, temperature=0.0, cache_prefix=None):
        assert system_prompt == GROUNDING_VERIFIER_SYSTEM_PROMPT
        if self.raises:
            raise RuntimeError("judge unavailable")
        sentence = user_message.split("SENTENCE:\n", 1)[1].strip()
        self.seen.append(sentence)
        self.cache_prefixes.append(cache_prefix)
        verdict = "UNSUPPORTED" if sentence in self.unsupported else "SUPPORTED"
        return {"content": verdict}


class TestSplitSentences:
    def test_plain_prose(self):
        assert split_sentences(ANSWER) == SENTENCES

    def test_statutory_citation_is_not_shattered(self):
        """'Reg. 892 s. 4.4(2)' must survive as one sentence — naive splitting
        produces fragments like '892 s.' that any judge rightly rejects."""
        text = "You must report the defect under Reg. 892 s. 4.4(2) within 30 days."
        assert split_sentences(text) == [text]

    def test_terminal_punctuation_inside_parenthetical(self):
        text = (
            "Tarion publishes guidance on this "
            "(see Tarion's 'Problem with your appliances? Who you gonna call?' page). "
            "The warranty still applies."
        )
        out = split_sentences(text)
        assert len(out) == 2
        assert out[0].endswith("page).")
        assert "Who you gonna call?" in out[0]

    def test_multi_sentence_with_abbreviations(self):
        text = "See ss. 1 and 2. The Corporation shall pay. Art. 5 applies."
        assert split_sentences(text) == [
            "See ss. 1 and 2.",
            "The Corporation shall pay.",
            "Art. 5 applies.",
        ]

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_unbalanced_paren_does_not_swallow_the_rest(self):
        """A stray ')' must not make every later boundary look 'inside'."""
        text = "First sentence). Second sentence. Third sentence."
        assert len(split_sentences(text)) == 3


@pytest.mark.asyncio
async def test_all_supported_returns_answer_untouched():
    judge = StubJudge()
    outcome = await validate_grounding(ANSWER, SOURCES, llm=judge)
    assert outcome.grounded is True
    assert outcome.answer == ANSWER
    assert outcome.removed == []
    assert [s for s, _ in outcome.verdicts] == SENTENCES
    assert all(ok for _, ok in outcome.verdicts)
    assert judge.seen == SENTENCES


@pytest.mark.asyncio
async def test_unsupported_sentence_is_stripped():
    judge = StubJudge(unsupported={SENTENCES[1]})
    outcome = await validate_grounding(ANSWER, SOURCES, llm=judge)
    assert outcome.grounded is True
    assert outcome.removed == [SENTENCES[1]]
    assert SENTENCES[1] not in outcome.answer
    assert outcome.answer == f"{SENTENCES[0]} {SENTENCES[2]}"
    assert outcome.verdicts[1] == (SENTENCES[1], False)


@pytest.mark.asyncio
async def test_below_keep_ratio_abstains_entirely():
    """A gutted answer is more dangerous than no answer."""
    judge = StubJudge(unsupported={SENTENCES[0], SENTENCES[1]})
    outcome = await validate_grounding(ANSWER, SOURCES, llm=judge)
    assert outcome.grounded is False
    assert outcome.answer == ""
    assert len(outcome.removed) == 2
    # Verdicts are still reported so the abstain is explainable.
    assert len(outcome.verdicts) == 3


@pytest.mark.asyncio
async def test_all_unsupported_abstains():
    judge = StubJudge(unsupported=set(SENTENCES))
    outcome = await validate_grounding(ANSWER, SOURCES, llm=judge)
    assert outcome.grounded is False
    assert outcome.answer == ""


@pytest.mark.asyncio
async def test_judge_failure_fails_closed():
    """An unavailable judge must not fail open."""
    judge = StubJudge(raises=True)
    outcome = await validate_grounding(ANSWER, SOURCES, llm=judge)
    assert outcome.grounded is False
    assert outcome.answer == ""
    assert outcome.removed == SENTENCES


@pytest.mark.asyncio
async def test_empty_answer_is_passed_through():
    judge = StubJudge()
    outcome = await validate_grounding("   ", SOURCES, llm=judge)
    assert outcome.grounded is True
    assert outcome.answer == ""
    assert judge.seen == []


@pytest.mark.asyncio
async def test_sources_ride_the_cache_prefix():
    """Sources are identical across sentences, so they belong in the
    cacheable prefix — one payment instead of one per sentence."""
    judge = StubJudge()
    await validate_grounding(ANSWER, SOURCES, llm=judge)
    assert len(judge.cache_prefixes) == 3
    assert all(SOURCES in (p or "") for p in judge.cache_prefixes)


@pytest.mark.asyncio
async def test_keep_ratio_is_configurable():
    judge = StubJudge(unsupported={SENTENCES[0], SENTENCES[1]})
    outcome = await validate_grounding(ANSWER, SOURCES, llm=judge, min_keep_ratio=0.3)
    assert outcome.grounded is True
    assert outcome.answer == SENTENCES[2]


@pytest.mark.asyncio
async def test_concurrency_does_not_change_verdicts():
    """Sentences are judged independently; concurrency is a latency knob."""
    seq = await validate_grounding(ANSWER, SOURCES, llm=StubJudge(unsupported={SENTENCES[1]}))
    par = await validate_grounding(
        ANSWER, SOURCES, llm=StubJudge(unsupported={SENTENCES[1]}), concurrency=5
    )
    assert seq.answer == par.answer
    assert seq.verdicts == par.verdicts
