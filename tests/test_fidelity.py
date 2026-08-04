"""Fidelity verifier: pure fixture-driven tests — no network, no DB.

The "corpus" side is reconstructed through the REAL ingest contract
(``apply_section_filter`` + ``chunk_text``) so the verifier is tested
against exactly what a faithful ingest stores; the source side is the
committed HTML fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import substantiate.fidelity.run as run
import substantiate.fidelity.verify as vsf
from substantiate.fidelity.sources.elaws import parse_statute_html
from substantiate.fidelity.spec import DocSpec, apply_section_filter, chunk_text, section_label

FIXTURE_DIR = Path(__file__).parent / "fixtures"

SYNTH = DocSpec(
    doc_id="test.synthetic-act",
    title="Synthetic Act, 2026",
    url="https://www.ontario.ca/laws/statute/00x00",
    fixture_filename="synthetic-act.html",
)
OREG = DocSpec(
    doc_id="test.oreg-242-21",
    title="O. Reg. 242/21 — Mediation Prior to Notice of Decision",
    url="https://www.ontario.ca/laws/regulation/210242",
    fixture_filename="oreg-242-21.html",
)
ROSTER = [SYNTH, OREG]


def _fixture_html(spec: DocSpec) -> str:
    return (FIXTURE_DIR / spec.fixture_filename).read_text(encoding="utf-8")


def faithful_corpus(spec: DocSpec, html: str) -> dict[str, str]:
    """section_id -> stored chunk text, via the real ingest contract
    (including last-write-wins on duplicate section ids)."""
    sections = apply_section_filter(spec, parse_statute_html(html))
    return {sec.section_id: chunk_text(spec, sec) for sec in sections}


@pytest.fixture(scope="module")
def synth_html() -> str:
    return _fixture_html(SYNTH)


@pytest.fixture(scope="module")
def synth_corpus(synth_html) -> dict[str, str]:
    return faithful_corpus(SYNTH, synth_html)


class TestSpec:
    def test_section_label_prefers_headnote_then_topic_then_title(self, synth_html):
        sections = parse_statute_html(synth_html)
        labels = [section_label(SYNTH, s) for s in sections]
        assert labels == [
            "Definitions — Definitions and Administration",
            "Compensation — Definitions and Administration",
            "Protections",
            "Protections",
        ]
        # A regulation with no headnotes/topics falls back to the title.
        oreg_sections = parse_statute_html(_fixture_html(OREG))
        assert any(section_label(OREG, s) == OREG.title for s in oreg_sections)

    def test_apply_section_filter_fails_loudly_on_missing_id(self, synth_html):
        spec = DocSpec(
            doc_id=SYNTH.doc_id,
            title=SYNTH.title,
            url=SYNTH.url,
            section_filter=frozenset({"1", "9999"}),
        )
        with pytest.raises(ValueError, match="9999"):
            apply_section_filter(spec, parse_statute_html(synth_html))


class TestSplitLabel:
    @pytest.mark.parametrize("spec", ROSTER, ids=lambda s: s.doc_id)
    def test_round_trip_on_every_fixture(self, spec):
        """label + separator + body reassembles the stored chunk, and the
        body never contains the separator (parser output has no newlines)."""
        html = _fixture_html(spec)
        for sec in apply_section_filter(spec, parse_statute_html(html)):
            stored = chunk_text(spec, sec)
            label, body = vsf.split_label(stored)
            assert "\n\n" not in body
            assert "\n" not in sec.text
            assert f"{label}\n\n{body}" == stored
            assert body == sec.text
            assert label == section_label(spec, sec)

    def test_no_separator_treated_as_all_body(self):
        assert vsf.split_label("just a body") == ("", "just a body")


@pytest.mark.parametrize("spec", ROSTER, ids=lambda s: s.doc_id)
def test_expected_chunks_matches_ingest_contract(spec):
    """The L1 reference is locked to the ingest contract — same keys, same
    text, including last-write-wins and section filters."""
    html = _fixture_html(spec)
    sections = parse_statute_html(html)
    assert vsf.expected_chunks(spec, sections) == faithful_corpus(spec, html)


class TestL1:
    def test_faithful_corpus_is_all_ok(self, synth_html, synth_corpus):
        findings = vsf.verify_doc(SYNTH, synth_html, synth_corpus)
        assert findings, "expected one finding per section"
        assert {f.verdict for f in findings} == {"ok"}
        assert {f.section_id for f in findings} == set(synth_corpus)

    def test_mutated_body_yields_one_text_mismatch(self, synth_html, synth_corpus):
        sid = sorted(synth_corpus)[0]
        corpus = dict(synth_corpus)
        corpus[sid] = corpus[sid].replace("promise", "premise", 1)
        findings = vsf.verify_doc(SYNTH, synth_html, corpus)
        mismatches = [f for f in findings if f.verdict == "text_mismatch"]
        assert [f.section_id for f in mismatches] == [sid]
        assert "body" in mismatches[0].detail
        assert mismatches[0].diff

    def test_label_only_mutation_reports_label_drift(self, synth_html, synth_corpus):
        sid, stored = next(iter(sorted(synth_corpus.items())))
        label, body = vsf.split_label(stored)
        corpus = dict(synth_corpus)
        corpus[sid] = f"WRONG {label}\n\n{body}"
        findings = vsf.verify_doc(SYNTH, synth_html, corpus)
        mismatches = [f for f in findings if f.verdict == "text_mismatch"]
        assert [f.section_id for f in mismatches] == [sid]
        assert "label" in mismatches[0].detail
        assert "body" not in mismatches[0].detail

    def test_diff_truncates_at_max_diff_lines(self, synth_html, synth_corpus):
        sid = max(synth_corpus, key=lambda k: len(synth_corpus[k]))
        corpus = dict(synth_corpus)
        # Reverse the body's words — a maximally noisy diff.
        label, body = vsf.split_label(corpus[sid])
        corpus[sid] = f"{label}\n\n{' '.join(reversed(body.split(' ')))}"
        findings = vsf.verify_doc(SYNTH, synth_html, corpus, max_diff_lines=5)
        mismatch = next(f for f in findings if f.verdict == "text_mismatch")
        lines = mismatch.diff.splitlines()
        assert len(lines) == 6  # 5 diff lines + truncation marker
        assert "more diff lines" in lines[-1]


class TestL3:
    def test_deleted_section_is_missing_in_corpus(self, synth_html, synth_corpus):
        sid = sorted(synth_corpus)[1]
        corpus = {k: v for k, v in synth_corpus.items() if k != sid}
        findings = vsf.verify_doc(SYNTH, synth_html, corpus)
        missing = [f for f in findings if f.verdict == "missing_in_corpus"]
        assert [f.section_id for f in missing] == [sid]
        assert "parser extracts it" in missing[0].detail

    def test_fake_section_is_extra_in_corpus(self, synth_html, synth_corpus):
        corpus = dict(synth_corpus)
        corpus["999"] = "Fake\n\n999 A section the official text has never heard of."
        findings = vsf.verify_doc(SYNTH, synth_html, corpus)
        extra = [f for f in findings if f.verdict == "extra_in_corpus"]
        assert [f.section_id for f in extra] == ["999"]

    def test_empty_corpus_flags_every_section_with_hint(self, synth_html, synth_corpus):
        findings = vsf.verify_doc(SYNTH, synth_html, {})
        assert {f.verdict for f in findings} == {"missing_in_corpus"}
        assert {f.section_id for f in findings} == set(synth_corpus)
        assert all("zero corpus rows" in f.detail for f in findings)

    def test_section_filter_scopes_the_inventory(self, synth_html):
        spec = DocSpec(
            doc_id=SYNTH.doc_id,
            title=SYNTH.title,
            url=SYNTH.url,
            section_filter=frozenset({"1", "2"}),
        )
        corpus = faithful_corpus(spec, synth_html)
        assert set(corpus) == {"1", "2"}
        findings = vsf.verify_doc(spec, synth_html, corpus)
        assert {f.verdict for f in findings} == {"ok"}
        assert {f.section_id for f in findings} == {"1", "2"}

    def test_filter_id_absent_from_source_and_corpus_is_fetch_error(self, synth_html):
        spec = DocSpec(
            doc_id=SYNTH.doc_id,
            title=SYNTH.title,
            url=SYNTH.url,
            section_filter=frozenset({"1", "9999"}),
        )
        corpus = {k: v for k, v in faithful_corpus(SYNTH, synth_html).items() if k == "1"}
        findings = vsf.verify_doc(spec, synth_html, corpus)
        errors = [f for f in findings if f.verdict == "fetch_error"]
        assert [f.section_id for f in errors] == ["9999"]
        assert "source restructured" in errors[0].detail


class TestL2:
    @pytest.mark.parametrize("spec", ROSTER, ids=lambda s: s.doc_id)
    def test_all_faithful_bodies_are_contained(self, spec):
        """Every real section body survives the independent extraction —
        including sections with interleaved headnote/amendments paragraphs
        (the contiguity hazard the shared skip-policy exists for)."""
        html = _fixture_html(spec)
        haystack = vsf.norm(vsf.independent_body_text(html))
        for stored in faithful_corpus(spec, html).values():
            body = vsf.split_label(stored)[1]
            assert vsf.norm(body) in haystack

    def test_word_swap_fails_containment(self, synth_html, synth_corpus):
        sid = sorted(synth_corpus)[0]
        corpus = dict(synth_corpus)
        label, body = vsf.split_label(corpus[sid])
        words = body.split(" ")
        words[0], words[-1] = words[-1], words[0]
        corpus[sid] = f"{label}\n\n{' '.join(words)}"
        findings = vsf.verify_doc(SYNTH, synth_html, corpus)
        assert any(f.verdict == "containment_fail" and f.section_id == sid for f in findings)

    def test_stable_parser_bug_signature(self, synth_html, synth_corpus, monkeypatch):
        """L1 passes (stored == expected) but the independent extraction
        disagrees — the exact case L2 exists for."""
        monkeypatch.setattr(vsf, "independent_body_text", lambda html: "unrelated text")
        findings = vsf.verify_doc(SYNTH, synth_html, synth_corpus)
        fails = [f for f in findings if f.verdict == "containment_fail"]
        assert len(fails) == len(synth_corpus)
        assert all("stable-parser-bug signature" in f.detail for f in fails)


@pytest.mark.parametrize("spec", ROSTER, ids=lambda s: s.doc_id)
def test_inventory_agrees_with_parser_on_all_fixtures(spec):
    """The regex inventory and the HTMLParser must see the same section-id
    set on every fixture.  A future divergence here is a real signal (one
    of the two extractors is wrong), not test flake."""
    html = _fixture_html(spec)
    inventory = vsf.independent_section_inventory(html)
    parsed = {s.section_id for s in parse_statute_html(html)}
    if spec.section_filter is not None:
        inventory &= spec.section_filter
        parsed &= spec.section_filter
    assert inventory == parsed


class TestEndToEndOffline:
    def _run(self, corpus_by_doc, argv):
        return run.main(
            argv,
            roster=ROSTER,
            corpus_reader=lambda doc_id: corpus_by_doc[doc_id],
            fixtures_dir=FIXTURE_DIR,
        )

    def test_faithful_run_exits_zero_and_writes_report(self, tmp_path, synth_corpus):
        out = tmp_path / "reports" / "fidelity.json"
        rc = self._run(
            {SYNTH.doc_id: synth_corpus},
            ["--doc-id", SYNTH.doc_id, "--from-fixture", "--json-out", str(out)],
        )
        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["meta"]["mode"] == "fixture"
        assert payload["meta"]["docs"] == 1
        doc = payload["docs"][SYNTH.doc_id]
        assert doc["counts"] == {"ok": len(synth_corpus)}
        assert set(doc["sections"]) == set(synth_corpus)

    def test_mutated_corpus_exits_five(self, synth_corpus):
        sid = sorted(synth_corpus)[0]
        corpus = dict(synth_corpus)
        corpus[sid] = corpus[sid].replace("promise", "premise", 1)
        rc = self._run({SYNTH.doc_id: corpus}, ["--doc-id", SYNTH.doc_id, "--from-fixture"])
        assert rc == 5

    def test_fetch_failure_exits_two(self, monkeypatch, synth_corpus):
        async def _no_html(spec, *, from_fixture, fixtures_dir):
            return None

        monkeypatch.setattr(run, "_fetch_html", _no_html)
        rc = self._run({SYNTH.doc_id: synth_corpus}, ["--doc-id", SYNTH.doc_id, "--from-fixture"])
        assert rc == 2

    def test_unknown_doc_id_exits_one(self, synth_corpus):
        rc = self._run({SYNTH.doc_id: synth_corpus}, ["--doc-id", "not.a.real.doc"])
        assert rc == 1
