"""Three-layer corpus fidelity verification — pure, DB-free core.

For one document, given (a) its :class:`DocSpec`, (b) today's official
source HTML, and (c) the corpus's stored chunks, run:

L1 — currency + parser drift.  Re-run the parser on the fresh source,
     rebuild the exact chunk text ingest would store
     (``{label}\\n\\n{body}``, last-write-wins on duplicate section ids,
     honouring ``section_filter``), and exact-compare per section id.
     Both sides share one pipeline, so ANY diff is real: either the
     source changed since ingest or the parser changed.

L2 — extraction fidelity (independent).  If ingest used the same parser,
     a stable parser bug reproduces identically under L1.  L2
     re-extracts the document text with a second mechanism (regex over
     ``<p>`` blocks + tag-strip + ``html.unescape``) that shares only
     the DECLARATIVE class-skip policy with the parser, none of its
     machinery.  Every stored section body must appear as a contiguous
     normalized substring of that independently extracted text.
     L1-pass + L2-fail is the signature of a parser bug that survived
     ingest.

L3 — silent drops.  An independent section-id inventory (regex over
     ``p.section`` lead numbers) is compared against the corpus BOTH
     ways: sections the official text has but the corpus lacks
     (``missing_in_corpus``) and corpus sections the official text no
     longer shows (``extra_in_corpus``).

Verdicts: ``ok`` / ``text_mismatch`` / ``missing_in_corpus`` /
``extra_in_corpus`` / ``containment_fail`` / ``fetch_error``.
"""

from __future__ import annotations

import difflib
import html as html_lib
import re
from dataclasses import dataclass, field

from bearout.fidelity.sources.elaws import (
    SECTION_LEAD_RE,
    StatuteSection,
    parse_statute_html,
)
from bearout.fidelity.spec import DocSpec, chunk_text

FIDELITY_VERDICTS = frozenset(
    {"text_mismatch", "missing_in_corpus", "extra_in_corpus", "containment_fail"}
)


# ---------------------------------------------------------------------------
# Independent extraction (L2/L3 side) — regex over <p> blocks, no HTMLParser.
# Shares the parser's declarative class-skip POLICY (which paragraph classes
# are not part of a section body) but none of its machinery: buffering,
# nesting, grouping and entity handling are all re-done differently, which is
# exactly what lets L2/L3 catch bugs in the parser's machinery.
# ---------------------------------------------------------------------------

_P_BLOCK_RE = re.compile(r"<p\b(?P<attrs>[^>]*)>(?P<body>.*?)</p>", re.S | re.I)
_CLASS_RE = re.compile(r"""class\s*=\s*["']([^"']*)["']""", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
# toc/footnote/amendments are skipped outright by the parser; heading1 and
# headnote become topic_group/heading (the label), not section-body text.
_SKIP_CLASS_RE = re.compile(r"toc|footnote|amendments|heading1|headnote", re.I)


def norm(s: str) -> str:
    """House comparison normalizer: collapse whitespace (Unicode ``\\s``
    eats NBSP), lowercase, straighten curly double quotes.  Deliberately
    minimal — over-normalization hides real diffs."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower().replace("“", '"').replace("”", '"')


def strip_paragraphs(html: str) -> list[tuple[str, str]]:
    """(class, text) for every ``<p>`` block, extracted by regex: inner tags
    stripped, entities unescaped, whitespace collapsed."""
    out: list[tuple[str, str]] = []
    for m in _P_BLOCK_RE.finditer(html):
        cls_m = _CLASS_RE.search(m.group("attrs"))
        cls = cls_m.group(1) if cls_m else ""
        text = html_lib.unescape(_TAG_RE.sub("", m.group("body")))
        text = re.sub(r"\s+", " ", text).strip()
        out.append((cls, text))
    return out


def independent_body_text(html: str) -> str:
    """The containment haystack: all paragraph text that the class-skip
    policy counts as law-body, joined in document order."""
    return " ".join(
        text for cls, text in strip_paragraphs(html) if text and not _SKIP_CLASS_RE.search(cls)
    )


def independent_section_inventory(html: str) -> set[str]:
    """Section ids seen in the raw HTML, independent of the parser.

    Mirrors the parser's class test (``section`` in class, ``subsection``
    not in class) and its lead-number regex, nothing else.  Set semantics
    collapse e-Laws' historical/in-force duplicate renderings.
    """
    inventory: set[str] = set()
    for cls, text in strip_paragraphs(html):
        cls_norm = cls.strip().lower()
        if "section" in cls_norm and "subsection" not in cls_norm and text:
            m = SECTION_LEAD_RE.match(text)
            if m:
                inventory.add(m.group("num"))
    return inventory


# ---------------------------------------------------------------------------
# Expected-text reconstruction (L1 side) — mirrors ingest exactly.
# ---------------------------------------------------------------------------


def expected_chunks(spec: DocSpec, sections: list[StatuteSection]) -> dict[str, str]:
    """section_id -> the exact chunk text ingest would store.

    Applies ``section_filter`` non-raising (a filter id missing from the
    parse is reported by ``verify_doc``, not raised — the verifier's job is
    to report divergence, not refuse to look at it) and mirrors ingest's
    last-write-wins on duplicate section ids.
    """
    out: dict[str, str] = {}
    for sec in sections:
        if spec.section_filter is not None and sec.section_id not in spec.section_filter:
            continue
        out[sec.section_id] = chunk_text(spec, sec)
    return out


def split_label(stored: str) -> tuple[str, str]:
    """Split a stored chunk into (label, body).

    Safe because parser output contains no newlines at all (per-paragraph
    ``\\s+`` collapse + single-space joins), so the first ``\\n\\n`` is
    always the ingest-inserted separator.  Defensive fallback: a chunk
    with no separator is treated as all-body.
    """
    if "\n\n" in stored:
        label, body = stored.split("\n\n", 1)
        return label, body
    return "", stored


def word_diff(expected: str, stored: str, max_lines: int) -> str:
    """Truncated unified diff over word tokens."""
    lines = list(
        difflib.unified_diff(
            expected.split(" "),
            stored.split(" "),
            fromfile="expected (source today)",
            tofile="stored (corpus)",
            lineterm="",
            n=2,
        )
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (+{len(lines) - max_lines} more diff lines)"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionFinding:
    doc_id: str
    section_id: str
    verdict: str  # ok | text_mismatch | missing_in_corpus | extra_in_corpus
    #             # | containment_fail | fetch_error
    detail: str = ""
    diff: str = field(default="", repr=False)


def section_sort_key(sid: str) -> tuple:
    """Numeric-aware ordering for ids like '2', '2.0.1', '17.1'."""
    try:
        return tuple(int(p) for p in sid.split("."))
    except ValueError:
        return (float("inf"), sid)


def verify_doc(
    spec: DocSpec,
    html: str,
    corpus: dict[str, str],
    *,
    max_diff_lines: int = 20,
) -> list[SectionFinding]:
    """Run L3 -> L1 -> L2 for one document; one finding per section."""
    findings: list[SectionFinding] = []

    sections = parse_statute_html(html)
    if not sections:
        return [
            SectionFinding(
                spec.doc_id, "*", "fetch_error", "parser emitted zero sections from source HTML"
            )
        ]
    expected = expected_chunks(spec, sections)
    inventory = independent_section_inventory(html)

    empty_corpus_hint = " (doc has zero corpus rows — never ingested?)" if not corpus else ""

    # Respect the spec's section_filter: only the filtered subset was ever
    # meant to be in the corpus.  A filter id absent from today's source HTML
    # means the source was restructured under us — report, don't crash.
    if spec.section_filter is not None:
        for sid in sorted(spec.section_filter - inventory, key=section_sort_key):
            if sid not in corpus:
                findings.append(
                    SectionFinding(
                        spec.doc_id,
                        sid,
                        "fetch_error",
                        "section_filter id absent from both source HTML and corpus — "
                        "source restructured?",
                    )
                )
            # else: it will surface below as extra_in_corpus (corpus holds a
            # section today's official text no longer shows).
        inventory &= spec.section_filter

    # --- L3: inventory comparison, both directions -------------------------
    for sid in sorted(inventory - corpus.keys(), key=section_sort_key):
        parser_sees_it = sid in expected
        detail = (
            "in official text; parser extracts it — corpus is stale or ingest dropped it"
            if parser_sees_it
            else "in official text; current parser ALSO fails to extract it — parser bug"
        ) + empty_corpus_hint
        findings.append(SectionFinding(spec.doc_id, sid, "missing_in_corpus", detail))

    for sid in sorted(corpus.keys() - inventory, key=section_sort_key):
        findings.append(
            SectionFinding(
                spec.doc_id,
                sid,
                "extra_in_corpus",
                "in corpus but not in today's official text — repealed/renumbered "
                "section or corpus contamination",
            )
        )

    # --- L1: exact chunk comparison on the common set ----------------------
    flagged: set[str] = {f.section_id for f in findings}
    haystack = norm(independent_body_text(html))

    for sid in sorted(corpus.keys() & inventory, key=section_sort_key):
        stored = corpus[sid]
        problems: list[SectionFinding] = []

        exp = expected.get(sid)
        if exp is not None and stored != exp:
            exp_label, exp_body = split_label(exp)
            stored_label, stored_body = split_label(stored)
            drift = []
            if exp_label != stored_label:
                drift.append("label")
            if exp_body != stored_body:
                drift.append("body")
            problems.append(
                SectionFinding(
                    spec.doc_id,
                    sid,
                    "text_mismatch",
                    f"{' + '.join(drift) or 'text'} differs from today's official text",
                    word_diff(exp, stored, max_diff_lines),
                )
            )

        # --- L2: independent containment (runs even when L1 passed) --------
        body = split_label(stored)[1]
        if norm(body) not in haystack:
            problems.append(
                SectionFinding(
                    spec.doc_id,
                    sid,
                    "containment_fail",
                    "stored body is not a contiguous substring of the independently "
                    "extracted official text"
                    + (
                        " (L1 passed — stable-parser-bug signature)"
                        if exp is not None and stored == exp
                        else ""
                    ),
                )
            )

        if problems:
            findings.extend(problems)
            flagged.add(sid)
        elif sid not in flagged:
            findings.append(SectionFinding(spec.doc_id, sid, "ok"))

    return findings


__all__ = [
    "FIDELITY_VERDICTS",
    "SectionFinding",
    "expected_chunks",
    "independent_body_text",
    "independent_section_inventory",
    "norm",
    "section_sort_key",
    "split_label",
    "strip_paragraphs",
    "verify_doc",
    "word_diff",
]
