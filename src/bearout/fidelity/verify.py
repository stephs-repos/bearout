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

L4 — coverage.  L1 compares stored text against a rebuild by the same
     parser, and L2 asks whether stored text is a SUBSET of the official
     text.  A truncated section satisfies both: it is byte-identical to
     what that parser produces today, and a truncation is still a
     subset.  L4 asks the other direction — is the official text
     COVERED by what is stored — and reports law-body paragraphs that
     reached no chunk (``incomplete``).  Fidelity is not completeness,
     and only this layer measures the second one.

Verdicts: ``ok`` / ``text_mismatch`` / ``missing_in_corpus`` /
``extra_in_corpus`` / ``containment_fail`` / ``incomplete`` /
``fetch_error``.
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
    {"text_mismatch", "missing_in_corpus", "extra_in_corpus", "containment_fail", "incomplete"}
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


def body_paragraphs_for_coverage(html: str) -> list[str]:
    """Law-body paragraphs a complete corpus must cover (the L4 haystack side).

    Same declarative class-skip policy as L2, taken from the first
    ``p.section`` onward.  Everything before the first section start is
    front matter: e-Laws renders a statute's table of contents as
    ``p.table`` rows that carry no ``toc`` class, and those are not law
    text and belong in no chunk.

    Deliberately parser-free — class names only, no lead-number regex —
    so this layer shares no machinery with the parser it checks.
    """
    paras = [(cls.strip().lower(), text) for cls, text in strip_paragraphs(html) if text]
    start = next(
        (i for i, (cls, _) in enumerate(paras) if "section" in cls and "subsection" not in cls),
        None,
    )
    if start is None:
        return []
    return [text for cls, text in paras[start:] if not _SKIP_CLASS_RE.search(cls)]


def _within(key: tuple, lo: tuple | None, hi: tuple | None) -> bool:
    """Is ``key`` strictly between the two section sort keys?  ``None`` is an
    open bound (document start / document end)."""
    return (lo is None or lo < key) and (hi is None or key < hi)


def coverage_findings(
    spec: DocSpec,
    corpus: dict[str, str],
    paragraphs: list[str],
    *,
    explained_by: set[str] | None = None,
    max_detail_chars: int = 120,
) -> list[SectionFinding]:
    """L4 — every law-body paragraph must appear in some stored chunk.

    Uncovered paragraphs are attributed to the section whose chunk should
    have held them: a stored section sorting inside the gap when there is
    one (its body was edited out from under them), otherwise the last
    section that did cover a paragraph (its chunk stops short).  Two
    deliberate silences:

    * A paragraph found in the corpus but out of document order is
      covered, not a gap.  This layer reports omission, not disorder.
    * A run of uncovered paragraphs is dropped when a section L3 already
      reported ``missing_in_corpus`` falls inside it (``explained_by``).
      An absent section has absent paragraphs; saying so twice, the
      second time blamed on the section before it, is worse than not
      saying it.  What survives is text that vanished with no other
      finding to explain it.
    """
    bodies = {sid: norm(split_label(text)[1]) for sid, text in corpus.items()}
    order = sorted(bodies, key=section_sort_key)
    everything_stored = "\n".join(bodies.values())
    missing_keys = [section_sort_key(sid) for sid in (explained_by or set())]

    owner: str | None = None
    owner_idx = 0
    runs: list[tuple[str | None, list[str], str | None]] = []
    open_run: list[str] = []

    def _close_run(next_owner: str | None) -> None:
        nonlocal open_run
        if open_run:
            runs.append((owner, open_run, next_owner))
            open_run = []

    for text in paragraphs:
        needle = norm(text)
        if not needle:
            continue
        hit = owner if owner is not None and needle in bodies[owner] else None
        if hit is None:
            for i in range(owner_idx, len(order)):
                if needle in bodies[order[i]]:
                    hit, owner_idx = order[i], i
                    break
        if hit is not None:
            _close_run(hit)
            owner = hit
        elif needle not in everything_stored:
            open_run.append(text)
    _close_run(None)

    gaps: dict[tuple[str | None, str], list[str]] = {}
    for before, texts, after in runs:
        lo = section_sort_key(before) if before is not None else None
        hi = section_sort_key(after) if after is not None else None
        if any(_within(k, lo, hi) for k in missing_keys):
            continue  # L3 already reported the absent section that owns this text
        # Prefer a stored section that sorts INSIDE the gap: when s.2's body has
        # been edited, its official paragraph goes uncovered, and blaming that on
        # s.1 — the last section that still matched — points at the wrong chunk.
        inside = [sid for sid in order if _within(section_sort_key(sid), lo, hi)]
        key = (inside[0], "body") if inside else (before, "tail")
        gaps.setdefault(key, []).extend(texts)

    findings: list[SectionFinding] = []
    for (sid, kind), missing in gaps.items():
        chars = sum(len(t) for t in missing)
        if sid is None:
            where = "no stored section precedes them"
        elif kind == "body":
            where = f"the stored body of s.{sid} does not contain them"
        else:
            where = f"the chunk for s.{sid} stops short"
        findings.append(
            SectionFinding(
                spec.doc_id,
                sid if sid is not None else "*",
                "incomplete",
                f"{len(missing)} official paragraph(s), {chars} chars, reached no chunk "
                f"({where}); first: {missing[0][:max_detail_chars]!r}",
            )
        )
    return findings


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
    """Run L3 -> L4 -> L1 -> L2 for one document; one finding per section."""
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

    # --- L4: coverage of the official text by the corpus -------------------
    # Runs before L1/L2 so a section whose chunk stops short is never also
    # reported "ok".  Skipped when the spec restricts ingest to a subset:
    # there, most of the document is uncovered BY DESIGN, and "does the
    # corpus cover the document" is the wrong question to ask of it.
    if spec.section_filter is None:
        findings.extend(
            coverage_findings(
                spec,
                corpus,
                body_paragraphs_for_coverage(html),
                explained_by={f.section_id for f in findings if f.verdict == "missing_in_corpus"},
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
    "body_paragraphs_for_coverage",
    "coverage_findings",
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
