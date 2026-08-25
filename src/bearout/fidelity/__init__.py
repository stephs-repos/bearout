"""Corpus fidelity verification: does your corpus match the law, today?

Three layers, per document (see :mod:`bearout.fidelity.verify` for
the full contract):

L1 — currency + parser drift: re-fetch the official text, rebuild the
     exact chunk text ingest would store, exact-compare per section.
L2 — extraction fidelity: re-extract with a second, independent
     mechanism; every stored section body must appear as a contiguous
     normalized substring of it.  L1-pass + L2-fail is the signature of
     a parser bug that survived ingest.
L3 — silent drops: an independent section inventory compared BOTH ways
     (sections the official text has but the corpus lacks, and corpus
     sections the official text no longer shows).
L4 — coverage: every law-body paragraph of the official text must appear
     in some stored chunk.  L1 compares the parser against itself and L2
     is a subset test, so a truncated section passes both; only L4 asks
     whether what is stored COVERS what the law says.

First authoritative-source adapter: Ontario's e-Laws JSON API
(:mod:`bearout.fidelity.sources.elaws`).
"""

from bearout.fidelity.spec import (
    DocSpec,
    apply_section_filter,
    chunk_text,
    parse_for_ingest,
    section_label,
)
from bearout.fidelity.verify import (
    FIDELITY_VERDICTS,
    SectionFinding,
    body_paragraphs_for_coverage,
    coverage_findings,
    verify_doc,
)

__all__ = [
    "FIDELITY_VERDICTS",
    "DocSpec",
    "SectionFinding",
    "apply_section_filter",
    "body_paragraphs_for_coverage",
    "chunk_text",
    "coverage_findings",
    "parse_for_ingest",
    "section_label",
    "verify_doc",
]
