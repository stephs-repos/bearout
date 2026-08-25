"""The ingest contract: what a corpus chunk for a document section looks like.

The verifier reconstructs the EXACT text your ingest stored — so the label
and chunk construction live here, shared between your ingest pipeline and
the verifier, and can never drift apart.

Ingest through :func:`parse_for_ingest`: it refuses a document the parser
could not fully place, on the principle that a corpus with a hidden gap is
worse than no corpus, because it reads as complete law.
"""

from __future__ import annotations

from dataclasses import dataclass

from bearout.fidelity.sources.elaws import (
    ParseAnomalyError,
    StatuteSection,
    parse_statute_document,
)


@dataclass(frozen=True)
class DocSpec:
    """One verifiable document (statute, regulation, standard, ...)."""

    doc_id: str
    title: str
    url: str
    fixture_filename: str | None = None  # for offline runs
    # Ingest only the named section_ids (None = whole document).  Lets a
    # targeted subset of a large act enter the corpus without dragging in
    # hundreds of off-topic sections.  On the ingest side a filter entry
    # that matches no parsed section fails loudly (apply_section_filter);
    # the verifier reports it instead of raising.
    section_filter: frozenset[str] | None = None


def section_label(spec: DocSpec, sec: StatuteSection) -> str:
    """The label prepended to the embedded chunk text
    (headnote > topic group > document title)."""
    label_parts: list[str] = []
    if sec.heading:
        label_parts.append(sec.heading)
    if sec.topic_group and sec.topic_group != sec.heading:
        label_parts.append(sec.topic_group)
    return " — ".join(label_parts) if label_parts else spec.title


def chunk_text(spec: DocSpec, sec: StatuteSection) -> str:
    """The exact chunk text ingest stores for one section:
    ``{label}\\n\\n{body}``."""
    return f"{section_label(spec, sec)}\n\n{sec.text}".strip()


def apply_section_filter(spec: DocSpec, sections: list[StatuteSection]) -> list[StatuteSection]:
    """Restrict parsed sections to ``spec.section_filter`` (no-op when None).

    Fails loudly if any named section is missing from the parse — a filter
    typo or an upstream amendment must never become a silent partial
    ingest (parsers that silently drop entries are the bug class the
    fidelity verifier exists to catch).
    """
    if spec.section_filter is None:
        return sections
    kept = [s for s in sections if s.section_id in spec.section_filter]
    missing = spec.section_filter - {s.section_id for s in kept}
    if missing:
        raise ValueError(f"{spec.doc_id}: section_filter ids missing from parse: {sorted(missing)}")
    return kept


def parse_for_ingest(spec: DocSpec, html: str) -> list[StatuteSection]:
    """Parse a document FOR INGEST, refusing anything the parser could not place.

    The loud counterpart to
    :func:`bearout.fidelity.sources.elaws.parse_statute_html`.  Two ways
    a partial document is stopped from becoming a corpus:

    * a paragraph the parser could not place raises
      :class:`ParseAnomalyError`, and
    * a ``section_filter`` id missing from the parse raises
      ``ValueError`` (:func:`apply_section_filter`).

    A caller who has reviewed the anomalies and accepts them ingests via
    :func:`parse_statute_document` instead.  That is deliberately a
    different call rather than a flag on this one, so accepting an
    imperfect parse stays visible in the calling code rather than
    becoming a default nobody revisits.
    """
    doc = parse_statute_document(html)
    if doc.anomalies:
        raise ParseAnomalyError(doc.anomalies)
    return apply_section_filter(spec, doc.sections)


__all__ = [
    "DocSpec",
    "apply_section_filter",
    "chunk_text",
    "parse_for_ingest",
    "section_label",
]
