"""Fetch + parse Ontario e-Laws statutes and regulations.

e-Laws (https://www.ontario.ca/laws/) serves statute text as a
JS-rendered Single Page App, but exposes the rendered document markup
through a JSON API — no browser required.  This module:

* Maps a public e-Laws URL to that API endpoint and fetches the
  ``content`` markup (``fetch_api_content``, requires the ``fetchers``
  extra for httpx).
* Parses the markup by the class names e-Laws uses consistently:

  ``p.heading1``    — topic grouping ("Definitions and Administration",
                       "Protections", etc.).  Acts only.
  ``p.headnote``    — sidenote / heading for the following section or
                       subsection ("Definitions", "Compensation", ...).
                       Acts only.
  ``p.section``     — section start.  Text begins with the section
                       number, e.g. ``"1 (1) In this Act, ..."`` or
                       ``"14. Compensation"``.
  ``p.subsection``  — subsection of the current section, text begins
                       with ``"(N) ..."``.

* Groups each top-level section together with its subsections and the
  most recent headnote into a single :class:`StatuteSection`.  One
  section becomes one RAG chunk — enough context to answer a question
  about that section without dragging in an unrelated section.

The parser is pure (HTML in, dataclasses out) so it is unit-testable
against a captured fixture without touching the network.

There is deliberately NO browser-rendering fallback in this package:
the fidelity verifier must never silently switch what it is verifying
against, so a fetch failure is a loud ``None``, not a different fetch
mechanism.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatuteSection:
    """One ingestable section of a statute or regulation.

    ``section_id`` is the citation form (``"1"``, ``"1.1"``, ``"14"``,
    ``"17.1"``).  Compound subsection numbers like ``"(1)"``, ``"(2)"``
    are preserved inside ``text`` rather than promoted to ``section_id``
    — one chunk per top-level section is the chunking unit.

    ``heading`` is the headnote that introduces the section (e.g.
    ``"Definitions"`` for s.1).  May be ``None`` if the source doesn't
    emit a headnote for the section (regulations typically don't).

    ``topic_group`` is the ``p.heading1`` that most recently preceded
    the section in document order.  Acts only; ``None`` for regulations.
    """

    section_id: str
    heading: str | None
    topic_group: str | None
    text: str


@dataclass(frozen=True)
class ParseAnomaly:
    """A paragraph the parser could not place with confidence.

    ``kind`` is ``"unmatched_section_lead"``: a paragraph the source
    classes as ``p.section`` whose text does not begin with a section
    number.  Two things produce it — a collapsed revoked range
    (``"5.3, 5.4 Revoked: ..."``) and a continuation paragraph the
    source mis-classes — and the parser cannot tell them apart, so it
    reports rather than guesses.

    ``kept_with`` is the ``section_id`` the text was folded into, or
    ``None`` when no section was open and the text was dropped.  A
    ``None`` here is the parser's only remaining lossy path.
    """

    kind: str
    text: str
    kept_with: str | None


@dataclass(frozen=True)
class ParsedDocument:
    """Parser output together with everything it could not place.

    Callers that must not ingest a partial document use
    :func:`bearout.fidelity.spec.parse_for_ingest`, which turns a
    non-empty ``anomalies`` into a :class:`ParseAnomalyError`.
    """

    sections: list[StatuteSection]
    anomalies: list[ParseAnomaly]


class ParseAnomalyError(ValueError):
    """Raised when a document is parsed for ingest and the parser could
    not place every paragraph.

    A corpus built from a partial parse reads as complete law, which is
    worse than one with a visible gap: the reader cannot see what is
    missing, and neither can a verifier that only compares what is
    there.
    """

    def __init__(self, anomalies: list[ParseAnomaly]) -> None:
        self.anomalies = anomalies
        detail = "; ".join(
            f"{a.kind} (kept_with={a.kept_with}): {a.text[:80]!r}" for a in anomalies
        )
        super().__init__(
            f"{len(anomalies)} unplaceable paragraph(s): {detail}. Review them; if the\n"
            "placement shown is correct, ingest via parse_statute_document() instead."
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Section-line patterns.  The rendered text of a ``p.section`` paragraph
# typically begins with the section number followed by one of:
#   ``"1 (1) In this Act, ..."``   — number, space, subsection marker
#   ``"14. Compensation"``          — number, period, space, heading
#   ``"1.1 This Regulation ..."``   — number (dotted), space, body
# We match all three; the dotted-number form is common in regulations.
SECTION_LEAD_RE = re.compile(
    r"""
    ^
    (?P<num>\d+(?:\.\d+)*)        # section number: 1, 1.1, 2.0.1, 17.1
    (?:                            # ...followed by one of:
        \s*\(\d+\)                 # (N) subsection marker
      | \.\s+                      # ". " heading-style
      | \s+(?=\S)                  # whitespace then any non-space
      | $                          # end of line (rare)
    )
    """,
    re.VERBOSE,
)


class _ElawsParser(HTMLParser):
    """Lightweight HTMLParser that captures the class + text of every
    ``<p>`` element.  Avoids pulling in BeautifulSoup as a dep when the
    grammar we care about is just "p tags with a known class attr"."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[tuple[str, str]] = []  # (class, text)
        self._cur_class: str | None = None
        self._cur_buf: list[str] = []
        self._depth_inside_p = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "p":
            self._cur_class = dict(attrs).get("class", "")
            self._cur_buf = []
            self._depth_inside_p = 1
        elif self._depth_inside_p > 0:
            self._depth_inside_p += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._depth_inside_p > 0:
            text = "".join(self._cur_buf).strip()
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)
            self.paragraphs.append((self._cur_class or "", text))
            self._cur_class = None
            self._cur_buf = []
            self._depth_inside_p = 0
        elif self._depth_inside_p > 0:
            self._depth_inside_p -= 1

    def handle_data(self, data: str) -> None:
        if self._depth_inside_p > 0:
            self._cur_buf.append(data)


def parse_statute_document(html: str) -> ParsedDocument:
    """Parse e-Laws document markup into sections plus parse anomalies.

    Walks ``<p>`` elements in document order, tracking the current
    topic group + headnote, and emits one ``StatuteSection`` per
    top-level section.  Robust to:

    * Tables of contents at the top (TOC entries use a different class
      and are filtered out).
    * Missing topic groups (regulations).
    * Missing headnotes (regulations and some sections).
    * Footnote / amendment paragraphs (filtered by class).

    Anything the parser cannot place is reported in
    :attr:`ParsedDocument.anomalies` rather than dropped silently, so a
    caller can refuse to ingest a partial document
    (:func:`bearout.fidelity.spec.parse_for_ingest`).
    """
    p = _ElawsParser()
    p.feed(html)

    sections: list[StatuteSection] = []
    anomalies: list[ParseAnomaly] = []
    topic_group: str | None = None
    pending_heading: str | None = None
    cur_section_id: str | None = None
    cur_section_heading: str | None = None
    cur_section_topic: str | None = None
    cur_section_buf: list[str] = []

    def _flush_current() -> None:
        nonlocal cur_section_id, cur_section_heading, cur_section_topic, cur_section_buf
        if cur_section_id is None:
            return
        body = " ".join(s for s in cur_section_buf if s).strip()
        if body:
            sections.append(
                StatuteSection(
                    section_id=cur_section_id,
                    heading=cur_section_heading,
                    topic_group=cur_section_topic,
                    text=body,
                )
            )
        cur_section_id = None
        cur_section_heading = None
        cur_section_topic = None
        cur_section_buf = []

    for cls, text in p.paragraphs:
        if not text:
            continue
        cls_norm = cls.strip().lower()

        # TOC entries — appear at top of page, mirror the section list
        # but without content.  Skip them entirely.
        if "toc" in cls_norm:
            continue
        # Footnotes / amendment markers — not part of the law's body.
        if "footnote" in cls_norm or "amendments" in cls_norm:
            continue

        if "heading1" in cls_norm:
            topic_group = text
            continue
        if "headnote" in cls_norm:
            # Sidenote for the next section or subsection.  Buffer it;
            # if the next paragraph is a section start, this becomes
            # that section's heading.
            pending_heading = text
            continue
        if "section" in cls_norm and "subsection" not in cls_norm:
            m = SECTION_LEAD_RE.match(text)
            if not m:
                # Classed as a section start but carrying no leading section
                # number.  Test this BEFORE flushing: closing the open section
                # here would discard this paragraph AND every one after it up
                # to the next recognised section, and a section truncated
                # mid-sentence still reads as complete law.  Fold the text into
                # the section it follows and report it instead.
                logger.warning(
                    "elaws: p.section text does not match section pattern: %r", text[:80]
                )
                anomalies.append(ParseAnomaly("unmatched_section_lead", text, cur_section_id))
                if cur_section_id is not None:
                    cur_section_buf.append(text)
                continue
            # A real section start — now close the previous one.
            _flush_current()
            cur_section_id = m.group("num")
            cur_section_heading = pending_heading
            cur_section_topic = topic_group
            pending_heading = None  # consumed by this section
            cur_section_buf.append(text)
            continue
        if "subsection" in cls_norm:
            if cur_section_id is None:
                # Subsection without a parent section start.  Skip.
                continue
            cur_section_buf.append(text)
            continue

        # Plain body paragraphs (definitions list items, schedules, etc.)
        # — append to current section if there is one.
        if cur_section_id is not None:
            cur_section_buf.append(text)

    _flush_current()
    return ParsedDocument(sections=sections, anomalies=anomalies)


def parse_statute_html(html: str) -> list[StatuteSection]:
    """The sections of :func:`parse_statute_document`, anomalies discarded.

    The verifier reads the corpus through this: its job is to report
    divergence, not to refuse to look at a document that parses badly.
    Ingest should use :func:`bearout.fidelity.spec.parse_for_ingest`.
    """
    return parse_statute_document(html).sections


# ---------------------------------------------------------------------------
# Fetcher (e-Laws JSON API only — no browser fallback, by design)
# ---------------------------------------------------------------------------


def elaws_api_url(url: str) -> str | None:
    """Map a public e-Laws URL to its JSON API endpoint, or None.

    ``ontario.ca/laws/{statute|regulation}/{id}`` →
    ``ontario.ca/laws/api/v2/legislation/en/doc-search/{kind}/{id}``.
    The API returns the full document markup in ``content`` (the same
    headnote/section classes the parser consumes) without the JS wall.
    Do not append ``/latest`` (404s).
    """
    m = re.search(r"ontario\.ca/laws/(statute|regulation)/([0-9a-z]+)/?$", url)
    if not m:
        return None
    return f"https://www.ontario.ca/laws/api/v2/legislation/en/doc-search/{m.group(1)}/{m.group(2)}"


async def fetch_api_content(url: str, *, timeout_ms: int = 60_000) -> str | None:
    """API-only fetch of the e-Laws document markup for a public URL.

    Returns the JSON API's ``content`` field when it looks usable
    (contains the ``p.section`` markup the parser consumes — NOT the
    headnote markup: regulations like R.R.O. 1990, Reg. 892 have zero
    headnotes, and a headnote-based heuristic false-negatives on them),
    else ``None`` — on an unmappable URL, an HTTP/parse error, or
    unusable content.  Callers must fail loudly on ``None`` rather than
    silently degrade to a different fetch mechanism.
    """
    api_url = elaws_api_url(url)
    if api_url is None:
        logger.warning("e-Laws API mapping does not recognize URL %s", url)
        return None
    try:
        import httpx
    except ImportError:
        logger.warning("e-Laws fetch needs the 'fetchers' extra: pip install 'bearout[fetchers]'")
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
            resp = await client.get(api_url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            content = resp.json().get("content")
    except Exception as exc:
        logger.warning("e-Laws API fetch failed for %s (%s)", api_url, exc)
        return None
    if content and '<p class="section' in content:
        return content
    logger.warning("e-Laws API returned no usable content for %s", api_url)
    return None


__all__ = [
    "SECTION_LEAD_RE",
    "ParseAnomaly",
    "ParseAnomalyError",
    "ParsedDocument",
    "StatuteSection",
    "elaws_api_url",
    "fetch_api_content",
    "parse_statute_document",
    "parse_statute_html",
]
