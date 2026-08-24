"""Corpus fidelity against a live government source. No API key, no cost.

Grounding an answer against a stale index is grounding against nothing.
This example verifies a stored corpus against Ontario's e-Laws API — the
authoritative publisher of the regulation — and shows each of the three
layers catching a defect the previous one structurally cannot.

It runs in two passes:

  Pass 1  A faithful corpus, built through the real ingest contract
          (``chunk_text``).  Every section verifies.  This is the boring,
          correct case, and it is what a healthy scheduled run looks like.

  Pass 2  The same corpus with three realistic defects injected — the
          kinds that occur in practice and that nothing else in a RAG
          stack would notice:

            * a section silently dropped by a parser bug
            * a stored figure that no longer matches the official text
            * a section the corpus still holds after it left the source

Usage::

    uv run python examples/03_fidelity_elaws.py
    uv run python examples/03_fidelity_elaws.py --offline   # cached fixture

    # Or point it at any Ontario statute or regulation:
    uv run python examples/03_fidelity_elaws.py \\
        --url https://www.ontario.ca/laws/statute/90o31 \\
        --title "Ontario New Home Warranties Plan Act"

Exit code is 0 in both passes: the injected defects are the point of the
demo, not a failure of it.  For the real exit-code contract (0 ok / 2
fetch error / 5 fidelity violation), see ``bearout.fidelity.run``.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from bearout.fidelity import DocSpec, SectionFinding, chunk_text, verify_doc
from bearout.fidelity.sources.elaws import (
    SECTION_LEAD_RE,
    elaws_api_url,
    fetch_api_content,
    parse_statute_html,
)

# A short, real Ontario regulation — small enough to read in full, and
# governed by the same e-Laws API as every other Ontario instrument.
OREG = DocSpec(
    doc_id="oreg-242-21",
    title="O. Reg. 242/21 — Mediation Prior to Notice of Decision",
    url="https://www.ontario.ca/laws/regulation/210242",
    fixture_filename="oreg-242-21.html",
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def spec_from_url(url: str, title: str | None) -> DocSpec | None:
    """Build a DocSpec for any e-Laws statute or regulation URL, or None.

    Returns None when the URL is not one the e-Laws adapter can map to its
    JSON API — caught here so the run fails with an explanation rather than
    an opaque fetch error several steps later.
    """
    if elaws_api_url(url) is None:
        return None

    match = re.search(r"/laws/(statute|regulation)/([0-9a-z]+)", url)
    slug = f"{match.group(1)}-{match.group(2)}" if match else "elaws-doc"
    return DocSpec(doc_id=slug, title=title or slug, url=url)


def build_faithful_corpus(spec: DocSpec, html: str) -> dict[str, str]:
    """What a correct ingest would have stored, via the shared contract.

    ``chunk_text`` is the single definition of a stored chunk, imported by
    both the ingest pipeline and the verifier — so the two can never drift
    into disagreeing about what "correct" means.
    """
    return {sec.section_id: chunk_text(spec, sec) for sec in parse_statute_html(html)}


def _bump_a_cross_reference(chunk: str) -> tuple[str, str, str] | None:
    """Alter one multi-digit figure inside a chunk's *body*, if it has one.

    A stored chunk is ``{label}\\n\\n{body}``, and the body opens with its own
    section number.  Both are skipped: editing the label or the section number
    would demonstrate a less interesting failure than a wrong cross-reference
    buried in the prose, which is what real amendment drift looks like.
    """
    sep = chunk.find("\n\n")
    if sep == -1:
        return None
    body_start = sep + 2
    body = chunk[body_start:]

    # SECTION_LEAD_RE is ^-anchored, so it must be matched against the body as
    # its own string — passing an offset to .match() would leave ^ unsatisfied
    # and silently skip the strip.
    lead = SECTION_LEAD_RE.match(body)
    search_from = lead.end() if lead else 0

    match = re.compile(r"\b(\d{2,})\b").search(body, search_from)
    if match is None:
        return None

    was = match.group(1)
    now = str(int(was) + 1)
    start, end = match.span(1)
    return chunk[:body_start] + body[:start] + now + body[end:], was, now


def inject_defects(corpus: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Introduce three realistic corpus defects; return it plus a description."""
    damaged = dict(corpus)
    notes: list[str] = []

    section_ids = sorted(damaged)
    if not section_ids:
        return damaged, notes

    # 1. A parser drops a section and says nothing. A corpus holding 4 of 5
    #    sections looks identical to one holding all 5 — until somebody asks
    #    about the missing one.
    dropped = section_ids[-1]
    del damaged[dropped]
    notes.append(f"deleted s.{dropped} from the corpus (simulating a silent parser drop)")

    # 2. The stored text drifts from the official text. Either the law was
    #    amended and nobody re-ingested, or the extraction was wrong all along.
    #    L1 cannot tell you which; it tells you that they disagree — which is
    #    the whole job, because either way the corpus is now wrong.
    for sid in section_ids:
        edited = _bump_a_cross_reference(damaged[sid])
        if edited is None:
            continue
        damaged[sid], was, now = edited
        notes.append(
            f"changed a cross-reference in s.{sid} from {was} to {now} (simulating amendment drift)"
        )
        break

    # 3. The corpus holds a section the source no longer shows — a repeal or
    #    renumbering that the index never caught up with.
    damaged["99"] = "Phantom section — repealed upstream, still indexed here."
    notes.append("added a phantom s.99 (simulating a repealed section left behind)")

    return damaged, notes


def report(findings: list[SectionFinding]) -> None:
    problems = [f for f in findings if f.verdict != "ok"]
    clean = {f.section_id for f in findings if f.verdict == "ok"}
    flagged = {f.section_id for f in problems}

    print(
        f"  sections: {len(clean | flagged)}   clean: {len(clean)}   "
        f"flagged: {len(flagged)}   findings: {len(problems)}"
    )
    for finding in problems:
        print(f"    [{finding.verdict}] s.{finding.section_id}: {finding.detail}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        help="verify any ontario.ca/laws/{statute,regulation}/... document "
        "instead of the built-in one",
    )
    parser.add_argument(
        "--title",
        help="document title for --url; it becomes the chunk label on sections "
        "that carry no headnote of their own (most regulations)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="read the captured fixture instead of fetching e-Laws live (built-in document only)",
    )
    args = parser.parse_args()

    if args.url:
        if args.offline:
            print(
                "--offline reads a captured fixture, and there is none for an "
                "arbitrary --url. Drop --offline to fetch it live.",
                file=sys.stderr,
            )
            return 1
        spec = spec_from_url(args.url, args.title)
        if spec is None:
            print(
                f"Not a recognized e-Laws document URL: {args.url}\n"
                "Expected a form like:\n"
                "  https://www.ontario.ca/laws/regulation/210242\n"
                "  https://www.ontario.ca/laws/statute/90o31",
                file=sys.stderr,
            )
            return 1
    else:
        spec = OREG

    print(f"Document: {spec.title}")
    print(f"Source:   {spec.url}\n")

    if args.offline:
        path = FIXTURE_DIR / (spec.fixture_filename or "")
        if not path.is_file():
            print(f"Fixture not found: {path}", file=sys.stderr)
            return 1
        html = path.read_text(encoding="utf-8")
        print("Fetched from the captured fixture (a snapshot, not authoritative).\n")
    else:
        print("Fetching the official text from Ontario's e-Laws API...")
        html = await fetch_api_content(spec.url) or ""
        if not html:
            print(
                "Fetch failed — e-Laws may be unreachable from here, or the "
                "document id may not exist.",
                file=sys.stderr,
            )
            return 2
        print("Fetched.\n")

    corpus = build_faithful_corpus(spec, html)
    if not corpus:
        print("Parser produced no sections — nothing to verify.", file=sys.stderr)
        return 2

    print("=" * 68)
    print("PASS 1 — a faithful corpus")
    print("=" * 68)
    report(verify_doc(spec, html, corpus))

    damaged, notes = inject_defects(corpus)
    print()
    print("=" * 68)
    print("PASS 2 — the same corpus, with three defects injected")
    print("=" * 68)
    for note in notes:
        print(f"  · {note}")
    print()
    report(verify_doc(spec, html, damaged))

    print(
        "\nNone of these defects change the shape of a retrieval result: the index\n"
        "still returns confident, well-formed passages. A grounding gate checking\n"
        "answers against them would certify every one. That is why the corpus is\n"
        "verified against its source, on a schedule, rather than assumed correct."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
