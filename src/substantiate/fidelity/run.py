"""Fidelity verification runner: fetch → verify → report → exit code.

Library-first: you supply the roster (a list of :class:`DocSpec`) and a
``CorpusReader`` (``doc_id -> {section_id: stored_chunk_text}``), and
compose your own console entry point::

    from substantiate.fidelity import DocSpec, run

    ROSTER = [DocSpec(doc_id=..., title=..., url=..., fixture_filename=...)]

    def read_corpus(doc_id: str) -> dict[str, str]:
        ...  # your store; see substantiate.adapters.postgres for a reference

    if __name__ == "__main__":
        raise SystemExit(run.main(roster=ROSTER, corpus_reader=read_corpus))

Exit codes: 0 = all sections ok; 1 = usage error; 2 = a fetch/parse
failure prevented verification for at least one doc; 5 = fidelity
violations found.  5 wins over 2 when both occur — a proven violation
is the stronger CI signal (fetch errors still appear in the summary).

``--from-fixture`` reads captured HTML from fixtures instead of fetching
live.  Fixtures are snapshots: an L1 mismatch in fixture mode may just
mean the fixture is stale.  Fixture mode is for plumbing tests and
offline runs, not authoritative verification.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from substantiate.fidelity.sources.elaws import fetch_api_content
from substantiate.fidelity.spec import DocSpec
from substantiate.fidelity.verify import (
    FIDELITY_VERDICTS,
    SectionFinding,
    verify_doc,
)

logger = logging.getLogger(__name__)

CorpusReader = Callable[[str], dict[str, str]]


async def _fetch_html(
    spec: DocSpec, *, from_fixture: bool, fixtures_dir: Path | None
) -> str | None:
    if from_fixture:
        if not spec.fixture_filename:
            logger.error("%s: no fixture_filename in roster", spec.doc_id)
            return None
        if fixtures_dir is None:
            logger.error("%s: --from-fixture requires --fixtures-dir", spec.doc_id)
            return None
        path = fixtures_dir / spec.fixture_filename
        if not path.is_file():
            logger.error("%s: missing fixture %s", spec.doc_id, path)
            return None
        return path.read_text(encoding="utf-8")
    return await fetch_api_content(spec.url)


async def verify_all(
    specs: list[DocSpec],
    corpus_reader: CorpusReader,
    *,
    from_fixture: bool = False,
    fixtures_dir: Path | None = None,
    max_diff_lines: int = 20,
) -> dict[str, list[SectionFinding]]:
    per_doc: dict[str, list[SectionFinding]] = {}
    for spec in specs:
        html = await _fetch_html(spec, from_fixture=from_fixture, fixtures_dir=fixtures_dir)
        if html is None:
            per_doc[spec.doc_id] = [
                SectionFinding(spec.doc_id, "*", "fetch_error", "could not fetch source HTML")
            ]
            continue
        corpus = corpus_reader(spec.doc_id)
        per_doc[spec.doc_id] = verify_doc(spec, html, corpus, max_diff_lines=max_diff_lines)
    return per_doc


def report(
    per_doc: dict[str, list[SectionFinding]],
    *,
    mode: str,
    json_out: Path | None,
) -> None:
    totals: dict[str, int] = {}
    for doc_id, findings in per_doc.items():
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.verdict] = counts.get(f.verdict, 0) + 1
            totals[f.verdict] = totals.get(f.verdict, 0) + 1
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{doc_id}  sections={len(findings)}  {summary}")
        for f in findings:
            if f.verdict == "ok":
                continue
            print(f"  [{f.verdict}] s.{f.section_id}: {f.detail}")
            if f.diff:
                for line in f.diff.splitlines():
                    print(f"    {line}")
    print("TOTALS  " + "  ".join(f"{k}={v}" for k, v in sorted(totals.items())))

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "generated_at": datetime.now(UTC).isoformat(),
                "mode": mode,
                "docs": len(per_doc),
            },
            "docs": {
                doc_id: {
                    "counts": {
                        v: sum(1 for f in findings if f.verdict == v)
                        for v in sorted({f.verdict for f in findings})
                    },
                    "sections": {
                        f.section_id: {
                            k: v
                            for k, v in asdict(f).items()
                            if k not in ("doc_id", "section_id") and v
                        }
                        for f in findings
                    },
                }
                for doc_id, findings in per_doc.items()
            },
        }
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("wrote %s", json_out)


def exit_code(per_doc: dict[str, list[SectionFinding]]) -> int:
    all_findings = [f for findings in per_doc.values() for f in findings]
    if any(f.verdict in FIDELITY_VERDICTS for f in all_findings):
        return 5
    if any(f.verdict == "fetch_error" for f in all_findings):
        return 2
    return 0


def main(
    argv: list[str] | None = None,
    *,
    roster: list[DocSpec],
    corpus_reader: CorpusReader,
    fixtures_dir: Path | None = None,
) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--doc-id",
        action="append",
        help="Restrict to one or more doc_ids (default: all in the roster)",
    )
    ap.add_argument(
        "--from-fixture",
        action="store_true",
        help="Read HTML from captured fixtures instead of the live source "
        "(plumbing/offline mode — fixtures are snapshots, not authoritative)",
    )
    ap.add_argument("--fixtures-dir", type=Path, default=fixtures_dir)
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the full per-section JSON report here",
    )
    ap.add_argument("--max-diff-lines", type=int, default=20)
    args = ap.parse_args(argv)

    target_ids = set(args.doc_id) if args.doc_id else {s.doc_id for s in roster}
    specs = [s for s in roster if s.doc_id in target_ids]
    unknown = target_ids - {s.doc_id for s in roster}
    if unknown:
        logger.error("unknown doc_id(s) in --doc-id: %s", sorted(unknown))
        return 1
    if not specs:
        logger.error("no doc_ids match --doc-id filter")
        return 1

    per_doc = asyncio.run(
        verify_all(
            specs,
            corpus_reader,
            from_fixture=args.from_fixture,
            fixtures_dir=args.fixtures_dir,
            max_diff_lines=args.max_diff_lines,
        )
    )
    report(per_doc, mode="fixture" if args.from_fixture else "live", json_out=args.json_out)
    return exit_code(per_doc)
