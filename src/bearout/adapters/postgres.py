"""Reference CorpusReader for a Postgres-backed corpus (e.g. pgvector).

Requires the ``pgvector`` extra: ``pip install bearout[pgvector]``.

The reader is read-only and query-parameterized: you supply the SQL that
maps a ``doc_id`` to ``(section_id, chunk_text)`` rows for your schema.
Example for a table with a JSONB ``metadata`` column::

    reader = postgres_corpus_reader(
        "host=localhost dbname=mydb user=postgres",
        "SELECT metadata->>'section_id', text FROM public.corpus_chunks "
        "WHERE metadata->>'source' = %s",
    )
"""

from __future__ import annotations

from collections.abc import Callable


def postgres_corpus_reader(conninfo: str, sql: str) -> Callable[[str], dict[str, str]]:
    """Build a ``CorpusReader``: ``doc_id -> {section_id: stored_text}``.

    ``sql`` must take exactly one ``%s`` parameter (the doc_id) and
    return ``(section_id, text)`` rows.  Rows with a NULL/empty
    section_id are skipped.
    """
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The Postgres corpus reader needs psycopg: pip install bearout[pgvector]"
        ) from exc

    def read(doc_id: str) -> dict[str, str]:
        out: dict[str, str] = {}
        with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
            cur.execute(sql, (doc_id,))
            for section_id, text in cur.fetchall():
                if section_id:
                    out[section_id] = text
        return out

    return read


__all__ = ["postgres_corpus_reader"]
