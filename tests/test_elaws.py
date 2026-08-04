"""e-Laws source adapter: parser structure + API-only fetcher.

The parser tests run against the synthetic act fixture (every paragraph
class exercised) and a captured real regulation. The fetcher tests
monkeypatch httpx — no network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from substantiate.fidelity.sources import elaws
from substantiate.fidelity.sources.elaws import (
    StatuteSection,
    elaws_api_url,
    fetch_api_content,
    parse_statute_html,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


class TestParserSyntheticAct:
    def test_full_structure(self):
        sections = parse_statute_html(_read("synthetic-act.html"))
        assert [s.section_id for s in sections] == ["1", "2", "3", "4"]

        s1, s2, s3, s4 = sections
        assert s1 == StatuteSection(
            section_id="1",
            heading="Definitions",
            topic_group="Definitions and Administration",
            text="1 (1) In this Act, “warranty” means a promise made by a vendor. "
            "(2) A promise includes an implied promise.",
        )
        # Headnote is consumed by exactly one section; topic persists.
        assert s2.heading == "Compensation"
        assert s2.topic_group == "Definitions and Administration"
        assert s2.text == "2. The Corporation shall pay compensation out of the guarantee fund."
        # A new heading1 switches the topic; no headnote -> None.
        assert (s3.heading, s3.topic_group) == (None, "Protections")
        assert s3.text.endswith("(2) The protection extends to a subsequent owner.")
        # toc / amendments / footnote paragraphs never leak into bodies.
        assert (s4.heading, s4.topic_group) == (None, "Protections")
        assert "transitional" not in " ".join(s.text for s in sections)
        assert "2020, c. 1" not in " ".join(s.text for s in sections)

    def test_real_regulation_parses(self):
        sections = parse_statute_html(_read("oreg-242-21.html"))
        assert len(sections) == 10
        assert all(s.heading is None or isinstance(s.heading, str) for s in sections)
        assert all(s.text for s in sections)


class TestApiUrlMapping:
    def test_statute_and_regulation_urls_map(self):
        assert (
            elaws_api_url("https://www.ontario.ca/laws/statute/90o31")
            == "https://www.ontario.ca/laws/api/v2/legislation/en/doc-search/statute/90o31"
        )
        assert (
            elaws_api_url("https://www.ontario.ca/laws/regulation/210242/")
            == "https://www.ontario.ca/laws/api/v2/legislation/en/doc-search/regulation/210242"
        )

    def test_non_elaws_url_is_none(self):
        assert elaws_api_url("https://example.com/not-elaws") is None


class TestFetchApiContent:
    """Unit tests with a monkeypatched httpx — no network."""

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def _fake_httpx(self, monkeypatch, payload=None, exc=None):
        import httpx

        outer = self

        class _FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, **kwargs):
                if exc is not None:
                    raise exc
                return outer._FakeResponse(payload)

        monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    def test_returns_content_on_usable_payload(self, monkeypatch):
        html = '<p class="headnote">Definitions</p><p class="section">1 (1) ...</p>'
        self._fake_httpx(monkeypatch, payload={"content": html})
        out = asyncio.run(fetch_api_content("https://www.ontario.ca/laws/statute/90o31"))
        assert out == html

    def test_returns_none_when_content_lacks_section_markup(self, monkeypatch):
        self._fake_httpx(monkeypatch, payload={"content": "<p>react shell</p>"})
        out = asyncio.run(fetch_api_content("https://www.ontario.ca/laws/statute/90o31"))
        assert out is None

    def test_headnote_less_regulation_content_is_usable(self, monkeypatch):
        """Regulations can have ZERO headnotes; the usability check keys on
        p.section markup, which every real document has."""
        html = '<p class="section">1 (1) The Corporation shall administer the Plan.</p>'
        self._fake_httpx(monkeypatch, payload={"content": html})
        out = asyncio.run(fetch_api_content("https://www.ontario.ca/laws/regulation/900892"))
        assert out == html

    def test_returns_none_on_http_error(self, monkeypatch):
        self._fake_httpx(monkeypatch, exc=RuntimeError("boom"))
        out = asyncio.run(fetch_api_content("https://www.ontario.ca/laws/statute/90o31"))
        assert out is None

    def test_returns_none_on_unmappable_url(self):
        assert asyncio.run(fetch_api_content("https://example.com/not-elaws")) is None

    def test_fetch_api_content_is_importable_from_elaws_module(self):
        assert elaws.fetch_api_content is fetch_api_content
