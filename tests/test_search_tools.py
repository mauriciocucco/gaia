from __future__ import annotations

import pytest

import hf_gaia_agent.tools as tools


def test_search_brave_html_parses_results(monkeypatch) -> None:
    html = """
    <html>
      <head><title>Example query - Brave Search</title></head>
      <body>
        <div class="result-content">
          <a href="https://example.com/article">
            <div class="search-snippet-title">Example Result</div>
          </a>
          <div class="generic-snippet">
            <div class="content">This is a useful snippet.</div>
          </div>
        </div>
      </body>
    </html>
    """

    class FakeResponse:
        text = html

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, _url: str, params: dict[str, str]) -> FakeResponse:
            assert params == {"q": "example query", "source": "web"}
            return FakeResponse()

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    results = tools._search_brave_html("example query", max_results=5)

    assert results == [
        {
            "title": "Example Result",
            "href": "https://example.com/article",
            "body": "This is a useful snippet.",
        }
    ]


def test_merge_search_results_deduplicates_by_url() -> None:
    merged = tools._merge_search_results(
        [
            {"title": "One", "href": "https://example.com/1", "body": "A"},
        ],
        [
            {"title": "Duplicate", "href": "https://example.com/1", "body": "B"},
            {"title": "Two", "href": "https://example.com/2", "body": "C"},
        ],
        max_results=5,
    )

    assert merged == [
        {"title": "One", "href": "https://example.com/1", "body": "A"},
        {"title": "Two", "href": "https://example.com/2", "body": "C"},
    ]


def test_web_search_falls_back_when_primary_provider_fails(monkeypatch) -> None:
    def fail_brave(query: str, *, max_results: int) -> list[dict[str, str]]:
        assert query == "fallback query"
        assert max_results == 3
        raise RuntimeError("captcha")

    def ok_ddg(query: str, *, max_results: int) -> list[dict[str, str]]:
        assert query == "fallback query"
        assert max_results == 3
        return [
            {
                "title": "Recovered Result",
                "href": "https://example.com/recovered",
                "body": "Recovered snippet",
            }
        ]

    def unused_bing(query: str, *, max_results: int) -> list[dict[str, str]]:
        raise AssertionError("bing should not be called once enough results were found")

    monkeypatch.setattr(tools, "_search_brave_html", fail_brave)
    monkeypatch.setattr(tools, "_search_duckduckgo", ok_ddg)
    monkeypatch.setattr(tools, "_search_bing_rss", unused_bing)

    result = tools.web_search.invoke({"query": "fallback query", "max_results": 3})

    assert "Recovered Result" in result
    assert "https://example.com/recovered" in result


def test_web_search_raises_when_all_providers_fail(monkeypatch) -> None:
    def fail(name: str):
        def _inner(query: str, *, max_results: int) -> list[dict[str, str]]:
            assert query == "broken query"
            assert max_results == 2
            raise RuntimeError(name)

        return _inner

    monkeypatch.setattr(tools, "_search_brave_html", fail("brave failed"))
    monkeypatch.setattr(tools, "_search_duckduckgo", fail("ddg failed"))
    monkeypatch.setattr(tools, "_search_bing_rss", fail("bing failed"))

    with pytest.raises(RuntimeError, match="All search providers failed"):
        tools.web_search.invoke({"query": "broken query", "max_results": 2})


def test_search_wikipedia_formats_results(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "query": {
                    "search": [
                        {
                            "title": "Mercedes Sosa",
                            "snippet": 'Argentine <span class="searchmatch">singer</span>',
                        }
                    ]
                }
            }

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str, params: dict[str, str]) -> FakeResponse:
            assert url == "https://en.wikipedia.org/w/api.php"
            assert params["srsearch"] == "Mercedes Sosa"
            return FakeResponse()

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.search_wikipedia.invoke({"query": "Mercedes Sosa", "max_results": 3})

    assert "Mercedes Sosa" in result
    assert "Argentine singer" in result
    assert "https://en.wikipedia.org/wiki/Mercedes_Sosa" in result


def test_fetch_url_falls_back_to_r_jina_when_direct_fetch_fails(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, *, text: str, url: str, status_error: Exception | None = None) -> None:
            self.text = text
            self.url = url
            self.headers = {"content-type": "text/html; charset=utf-8"}
            self._status_error = status_error

        def raise_for_status(self) -> None:
            if self._status_error is not None:
                raise self._status_error

    class FakeStatusError(Exception):
        pass

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str, params: dict[str, str] | None = None) -> FakeResponse:
            assert params is None
            self.calls.append(url)
            if len(self.calls) == 1:
                return FakeResponse(
                    text="forbidden",
                    url=url,
                    status_error=FakeStatusError("403"),
                )
            return FakeResponse(
                text="<html><head><title>Mirror</title></head><body><p>Recovered content</p></body></html>",
                url=url,
            )

    fake_client = FakeClient()
    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: fake_client)

    result = tools.fetch_url.invoke({"url": "https://example.com/article"})

    assert fake_client.calls == [
        "https://example.com/article",
        "https://r.jina.ai/https://example.com/article",
    ]
    assert "Recovered content" in result


def test_count_wikipedia_studio_albums_falls_back_to_rendered_tables(monkeypatch) -> None:
    class FakeStatusError(tools.httpx.HTTPError):
        pass

    class FakeResponse:
        def __init__(self, url: str) -> None:
            self.url = url

        def raise_for_status(self) -> None:
            raise FakeStatusError("403")

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, _url: str, *_args: object, **_kwargs: object) -> FakeResponse:
            return FakeResponse(_url)

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(
        tools,
        "extract_tables_from_url",
        lambda url, text_filter="", max_tables=5, max_rows_per_table=15: (
            "Table 1\n"
            "Caption: Studio albums\n"
            "Year | Album details\n"
            "2000 | Misa Criolla\n"
            "2003 | Acustico\n"
            "2009 | Cantora 1\n"
        ),
    )

    result = tools.count_wikipedia_studio_albums.invoke(
        {"artist_name": "Mercedes Sosa", "start_year": 2000, "end_year": 2009}
    )

    assert result == "3"


def test_fetch_wikipedia_page_returns_extract(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "query": {
                    "pages": [
                        {
                            "title": "Mercedes Sosa",
                            "fullurl": "https://en.wikipedia.org/wiki/Mercedes_Sosa",
                            "extract": "Mercedes Sosa was an Argentine singer.",
                        }
                    ]
                }
            }

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str, params: dict[str, str]) -> FakeResponse:
            assert url == "https://en.wikipedia.org/w/api.php"
            assert params["titles"] == "Mercedes Sosa"
            return FakeResponse()

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.fetch_wikipedia_page.invoke({"title": "Mercedes Sosa"})

    assert "Title: Mercedes Sosa" in result
    assert "Mercedes Sosa was an Argentine singer." in result


def test_extract_links_from_url_filters_and_resolves_relative_urls(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <a href="/paper.pdf">Observation Paper</a>
        <a href="https://example.com/about">About</a>
      </body>
    </html>
    """

    class FakeResponse:
        text = html
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://example.com/article"

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            assert url == "https://example.com/article"
            return FakeResponse()

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.extract_links_from_url.invoke(
        {
            "url": "https://example.com/article",
            "text_filter": "paper",
            "same_domain_only": True,
            "max_results": 10,
        }
    )

    assert "Observation Paper" in result
    assert "https://example.com/paper.pdf" in result
    assert "About" not in result


def test_find_text_in_url_returns_matching_lines(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_fetch_url_text",
        lambda url: "Title: Example\nalpha line\nbeta keyword match\nfinal line",
    )

    result = tools.find_text_in_url.invoke(
        {"url": "https://example.com", "query": "keyword", "max_matches": 3}
    )

    assert result == "beta keyword match"


def test_find_text_in_url_uses_token_overlap_when_exact_phrase_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_fetch_url_text",
        lambda url: (
            "Participating nations\n"
            "Number of athletes by National Olympic Committees\n"
            "Cuba | 1\n"
            "Panama | 1\n"
        ),
    )

    result = tools.find_text_in_url.invoke(
        {
            "url": "https://example.com/olympics",
            "query": "number of athletes by country",
            "max_matches": 2,
        }
    )

    assert "Number of athletes by National Olympic Committees" in result


def test_extract_tables_from_url_renders_table_rows(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <table>
          <caption>Roster</caption>
          <tr><th>No.</th><th>Name</th></tr>
          <tr><td>18</td><td>Yoshida</td></tr>
          <tr><td>19</td><td>Tamai</td></tr>
        </table>
      </body>
    </html>
    """

    class FakeResponse:
        text = html
        headers = {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            assert url == "https://example.com/roster"
            return FakeResponse()

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.extract_tables_from_url.invoke({"url": "https://example.com/roster"})

    assert "Table 1" in result
    assert "Caption: Roster" in result
    assert "18 | Yoshida" in result


def test_extract_tables_from_url_reads_tables_inside_html_comments(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <!--
        <table>
          <caption>Batting</caption>
          <tr><th>Player</th><th>BB</th><th>AB</th></tr>
          <tr><td>Player A</td><td>100</td><td>519</td></tr>
        </table>
        -->
      </body>
    </html>
    """

    class FakeResponse:
        text = html
        headers = {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            assert url == "https://example.com/hidden"
            return FakeResponse()

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.extract_tables_from_url.invoke({"url": "https://example.com/hidden"})

    assert "Caption: Batting" in result
    assert "Player A | 100 | 519" in result


def test_extract_tables_from_url_filters_to_relevant_table(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <table>
          <caption>Pitchers</caption>
          <tr><th>No.</th><th>Name</th></tr>
          <tr><td>19</td><td>Tamai</td></tr>
        </table>
        <table>
          <caption>Batting</caption>
          <tr><th>Player</th><th>AB</th></tr>
          <tr><td>Munson</td><td>519</td></tr>
        </table>
      </body>
    </html>
    """

    class FakeResponse:
        text = html
        headers = {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            assert url == "https://example.com/stats"
            return FakeResponse()

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.extract_tables_from_url.invoke(
        {
            "url": "https://example.com/stats",
            "text_filter": "batting at bats",
            "max_tables": 1,
        }
    )

    assert "Caption: Batting" in result
    assert "Munson | 519" in result
    assert "Caption: Pitchers" not in result


def test_extract_tables_from_url_falls_back_to_wikipedia_parse_api(monkeypatch) -> None:
    html = """
    <div class="mw-parser-output">
      <table>
        <caption>Participating National Olympic Committees</caption>
        <tr><th>Country</th><th>Athletes</th></tr>
        <tr><td>Cuba</td><td>1</td></tr>
        <tr><td>Panama</td><td>1</td></tr>
      </table>
    </div>
    """

    class FakeHTTPStatusError(Exception):
        pass

    class FakeResponse:
        def __init__(
            self,
            *,
            text: str = "",
            url: str = "",
            json_payload: dict[str, object] | None = None,
            status_error: Exception | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.text = text
            self.url = url
            self._json_payload = json_payload or {}
            self._status_error = status_error
            self.headers = headers or {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            if self._status_error is not None:
                raise self._status_error

        def json(self) -> dict[str, object]:
            return self._json_payload

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str, params: dict[str, str] | None = None) -> FakeResponse:
            if url == "https://en.wikipedia.org/wiki/1928_Summer_Olympics":
                assert params is None
                return FakeResponse(
                    url=url,
                    status_error=FakeHTTPStatusError("403"),
                )
            assert url == "https://en.wikipedia.org/w/api.php"
            assert params is not None
            assert params["action"] == "parse"
            assert params["page"] == "1928 Summer Olympics"
            return FakeResponse(
                url=url,
                json_payload={
                    "parse": {
                        "title": "1928 Summer Olympics",
                        "text": html,
                    }
                },
                headers={"content-type": "application/json"},
            )

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.extract_tables_from_url.invoke(
        {"url": "https://en.wikipedia.org/wiki/1928_Summer_Olympics"}
    )

    assert "Caption: Participating National Olympic Committees" in result
    assert "Cuba | 1" in result


def test_extract_links_from_url_falls_back_to_wikipedia_parse_api(monkeypatch) -> None:
    html = """
    <div class="mw-parser-output">
      <a href="/wiki/Paper">Observation paper</a>
      <a href="/wiki/Other">Other link</a>
    </div>
    """

    class FakeHTTPStatusError(Exception):
        pass

    class FakeResponse:
        def __init__(
            self,
            *,
            text: str = "",
            url: str = "",
            json_payload: dict[str, object] | None = None,
            status_error: Exception | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.text = text
            self.url = url
            self._json_payload = json_payload or {}
            self._status_error = status_error
            self.headers = headers or {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            if self._status_error is not None:
                raise self._status_error

        def json(self) -> dict[str, object]:
            return self._json_payload

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str, params: dict[str, str] | None = None) -> FakeResponse:
            if url == "https://en.wikipedia.org/wiki/1928_Summer_Olympics":
                assert params is None
                return FakeResponse(
                    url=url,
                    status_error=FakeHTTPStatusError("403"),
                )
            assert url == "https://en.wikipedia.org/w/api.php"
            assert params is not None
            return FakeResponse(
                url=url,
                json_payload={
                    "parse": {
                        "title": "1928 Summer Olympics",
                        "text": html,
                    }
                },
                headers={"content-type": "application/json"},
            )

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.extract_links_from_url.invoke(
        {
            "url": "https://en.wikipedia.org/wiki/1928_Summer_Olympics",
            "text_filter": "paper",
            "same_domain_only": True,
        }
    )

    assert "Observation paper" in result
    assert "https://en.wikipedia.org/wiki/Paper" in result


def test_extract_tables_from_url_falls_back_to_r_jina_markdown_table(monkeypatch) -> None:
    markdown = """
### Number of athletes by National Olympic Committees

| Country | Athletes |
| --- | --- |
| [Cuba](https://en.wikipedia.org/wiki/Cuba_at_the_1928_Summer_Olympics) | 1 |
| [Panama](https://en.wikipedia.org/wiki/Panama_at_the_1928_Summer_Olympics) | 1 |
| [Argentina](https://en.wikipedia.org/wiki/Argentina_at_the_1928_Summer_Olympics) | 81 |
"""

    class FakeHTTPStatusError(Exception):
        pass

    class FakeResponse:
        def __init__(
            self,
            *,
            text: str = "",
            url: str = "",
            json_payload: dict[str, object] | None = None,
            status_error: Exception | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.text = text
            self.url = url
            self._json_payload = json_payload or {}
            self._status_error = status_error
            self.headers = headers or {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            if self._status_error is not None:
                raise self._status_error

        def json(self) -> dict[str, object]:
            return self._json_payload

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str, params: dict[str, str] | None = None) -> FakeResponse:
            if url == "https://en.wikipedia.org/wiki/1928_Summer_Olympics":
                return FakeResponse(url=url, status_error=FakeHTTPStatusError("403"))
            if url == "https://en.wikipedia.org/w/api.php":
                return FakeResponse(url=url, status_error=FakeHTTPStatusError("403"))
            assert url == "https://r.jina.ai/https://en.wikipedia.org/wiki/1928_Summer_Olympics"
            assert params is None
            return FakeResponse(
                text=markdown,
                url=url,
                headers={"content-type": "text/plain; charset=utf-8"},
            )

    monkeypatch.setattr(tools.httpx, "Client", lambda **_kwargs: FakeClient())

    result = tools.extract_tables_from_url.invoke(
        {"url": "https://en.wikipedia.org/wiki/1928_Summer_Olympics"}
    )

    assert "Caption: Number of athletes by National Olympic Committees" in result
    assert "Country | Athletes" in result
    assert "Cuba | 1" in result
