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
