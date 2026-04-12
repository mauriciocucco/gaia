"""Tools package — re-exports all public symbols for backward compatibility.

External code can still do ``from hf_gaia_agent.tools import web_search``
or ``import hf_gaia_agent.tools as tools; tools.web_search``.
"""

from __future__ import annotations

from typing import Any

# -- HTTP helpers (kept public for monkeypatching in tests) --
import httpx  # noqa: F401  — tests monkeypatch ``tools.httpx``
import subprocess  # noqa: F401  — tests monkeypatch ``tools.subprocess``

from ._http import (
    HTTP_HEADERS as _HTTP_HEADERS,  # noqa: F401
    SEARCH_HEADERS as _SEARCH_HEADERS,  # noqa: F401
    truncate as _truncate,  # noqa: F401
)

# -- Parsing helpers --
from ._parsing import (
    html_to_text as _html_to_text,  # noqa: F401
    read_xlsx as _read_xlsx,  # noqa: F401
)

# -- Web helpers --
from ._web_helpers import extract_youtube_video_id  # noqa: F401

# -- Search --
from .search import (
    _search_brave_html,  # noqa: F401
    _search_duckduckgo,  # noqa: F401
    _search_bing_rss,  # noqa: F401
    _merge_search_results,  # noqa: F401
    _normalize_search_result,  # noqa: F401
    _format_search_results,  # noqa: F401
    web_search_result,  # noqa: F401
    search_wikipedia_result,  # noqa: F401
    web_search,  # noqa: F401
    search_wikipedia,  # noqa: F401
)

# -- Web / fetch --
from .web import (
    _fetch_url_result,  # noqa: F401
    _fetch_url_text,  # noqa: F401
    _fetch_html_text,  # noqa: F401
    fetch_url,  # noqa: F401
    fetch_wikipedia_page_result,  # noqa: F401
    fetch_wikipedia_page,  # noqa: F401
    extract_links_from_url_result,  # noqa: F401
    extract_links_from_url,  # noqa: F401
    find_text_in_url_result,  # noqa: F401
    find_text_in_url,  # noqa: F401
    extract_tables_from_url_result,  # noqa: F401
    extract_tables_from_url,  # noqa: F401
    count_wikipedia_studio_albums,  # noqa: F401
    count_wikipedia_studio_album_count_for_artist,  # noqa: F401
)

# -- Document --
from .document import (
    read_file_content_result,  # noqa: F401
    read_file_content,  # noqa: F401
    read_local_file,  # noqa: F401
    _transcribe_audio,  # noqa: F401
)

# -- Media --
from .media import (
    _check_binary,  # noqa: F401
    _download_video,  # noqa: F401
    _extract_frames,  # noqa: F401
    _extract_dense_frames,  # noqa: F401
    _encode_frame_base64,  # noqa: F401
    _is_counting_visual_question,  # noqa: F401
    _extract_json_object,  # noqa: F401
    _extract_max_count_from_payload,  # noqa: F401
    get_youtube_transcript_result,  # noqa: F401
    get_youtube_transcript,  # noqa: F401
    analyze_youtube_video,  # noqa: F401
)

# -- Compute --
from .compute import (
    calculate,  # noqa: F401
    execute_python_code,  # noqa: F401
)


def build_tools() -> list[Any]:
    return [
        web_search,
        search_wikipedia,
        fetch_url,
        fetch_wikipedia_page,
        extract_links_from_url,
        find_text_in_url,
        extract_tables_from_url,
        get_youtube_transcript,
        analyze_youtube_video,
        count_wikipedia_studio_albums,
        read_local_file,
        calculate,
        execute_python_code,
    ]


STRUCTURED_TOOL_INVOKERS: dict[str, Any] = {
    "web_search": web_search_result,
    "search_wikipedia": search_wikipedia_result,
    "fetch_url": _fetch_url_result,
    "fetch_wikipedia_page": fetch_wikipedia_page_result,
    "extract_links_from_url": extract_links_from_url_result,
    "find_text_in_url": find_text_in_url_result,
    "extract_tables_from_url": extract_tables_from_url_result,
    "get_youtube_transcript": get_youtube_transcript_result,
    "read_local_file": read_file_content_result,
}
