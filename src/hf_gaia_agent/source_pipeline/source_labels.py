"""Source-family labels for URLs.

These are descriptive labels for the kind of source behind a URL. They are
not executable adapters.
"""

from __future__ import annotations

from ._utils import registered_host


def source_family_for_url(url: str) -> str:
    """Return a descriptive source-family label for *url*."""
    host = registered_host(url)
    if host.endswith("wikipedia.org"):
        return "wikipedia_source"
    if host.endswith("universetoday.com"):
        return "article_source"
    if host.endswith("baseball-reference.com"):
        return "stats_table_source"
    if host.endswith("libretexts.org"):
        return "reference_text_source"
    return "generic_web_source"


# Backward-compatible aliases kept for older imports/callers.
source_label_for_url = source_family_for_url
select_adapter_name = source_family_for_url
