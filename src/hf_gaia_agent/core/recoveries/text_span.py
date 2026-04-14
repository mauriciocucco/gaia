"""Core recovery for referenced text-span lookups."""

from __future__ import annotations

from ...fallbacks.text_span import TextSpanFallback


class TextSpanRecovery(TextSpanFallback):
    name = "text_span"
