"""Legacy compatibility wrapper for :class:`TextSpanRecovery`."""

from __future__ import annotations

from ..core.recoveries.text_span import TextSpanRecovery


class TextSpanFallback(TextSpanRecovery):
    """Backward-compatible alias for the canonical recovery implementation."""

    name = "text_span"
