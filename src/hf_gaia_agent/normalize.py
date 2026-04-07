"""Answer extraction and cleanup utilities."""

from __future__ import annotations

import re

ANSWER_BLOCK_RE = re.compile(
    r"\[ANSWER\](?P<body>.*?)(?:\[/ANSWER\])?$",
    flags=re.IGNORECASE | re.DOTALL,
)
LEADING_LABEL_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|response|output)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
FENCE_RE = re.compile(r"^```[\w-]*\s*|\s*```$", flags=re.DOTALL)


def extract_answer_block(text: str) -> str:
    """Extract the answer body when a wrapper like [ANSWER]...[/ANSWER] is used."""
    candidate = (text or "").strip()
    match = ANSWER_BLOCK_RE.search(candidate)
    if match:
        return match.group("body").strip()
    return candidate


def _strip_outer_quotes(value: str) -> str:
    pairs = [('"', '"'), ("'", "'"), ("`", "`")]
    result = value.strip()
    changed = True
    while changed and len(result) >= 2:
        changed = False
        for left, right in pairs:
            if result.startswith(left) and result.endswith(right):
                result = result[1:-1].strip()
                changed = True
    return result


def normalize_submitted_answer(text: str) -> str:
    """Convert model output into the exact answer string to submit."""
    value = (text or "").strip()
    if not value:
        return ""

    value = FENCE_RE.sub("", value).strip()
    value = extract_answer_block(value)

    while True:
        updated = LEADING_LABEL_RE.sub("", value).strip()
        if updated == value:
            break
        value = updated

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) == 1:
        value = lines[0]
    elif lines:
        value = " ".join(lines)

    value = _strip_outer_quotes(value)
    value = re.sub(r"\s+,", ",", value)
    value = re.sub(r",\s*", ", ", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip()
