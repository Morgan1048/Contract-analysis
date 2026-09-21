"""Canonical text comparisons for extracted document content."""

from __future__ import annotations

import re


def canon(s: str) -> str:
    """Lowercase and remove intra-word whitespace for deterministic comparisons.

    Whitespace adjacent to a hyphen is retained so a broken ``"word- word"``
    remains distinct; parse-time normalization repairs that extraction defect.
    """
    lowered = s.lower()
    return "".join(
        char
        for index, char in enumerate(lowered)
        if not char.isspace()
        or (index > 0 and lowered[index - 1] == "-")
        or (index + 1 < len(lowered) and lowered[index + 1] == "-")
    )
