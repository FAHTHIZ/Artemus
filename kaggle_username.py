"""Normalize a Kaggle username secret without printing its raw contents.

Accepts the plain account handle, a quoted handle, a JSON kaggle.json object,
or a copied username field such as: `"username": "my-handle",`.
The normalized handle is written to a private local file for workflow use.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FIELD_RE = re.compile(
    r"""(?i)["']?username["']?\s*:\s*["']?([A-Za-z0-9][A-Za-z0-9._-]{0,63})["']?"""
)


def normalize_username(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("KAGGLE_USERNAME is empty")

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        candidate = parsed.get("username")
    elif isinstance(parsed, str):
        candidate = parsed
    else:
        match = _FIELD_RE.search(value)
        candidate = match.group(1) if match else value

    if not isinstance(candidate, str):
        raise ValueError("KAGGLE_USERNAME must contain a username field or handle")

    candidate = candidate.strip().strip("\\\"'").strip().rstrip(",").strip()
    if not _HANDLE_RE.fullmatch(candidate):
        raise ValueError("KAGGLE_USERNAME does not contain a valid Kaggle profile handle")
    return candidate


def main() -> None:
    raw = os.environ.get("KAGGLE_USERNAME", "")
    username = normalize_username(raw)
    Path(".kaggle-username").write_text(username + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
