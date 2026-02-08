# python -m py_compile utils_json.py

import json
from typing import Any


def extract_json(text: str) -> tuple[dict[str, Any] | None, str, str | None]:
    if not text or not isinstance(text, str):
        return None, text or "", "empty or invalid input"

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n", 1)
        if len(lines) > 1:
            cleaned = lines[1]
        else:
            cleaned = ""
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    start_idx = _find_json_start(cleaned)
    if start_idx == -1:
        return None, text, "no JSON object found"

    end_idx = _find_matching_brace(cleaned, start_idx)
    if end_idx == -1:
        return None, text, "unbalanced braces"

    json_str = cleaned[start_idx : end_idx + 1]

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return parsed, text, None
        return None, text, "parsed JSON is not an object"
    except json.JSONDecodeError as e:
        return None, text, f"JSON decode error: {e}"


def _find_json_start(text: str) -> int:
    for i, ch in enumerate(text):
        if ch == "{":
            return i
    return -1


def _find_matching_brace(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{":
        return -1

    depth = 0
    in_string = False
    escape_next = False
    i = start

    while i < len(text):
        ch = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == "\\":
            if in_string:
                escape_next = True
            i += 1
            continue

        if ch == '"':
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1
