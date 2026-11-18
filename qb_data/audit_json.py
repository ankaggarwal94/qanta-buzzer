"""Decode audited inputs without silently discarding duplicate fields or non-finite numbers."""

import json
from math import isfinite
from typing import Any


def loads(text: str | bytes) -> Any:
    """Parse one JSON document, including a UTF-8 BOM when given bytes."""

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_number(literal: str) -> float:
        number = float(literal)
        if not isfinite(number):
            raise ValueError(f"Non-finite JSON number: {literal}")
        return number

    return json.loads(
        text,
        object_pairs_hook=unique_keys,
        parse_constant=finite_number,
        parse_float=finite_number,
    )
