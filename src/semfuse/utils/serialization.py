"""Serialization helpers for persisting typed records.

Records are stored as JSON. ``datetime`` fields are serialized as ISO-8601
strings and restored on load. Enums are stored by value.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


def _default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as pretty JSON to ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, default=_default, indent=2, ensure_ascii=False))


def dumps_json(obj: Any) -> str:
    """Serialize ``obj`` to a JSON string."""
    return json.dumps(obj, default=_default, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Read JSON from ``path``."""
    return json.loads(Path(path).read_text())


def parse_datetime(value: Any) -> datetime | None:
    """Parse an ISO-8601 datetime string, returning None on falsy input."""
    if not value:
        return None
    return datetime.fromisoformat(value)
