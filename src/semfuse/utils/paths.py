"""Path validation utilities."""

from __future__ import annotations

from pathlib import Path

from semfuse.core.exceptions import ConfigurationError


def safe_resolve(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path, ensuring it does not escape ``base`` when provided."""
    p = Path(path).expanduser()
    if base is not None:
        base_p = Path(base).expanduser().resolve()
        resolved = (base_p / p).resolve() if not p.is_absolute() else p.resolve()
        try:
            resolved.relative_to(base_p)
        except ValueError as exc:
            raise ConfigurationError(
                f"Path {path!r} escapes the allowed base directory {base!r}."
            ) from exc
        return resolved
    return p.resolve()
