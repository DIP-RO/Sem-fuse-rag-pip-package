"""Document loader protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from semfuse.core.types import Document


@runtime_checkable
class DocumentLoader(Protocol):
    """Interface for turning a file into :class:`Document` objects."""

    extensions: tuple[str, ...]

    def load(self, path: str | Path) -> list[Document]:
        """Load ``path`` into one or more documents (e.g. one per PDF page)."""
        ...
