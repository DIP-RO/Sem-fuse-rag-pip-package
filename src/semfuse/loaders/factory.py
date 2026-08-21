"""Loader dispatch by file extension."""

from __future__ import annotations

from pathlib import Path

from semfuse.core.exceptions import DocumentLoadError
from semfuse.core.types import Document
from semfuse.loaders.base import DocumentLoader
from semfuse.loaders.docx import DocxLoader
from semfuse.loaders.pdf import PdfLoader
from semfuse.loaders.text import TextLoader

_LOADERS: tuple[DocumentLoader, ...] = (TextLoader(), PdfLoader(), DocxLoader())

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(
    ext for loader in _LOADERS for ext in loader.extensions
)


def get_loader(path: str | Path) -> DocumentLoader:
    """Return the loader responsible for ``path``'s extension."""
    suffix = Path(path).suffix.lower()
    for loader in _LOADERS:
        if suffix in loader.extensions:
            return loader
    raise DocumentLoadError(
        f"No loader for {suffix!r} files (path: {path}). "
        f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}."
    )


def load_document(path: str | Path) -> list[Document]:
    """Load ``path`` into documents using the extension-matched loader."""
    return get_loader(path).load(path)
