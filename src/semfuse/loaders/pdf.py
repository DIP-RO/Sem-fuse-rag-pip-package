"""PDF loader backed by ``pypdf`` (optional extra: ``semfuse[pdf]``)."""

from __future__ import annotations

from pathlib import Path

from semfuse.core.exceptions import DocumentLoadError
from semfuse.core.types import Document
from semfuse.language.detector import detect_language


class PdfLoader:
    """Loads a PDF as one document per non-empty page (page numbers 1-based)."""

    extensions: tuple[str, ...] = (".pdf",)

    def load(self, path: str | Path) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentLoadError(
                "pypdf is required to load PDF files. "
                "Install it with `pip install semfuse[pdf]`."
            ) from exc
        p = Path(path)
        if not p.exists():
            raise DocumentLoadError(f"File not found: {p}")
        try:
            reader = PdfReader(str(p))
        except Exception as exc:
            raise DocumentLoadError(f"Failed to parse PDF {p}: {exc}") from exc
        docs: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    source=str(p),
                    title=p.stem,
                    page=page_number,
                    language=detect_language(text),
                )
            )
        return docs
