"""Plain-text and Markdown loader."""

from __future__ import annotations

from pathlib import Path

from semfuse.core.exceptions import DocumentLoadError
from semfuse.core.types import Document
from semfuse.language.detector import detect_language


class TextLoader:
    """Loads UTF-8 text files (.txt, .md) as a single document."""

    extensions: tuple[str, ...] = (".txt", ".md")

    def load(self, path: str | Path) -> list[Document]:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8-sig")
        except FileNotFoundError as exc:
            raise DocumentLoadError(f"File not found: {p}") from exc
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(
                f"Could not decode {p} as UTF-8. Convert the file to UTF-8 and retry."
            ) from exc
        if not text.strip():
            return []
        return [
            Document(
                text=text,
                source=str(p),
                title=p.stem,
                language=detect_language(text),
            )
        ]
