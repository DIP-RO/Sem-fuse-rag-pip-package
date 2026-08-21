"""semfuse.loaders subpackage."""

from __future__ import annotations

from semfuse.loaders.base import DocumentLoader
from semfuse.loaders.docx import DocxLoader
from semfuse.loaders.factory import SUPPORTED_EXTENSIONS, get_loader, load_document
from semfuse.loaders.pdf import PdfLoader
from semfuse.loaders.text import TextLoader

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DocumentLoader",
    "DocxLoader",
    "PdfLoader",
    "TextLoader",
    "get_loader",
    "load_document",
]
