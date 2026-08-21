"""Phase 3: document loaders (TXT/MD/PDF/DOCX) and client file ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import SemFuse
from semfuse.core.enums import Language
from semfuse.core.exceptions import DocumentLoadError
from semfuse.loaders.factory import SUPPORTED_EXTENSIONS, get_loader, load_document

pypdf = pytest.importorskip("pypdf", reason="pypdf not installed (semfuse[pdf])")
docx_pkg = pytest.importorskip("docx", reason="python-docx not installed (semfuse[docx])")


def _write_pdf(path: Path, texts: list[str]) -> None:
    """Build a minimal text PDF with pypdf only (no external tools)."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DictionaryObject,
        NameObject,
        NumberObject,
        StreamObject,
    )

    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = StreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
        stream_ref = writer._add_object(stream)
        page[NameObject("/Contents")] = stream_ref
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
                NameObject("/ProcSet"): ArrayObject(
                    [NameObject("/PDF"), NameObject("/Text")]
                ),
            }
        )
        page[NameObject("/Rotate")] = NumberObject(0)
    with path.open("wb") as fh:
        writer.write(fh)


def test_supported_extensions() -> None:
    for ext in (".txt", ".md", ".pdf", ".docx"):
        assert ext in SUPPORTED_EXTENSIONS


def test_txt_loader(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("ঢাকা বাংলাদেশের রাজধানী।", encoding="utf-8")
    docs = load_document(f)
    assert len(docs) == 1
    assert docs[0].text.strip() == "ঢাকা বাংলাদেশের রাজধানী।"
    assert docs[0].language == Language.BN
    assert docs[0].title == "note"
    assert docs[0].source == str(f)


def test_md_loader(tmp_path: Path) -> None:
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nSome markdown content.", encoding="utf-8")
    docs = load_document(f)
    assert len(docs) == 1
    assert "markdown content" in docs[0].text


def test_txt_loader_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("   \n", encoding="utf-8")
    assert load_document(f) == []


def test_txt_loader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="not found"):
        load_document(tmp_path / "nope.txt")


def test_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "data.xyz"
    f.write_text("content")
    with pytest.raises(DocumentLoadError, match="No loader"):
        load_document(f)


def test_get_loader_dispatch() -> None:
    assert ".txt" in get_loader("a.txt").extensions
    assert ".pdf" in get_loader("a.PDF").extensions


def test_pdf_loader(tmp_path: Path) -> None:
    f = tmp_path / "doc.pdf"
    _write_pdf(f, ["First page content here.", "Second page content here."])
    docs = load_document(f)
    assert len(docs) == 2
    assert "First page" in docs[0].text
    assert docs[0].page == 1
    assert docs[1].page == 2
    assert docs[0].title == "doc"


def test_docx_loader(tmp_path: Path) -> None:
    f = tmp_path / "doc.docx"
    d = docx_pkg.Document()
    d.add_paragraph("First paragraph of the report.")
    d.add_paragraph("Second paragraph of the report.")
    d.save(str(f))
    docs = load_document(f)
    assert len(docs) == 1
    assert "First paragraph" in docs[0].text
    assert "Second paragraph" in docs[0].text


def test_add_file_and_metadata(db: SemFuse, tmp_path: Path) -> None:
    f = tmp_path / "notice.txt"
    f.write_text("CSE admission notice for the fall semester.", encoding="utf-8")
    added = db.add_file(f, metadata={"department": "CSE"})
    assert added == 1
    results = db.search("admission", filter={"department": "CSE"})
    assert results
    assert results[0].source == str(f)


def test_add_directory(db: SemFuse, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Document about mangoes.", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("Document about rivers.", encoding="utf-8")
    (tmp_path / "ignored.xyz").write_text("skipped")
    added = db.add_directory(tmp_path)
    assert added == 2
    assert db.search("mangoes")


def test_add_directory_non_recursive(db: SemFuse, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("top level", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("nested", encoding="utf-8")
    assert db.add_directory(tmp_path, recursive=False) == 1


def test_add_directory_not_a_directory(db: SemFuse, tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(DocumentLoadError, match="Not a directory"):
        db.add_directory(f)
