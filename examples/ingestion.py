"""Document ingestion — files, directories, chunking, and dedup."""

from __future__ import annotations

import tempfile
from pathlib import Path

from semfuse import SemFuse


def main() -> None:
    workdir = Path(tempfile.mkdtemp())
    (workdir / "notes.txt").write_text(
        "Dhaka is the capital of Bangladesh.\n\n"
        "The university admission exam is held in December.",
        encoding="utf-8",
    )
    (workdir / "notice.md").write_text(
        "# Notice\n\nSchool is closed next week for the holidays.",
        encoding="utf-8",
    )

    db = SemFuse(storage_path=workdir / ".semfuse")

    # Index a single file (PDF/DOCX work the same via semfuse[pdf]/[docx]).
    db.add_file(workdir / "notes.txt", metadata={"kind": "notes"})

    # Or a whole directory tree of supported files.
    added = db.add_directory(workdir, extensions=(".md",))
    print(f"Indexed {added} chunk(s) from markdown files")

    # Long documents are chunked recursively (chunk_size/chunk_overlap in
    # config); duplicates are skipped by content hash.
    db.add_file(workdir / "notes.txt")  # no-op: already indexed
    print(f"Total chunks: {db.count()}")

    for r in db.search("When is the admission exam?", top_k=2):
        print(f"  {r.score:.4f}  [{r.source}] {r.text[:60]}")


if __name__ == "__main__":
    main()
