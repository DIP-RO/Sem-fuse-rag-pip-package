"""SemFuse CLI (thin wrapper over the library API).

Subcommands::

    semfuse info
    semfuse index path/to/file.pdf docs/ --text "inline document"
    semfuse search "Bangladesh er capital ki?" --top-k 3 --json
    semfuse ask "bhorti porikkha kokhon?"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from semfuse import SemFuse
from semfuse.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr so Bangla/Unicode renders correctly.

    On Windows the default encoding is often cp1252 or similar, which drops
    Bangla vowel signs (া, ে, ী) silently. We reconfigure stdout to UTF-8
    so all Unicode text prints correctly on every platform.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _make_db(args: argparse.Namespace) -> SemFuse:
    kwargs: dict[str, object] = {
        "storage_path": args.storage,
        "collection": args.collection,
    }
    if args.provider:
        kwargs["embedding_provider"] = args.provider
    if args.model:
        kwargs["embedding_model"] = args.model
    return SemFuse(**kwargs)  # type: ignore[arg-type]


def _cmd_info(args: argparse.Namespace) -> int:
    db = _make_db(args)
    print(json.dumps(db.info(), indent=2, ensure_ascii=False))
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    if not args.paths and not args.text:
        print("error: provide at least one path or --text", file=sys.stderr)
        return 2
    db = _make_db(args)
    added = 0
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            added += db.add_directory(p, persist=False)
        else:
            added += db.add_file(p, persist=False)
    for text in args.text or []:
        added += db.add(text, persist=False)
    db.persist()
    print(f"Indexed {added} chunks (collection={args.collection!r}, total={db.count()}).")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    db = _make_db(args)
    results = db.search(args.query, top_k=args.top_k, mode=args.mode)
    if args.json:
        payload = [
            {
                "score": r.score,
                "text": r.text,
                "document_id": r.document_id,
                "chunk_id": r.chunk_id,
                "language": r.language.value,
                "source": r.source,
                "page": r.page,
                "metadata": r.metadata,
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if not results:
        print("No results.")
        return 0
    for r in results:
        print(f"{r.score:.4f}  {r.text}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    db = _make_db(args)
    response = db.ask(args.question, top_k=args.top_k)
    print(response.answer)
    if response.citations:
        print("\nSources:")
        for i, citation in enumerate(response.citations, start=1):
            origin = citation.source or "unknown"
            if citation.page is not None:
                origin += f", page {citation.page}"
            preview = citation.text if len(citation.text) <= 80 else citation.text[:77] + "..."
            print(f"  [{i}] ({origin}) {preview}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semfuse", description="SemFuse CLI")
    parser.add_argument("--storage", default=".semfuse", help="Storage path")
    parser.add_argument("--collection", default="default", help="Collection name")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["local", "hashing"],
        help="Embedding provider (default: local)",
    )
    parser.add_argument("--model", default=None, help="Embedding model name")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show index information")
    p_info.set_defaults(func=_cmd_info)

    p_index = sub.add_parser("index", help="Index files, directories, or inline text")
    p_index.add_argument("paths", nargs="*", help="Files or directories to index")
    p_index.add_argument("--text", action="append", help="Inline text to index (repeatable)")
    p_index.set_defaults(func=_cmd_index)

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Query text (any supported language)")
    p_search.add_argument("--top-k", type=int, default=None, help="Number of results")
    p_search.add_argument(
        "--mode",
        default=None,
        choices=["auto", "semantic", "keyword", "hybrid"],
        help="Search mode (default: auto)",
    )
    p_search.add_argument("--json", action="store_true", help="Emit JSON results")
    p_search.set_defaults(func=_cmd_search)

    p_ask = sub.add_parser("ask", help="Answer a question from the index (RAG)")
    p_ask.add_argument("question", help="Question text")
    p_ask.add_argument("--top-k", type=int, default=None, help="Context passages to retrieve")
    p_ask.set_defaults(func=_cmd_ask)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI top-level boundary
        logger.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
