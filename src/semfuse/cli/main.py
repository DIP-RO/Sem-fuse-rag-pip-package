"""Minimal CLI for SemFuse (thin wrapper over the library API).

Phase 1 provides ``info``. ``index``/``search`` arrive with later phases once
document loaders and the full retrieval pipeline are in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from semfuse import SemFuse
from semfuse.utils.logging import get_logger

logger = get_logger(__name__)


def _cmd_info(args: argparse.Namespace) -> int:
    db = SemFuse(storage_path=args.storage, collection=args.collection)
    print(json.dumps(db.info(), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semfuse", description="SemFuse CLI")
    parser.add_argument("--storage", default=".semfuse", help="Storage path")
    parser.add_argument("--collection", default="default", help="Collection name")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show index information")
    p_info.set_defaults(func=_cmd_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
