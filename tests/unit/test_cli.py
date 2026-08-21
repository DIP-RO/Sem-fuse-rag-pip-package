"""Phase 8: CLI subcommands (info/index/search/ask) with the offline provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semfuse.cli.main import main


def _base(tmp_path: Path) -> list[str]:
    return [
        "--storage",
        str(tmp_path / "store"),
        "--provider",
        "hashing",
        "--model",
        "hashing-ngram",
    ]


def test_cli_info_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*_base(tmp_path), "info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunk_count"] == 0
    assert payload["embedding_provider"] == "hashing"


def test_cli_index_and_search(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("Dhaka is the capital of Bangladesh.", encoding="utf-8")
    assert main([*_base(tmp_path), "index", str(doc), "--text", "The Eiffel Tower is in Paris."]) == 0
    out = capsys.readouterr().out
    assert "Indexed 2 chunks" in out

    assert main([*_base(tmp_path), "search", "capital of Bangladesh", "--top-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "Dhaka" in out


def test_cli_index_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Document about mangoes.", encoding="utf-8")
    (corpus / "b.md").write_text("Document about rivers.", encoding="utf-8")
    assert main([*_base(tmp_path), "index", str(corpus)]) == 0
    assert "Indexed 2 chunks" in capsys.readouterr().out


def test_cli_search_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*_base(tmp_path), "index", "--text", "Dhaka is the capital."]) == 0
    capsys.readouterr()
    assert main([*_base(tmp_path), "search", "capital", "--json", "--mode", "keyword"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert payload[0]["text"] == "Dhaka is the capital."
    assert set(payload[0]) >= {"score", "text", "document_id", "chunk_id", "language"}


def test_cli_search_no_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*_base(tmp_path), "index", "--text", "something"]) == 0
    capsys.readouterr()
    assert main([*_base(tmp_path), "search", "zzz", "--mode", "keyword"]) == 0
    assert "No results" in capsys.readouterr().out


def test_cli_ask(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*_base(tmp_path), "index", "--text", "ঢাকা বাংলাদেশের রাজধানী।"]) == 0
    capsys.readouterr()
    assert main([*_base(tmp_path), "ask", "Bangladesh er rajdhani kothay?"]) == 0
    out = capsys.readouterr().out
    assert "ঢাকা" in out
    assert "Sources:" in out


def test_cli_index_requires_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*_base(tmp_path), "index"]) == 2
    assert "provide at least one path" in capsys.readouterr().err


def test_cli_error_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*_base(tmp_path), "index", str(tmp_path / "missing.txt")]) == 1
    assert "error:" in capsys.readouterr().err
