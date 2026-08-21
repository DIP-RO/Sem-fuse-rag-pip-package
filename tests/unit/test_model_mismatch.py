"""Index version / model mismatch tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import SemFuse
from semfuse.core.config import SemFuseConfig
from semfuse.core.exceptions import IndexVersionError


def _cfg(storage: Path, model: str, dim: int) -> SemFuseConfig:
    return SemFuseConfig(
        embedding_provider="hashing",
        embedding_model=model,
        embedding_dimension=dim,
        storage_path=storage,
    )


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    storage = tmp_path / "mismatch"
    db1 = SemFuse(config=_cfg(storage, "hashing-ngram", 256))
    db1.add("some document")
    db1.close()

    with pytest.raises(IndexVersionError) as exc_info:
        SemFuse(config=_cfg(storage, "hashing-ngram", 128))
    assert "incompatible" in str(exc_info.value).lower()
    assert "reindex" in str(exc_info.value).lower() or "delete" in str(exc_info.value).lower()


def test_model_name_mismatch_raises(tmp_path: Path) -> None:
    storage = tmp_path / "model_mismatch"
    db1 = SemFuse(config=_cfg(storage, "hashing-ngram", 256))
    db1.add("some document")
    db1.close()

    with pytest.raises(IndexVersionError):
        SemFuse(config=_cfg(storage, "hashing-other", 256))


def test_compatible_config_reopens(tmp_path: Path) -> None:
    storage = tmp_path / "compatible"
    db1 = SemFuse(config=_cfg(storage, "hashing-ngram", 256))
    db1.add("some document")
    db1.close()

    db2 = SemFuse(config=_cfg(storage, "hashing-ngram", 256))
    assert db2.count() == 1
