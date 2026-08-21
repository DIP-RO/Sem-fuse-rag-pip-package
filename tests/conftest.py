"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import SemFuse
from semfuse.core.config import SemFuseConfig


def _hashing_config(storage: Path, collection: str = "default") -> SemFuseConfig:
    return SemFuseConfig(
        embedding_provider="hashing",
        embedding_model="hashing-ngram",
        embedding_dimension=256,
        storage_path=storage,
        collection=collection,
        lazy=True,
    )


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    return tmp_path / "semfuse_store"


@pytest.fixture
def db(tmp_storage: Path) -> SemFuse:
    """A fresh SemFuse using the deterministic hashing provider (offline)."""
    return SemFuse(config=_hashing_config(tmp_storage))
