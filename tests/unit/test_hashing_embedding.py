"""Hashing embedding provider tests (deterministic, offline)."""

from __future__ import annotations

import numpy as np

from semfuse.embeddings.hashing import HashingEmbeddingProvider


def test_dimension_and_name() -> None:
    emb = HashingEmbeddingProvider(dimension=128, model_name="test-hash")
    assert emb.dimension == 128
    assert emb.model_name == "test-hash"


def test_deterministic() -> None:
    emb = HashingEmbeddingProvider(dimension=128)
    a = emb.embed_query("Dhaka is the capital")
    b = emb.embed_query("Dhaka is the capital")
    assert np.allclose(a, b)


def test_shape_documents() -> None:
    emb = HashingEmbeddingProvider(dimension=64)
    vecs = emb.embed_documents(["one", "two", "three"])
    assert vecs.shape == (3, 64)


def test_empty_documents() -> None:
    emb = HashingEmbeddingProvider(dimension=64)
    vecs = emb.embed_documents([])
    assert vecs.shape == (0, 64)


def test_shared_substrings_score_higher() -> None:
    emb = HashingEmbeddingProvider(dimension=512)
    q = emb.embed_query("capital of Bangladesh")
    relevant = emb.embed_query("Dhaka is the capital of Bangladesh.")
    irrelevant = emb.embed_query("The Eiffel Tower is in Paris, France.")
    # Cosine similarity (vectors are L2-normalized).
    sim_rel = float(q @ relevant)
    sim_irr = float(q @ irrelevant)
    assert sim_rel > sim_irr
