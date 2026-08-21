# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- build stage
FROM --platform=$BUILDPLATFORM python:3.12-slim AS build

WORKDIR /src
RUN pip install --no-cache-dir build
COPY . .
RUN python -m build --wheel

# -------------------------------------------------------------- runtime stage
FROM python:3.12-slim

LABEL org.opencontainers.image.title="SemFuse" \
      org.opencontainers.image.description="Lightweight multilingual semantic retrieval with first-class Bangla, English, Banglish, and mixed-language support, plus optional RAG." \
      org.opencontainers.image.source="https://github.com/DIP-RO/Sem-fuse-rag-pip-package" \
      org.opencontainers.image.licenses="Apache-2.0"

# Install system deps for llama-cpp-python (builds from source on first install).
# These are small and work on both amd64 and arm64.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install the package with lightweight extras.
# No torch/CUDA — uses llama-cpp-python (~50 MB) for SLM, hashing for embeddings.
COPY --from=build /src/dist/*.whl /tmp/dist/
RUN pip install --no-cache-dir /tmp/dist/*.whl pypdf python-docx && rm -rf /tmp/dist

# One volume holds both the persistent index and the model cache.
# HuggingFace models are downloaded once and reused across containers.
ENV HF_HOME=/data/hf
WORKDIR /data
VOLUME /data

ENTRYPOINT ["semfuse"]
CMD ["--help"]
