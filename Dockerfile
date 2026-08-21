# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- build stage
# Use BUILDPLATFORM so the build stage runs natively (faster cross-builds).
FROM --platform=$BUILDPLATFORM python:3.12-slim AS build

WORKDIR /src
RUN pip install --no-cache-dir build
COPY . .
RUN python -m build --wheel

# -------------------------------------------------------------- runtime stage
# No platform pin — the image builds for both linux/amd64 and linux/arm64
# via docker buildx. The runtime has no torch/CUDA, just Python + numpy.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="SemFuse" \
      org.opencontainers.image.description="Lightweight multilingual semantic retrieval with first-class Bangla, English, Banglish, and mixed-language support, plus optional RAG." \
      org.opencontainers.image.source="https://github.com/DIP-RO/Sem-fuse-rag-pip-package" \
      org.opencontainers.image.licenses="Apache-2.0"

# Install system deps for llama-cpp-python (builds from source).
# build-essential + cmake are small and available on both amd64 and arm64.
# git is needed by pip for some source installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the package with lightweight extras.
# No torch/CUDA — uses llama-cpp-python (~50 MB) for SLM, hashing for embeddings.
# pypdf + python-docx for file loaders.
COPY --from=build /src/dist/*.whl /tmp/dist/
RUN pip install --no-cache-dir /tmp/dist/*.whl pypdf python-docx && rm -rf /tmp/dist

# Force UTF-8 so Bangla/Unicode renders correctly on ALL platforms
# (Linux, Windows containers, etc.). Without this, Windows containers
# may use cp1252 and silently drop Bangla vowel signs.
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# One volume holds both the persistent index and the model cache.
# HuggingFace models are downloaded once and reused across containers.
ENV HF_HOME=/data/hf
WORKDIR /data
VOLUME /data

ENTRYPOINT ["semfuse"]
CMD ["--help"]
