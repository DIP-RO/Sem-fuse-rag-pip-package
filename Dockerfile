# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- build stage
FROM python:3.12-slim AS build

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

# CPU-only torch keeps the image several GB smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --from=build /src/dist/*.whl /tmp/dist/
RUN pip install --no-cache-dir /tmp/dist/*.whl pypdf python-docx && rm -rf /tmp/dist

# One volume holds both the persistent index (/data/.semfuse, the CLI default
# relative to the workdir) and the HuggingFace model cache, so the embedding
# model is downloaded once and reused across containers.
ENV HF_HOME=/data/hf
WORKDIR /data
VOLUME /data

ENTRYPOINT ["semfuse"]
CMD ["--help"]
