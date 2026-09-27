# CrashVLM API, CPU image (probe backend). Build context = repo root.
#   docker build -t crashvlm .
#   docker run -p 8000:8000 -v ~/crashvlm_data/serve:/data/serve crashvlm
# The probes (scripts/export_probes.py) are mounted, not baked in: they are derived from the
# annotations and NHTSA footage, which the repo does not redistribute (docs/DATA_CARD.md).
# CPU-only PyTorch keeps the image small; the GPU/VLM backend is meant for Colab or a GPU host.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN uv venv /app/.venv \
    && uv pip install --python /app/.venv --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && uv pip install --python /app/.venv ".[serve]" "transformers>=4.56" sentencepiece protobuf
ENV CRASHVLM_DATA=/data CRASHVLM_BACKEND=probe PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/data/hf_cache
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "crashvlm.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
