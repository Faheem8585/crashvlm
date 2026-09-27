# CrashVLM API (CPU, probe backend). Build context = repo root.
#   docker build -t crashvlm .
#   docker run -p 8000:8000 -v ~/crashvlm_data/serve:/data/serve crashvlm
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev --extra ml --extra serve
ENV CRASHVLM_DATA=/data CRASHVLM_BACKEND=probe PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "crashvlm.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
