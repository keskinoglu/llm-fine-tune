FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Only polars + tqdm are needed at run time (verify-execution-engines import chain).
# uv sync would install the full lockfile closure (transformers, datasets, etc.) — overkill.
RUN uv pip install --system --no-cache polars tqdm

# Install the package itself (console scripts, internal modules) without re-pulling deps.
# Source is mounted at /workspace at run time; PYTHONPATH=/workspace/src makes Python
# prefer the mount over site-packages so edits are picked up without rebuilding.
COPY pyproject.toml README.md /build/
COPY src/ /build/src/
RUN uv pip install --system --no-cache --no-deps /build/

WORKDIR /workspace
