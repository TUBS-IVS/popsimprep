FROM mcr.microsoft.com/devcontainers/python:3.11

# System deps for geo/data stacks (GDAL/PostGIS client libs) + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libpq-dev \
    postgresql-client \
    curl ca-certificates \
    build-essential \
    gfortran \
    libopenblas0-pthread \
    && rm -rf /var/lib/apt/lists/*

# Install uv (system-wide so postCreateCommand can run it as vscode)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv

# Install just (command runner)
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
  | bash -s -- --to /usr/local/bin

# OPTIONAL: Keep Claude CLI if you still want it; otherwise delete this block.
RUN curl -fsSL https://claude.ai/install.sh | bash -s latest \
 && install -m 0755 /root/.local/bin/claude /usr/local/bin/claude

RUN printf '%s\n' '#!/usr/bin/env bash' 'exec claude --dangerously-skip-permissions "$@"' \
    > /usr/local/bin/cc \
 && chmod +x /usr/local/bin/cc

WORKDIR /workspace
CMD ["/bin/bash"]
