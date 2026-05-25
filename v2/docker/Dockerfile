# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Compile Python wheels in a separate stage so the final image stays slim.
FROM python:3.11-slim AS builder

# System dependencies required to build TA-Lib and other C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        libta-lib-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the requirements file first so Docker can cache the pip layer.
COPY v2/requirements.txt .

# Install Python dependencies into a custom prefix.
# MetaTrader5 is excluded because it is Windows-only and will fail on Linux.
RUN pip install --upgrade pip && \
    pip install \
        --prefix=/install \
        --no-cache-dir \
        $(grep -v -E '^\s*#|MetaTrader5' requirements.txt)


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="TradingBotV2"
LABEL description="TradingBotV2 — automated paper trading engine"

# Install only the runtime C libraries (no compilers).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libta-lib0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Python packages from the builder stage.
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy the full v2 package into /app/v2 so imports work as "from v2.xxx import ..."
COPY v2/ ./v2/

# Create the data directory that settings.py and the journal write to.
# The docker-compose volume mounts ./data here for persistence.
RUN mkdir -p /app/data

# FastAPI port
EXPOSE 8000

# Health-check: ping the /health endpoint every 30 s.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server.  main.py wires the scheduler + API together;
# for container-only deployments the API server is the process entry point.
CMD ["python", "-m", "uvicorn", "v2.api.api_server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
