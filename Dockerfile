FROM python:3.11-slim AS base

LABEL maintainer="Antigravity Labs"
LABEL description="Muninn: Local-first persistent memory engine for AI agents"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency definition first (layer caching)
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || pip install --no-cache-dir -e "."

# Copy application code
COPY . .

# Muninn environment
ENV MUNINN_DOCKER=1 \
    MUNINN_DATA_DIR=/data \
    MUNINN_HOST=0.0.0.0 \
    MUNINN_PORT=42069 \
    MUNINN_OLLAMA_URL=http://ollama:11434

# Create data volume mount point
VOLUME /data

# Expose server port
EXPOSE 42069

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:42069/health || exit 1

# Run server
CMD ["python", "server.py"]
