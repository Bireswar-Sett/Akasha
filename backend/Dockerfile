FROM python:3.11-slim

# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8080

WORKDIR /app

# ──────────────────────────────────────────────────────────────
# System dependencies
# ──────────────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────────────────────
# Python dependencies (cached layer — only invalidated when
# requirements.txt changes)
# ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────
# Application code
# ──────────────────────────────────────────────────────────────
COPY . .

# ──────────────────────────────────────────────────────────────
# Non-root user (ECS / production security best practice)
# ──────────────────────────────────────────────────────────────
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser \
    && chown -R appuser:appgroup /app

USER appuser

# ──────────────────────────────────────────────────────────────
# Health check
# ECS uses this to determine container readiness.
# ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/status || exit 1

# ──────────────────────────────────────────────────────────────
# Port
# ECS task definitions set PORT via environment; default is 8080.
# ──────────────────────────────────────────────────────────────
EXPOSE ${PORT}

# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
