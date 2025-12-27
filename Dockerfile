# Multi-stage Dockerfile for Predictive Maintenance ML Service
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p artifacts/models artifacts/logs artifacts/reports data/raw data/transformed data/validated \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Default command - run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Development
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython

USER appuser

# Override command for development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Training
FROM production as training

USER root

# Additional packages for training
RUN pip install --no-cache-dir \
    jupyter \
    tensorboard

USER appuser

# Training command
CMD ["python", "main.py"]
