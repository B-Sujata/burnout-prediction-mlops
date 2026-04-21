# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
# Installs all dependencies into a virtual environment so only the venv
# needs to be copied into the lean runtime image.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install system-level build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (leverage Docker layer cache)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# Lean image with only the venv, source code, and data.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="burnout-prediction-project"
LABEL description="AI-Based Student Burnout Prediction System"
LABEL version="2.0.0"

# Activate the venv built in Stage 1
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy project files (respects .dockerignore)
COPY --chown=appuser:appuser . .

# Ensure required runtime directories exist
RUN mkdir -p data/raw data/processed models results logs && \
    chown -R appuser:appuser /app

# Matplotlib non-interactive backend (already set in code but be explicit)
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# MLflow tracking directory
ENV MLFLOW_TRACKING_URI=mlruns

# Copy and set executable entrypoint
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser

# Expose MLflow UI port
EXPOSE 5000

# API keys — pass at runtime, never bake into image
# docker run -e ANTHROPIC_API_KEY=sk-ant-... or -e OPENAI_API_KEY=sk-...
ENV ANTHROPIC_API_KEY=""
ENV OPENAI_API_KEY=""

ENTRYPOINT ["/entrypoint.sh"]

# Default command — run the full pipeline
CMD ["pipeline"]
