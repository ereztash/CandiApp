# CandiApp - AI-Powered Resume Parsing Docker Image
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Download spaCy models
RUN python -m spacy download en_core_web_lg && \
    python -m spacy download es_core_news_lg && \
    python -m spacy download fr_core_news_lg && \
    python -m spacy download de_core_news_lg

# Stage 2: Production
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 candiapp && \
    chown -R candiapp:candiapp /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=candiapp:candiapp src/ ./src/
COPY --chown=candiapp:candiapp setup.py README.md ./

# Install package in development mode
RUN pip install -e .

# Create directories for data
RUN mkdir -p /app/data/uploads /app/data/features /app/logs && \
    chown -R candiapp:candiapp /app/data /app/logs

# Switch to non-root user
USER candiapp

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    CANDIAPP_DATA_DIR=/app/data \
    CANDIAPP_LOG_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import candiapp; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "candiapp"]

# Expose port (if running as API)
EXPOSE 8000

# Labels
LABEL maintainer="CandiApp Team" \
      version="1.0.0" \
      description="AI-Powered Resume Parsing & Analysis System"
