# ---------------------------------------------------------------------------
# Dockerfile for UrbanPulse — Smart Mobility Decision Intelligence
# Optimized for Google Cloud Run: small base image, no build cache bloat,
# unbuffered Python output for clean Cloud Logging, listens on 0.0.0.0:8080.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# so logs stream immediately to Cloud Run / Cloud Logging.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS-level dependencies required by common Python wheels (kept
# minimal to preserve a small final image size).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code.
COPY app.py tools.py ./

# Cloud Run injects the PORT environment variable and expects the container
# to listen on it; we default to 8080 for local/documentation consistency.
ENV PORT=8080
EXPOSE 8080

# Basic container health check (optional but recommended for production).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD \
    curl -f http://localhost:8080/_stcore/health || exit 1

# Run Streamlit, explicitly binding to all interfaces on port 8080 as
# required by Cloud Run, and disabling usage-stats collection prompts.
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8080", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
