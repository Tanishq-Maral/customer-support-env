# ============================================================
# CustomerSupportEnv — Dockerfile
#
# Builds a container that:
#   1. Serves the OpenEnv HTTP API on port 7860 (FastAPI/uvicorn)
#   2. Can run inference.py as a standalone script
#
# Required env vars at runtime:
#   HF_TOKEN       Your Hugging Face API key
#   API_BASE_URL   LLM endpoint  (default: HF Router)
#   MODEL_NAME     Model to use  (default: Llama-3.3-70B-Instruct)
#
# Build:
#   docker build -t customer-support-env .
#
# Run (server):
#   docker run -p 7860:7860 \
#     -e HF_TOKEN=hf_... \
#     -e API_BASE_URL=https://router.huggingface.co/v1 \
#     -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
#     customer-support-env
#
# Run (inference only):
#   docker run --rm \
#     -e HF_TOKEN=hf_... \
#     -e API_BASE_URL=https://router.huggingface.co/v1 \
#     -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
#     customer-support-env python inference.py
# ============================================================

FROM python:3.11-slim

LABEL maintainer="OpenEnv Submission"
LABEL description="CustomerSupportEnv — OpenEnv customer support agent environment"
LABEL version="1.0.0"

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . .

USER user

# Port
EXPOSE 7860

# Runtime env var defaults
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
ENV HF_TOKEN=""
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check — automated ping must return 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default: FastAPI server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
