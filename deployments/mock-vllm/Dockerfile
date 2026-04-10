FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY mock_vllm.py .

# Create cache directory for HuggingFace models with proper permissions
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Run as non-root user
RUN useradd -m -u 1000 vllmuser
USER vllmuser

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

CMD ["python", "mock_vllm.py"]
