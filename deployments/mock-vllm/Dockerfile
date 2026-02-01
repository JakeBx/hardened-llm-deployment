FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY mock_vllm.py .

# Run as non-root user
RUN useradd -m -u 1000 vllmuser
USER vllmuser

EXPOSE 8000

CMD ["python", "mock_vllm.py"]
