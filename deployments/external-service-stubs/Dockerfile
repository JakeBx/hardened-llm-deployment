FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY stubs.py .

# Run as non-root user
RUN useradd -m -u 1000 stubuser
USER stubuser

EXPOSE 8080

CMD ["python", "stubs.py"]
