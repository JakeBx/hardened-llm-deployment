# Local Testing Guide: SmolLM-360M

This guide covers multiple approaches to test the SmolLM-360M model locally, from running vLLM directly to full Docker Compose setups with the auth transformer.

## Prerequisites

- **Python 3.11+** with pip
- **Docker** (optional, for containerized testing)
- **GPU** with NVIDIA drivers (optional, CPU mode available)
- **HuggingFace account** (optional, for model caching)

## Quick Reference

| Testing Method | Complexity | GPU Required | Auth Stack |
|----------------|------------|--------------|------------|
| [Direct vLLM](#option-1-direct-vllm-server) | Simple | Optional | No |
| [Docker vLLM](#option-2-docker-vllm) | Simple | Optional | No |
| [Full Stack Docker](#option-3-full-stack-docker-compose) | Medium | Optional | Yes |
| [Transformers Only](#option-4-lightweight-transformers-testing) | Minimal | Optional | No |

---

## Option 1: Direct vLLM Server

Run vLLM directly with pip - fastest way to test the model.

### Install vLLM

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install vLLM (GPU)
pip install vllm

# OR for CPU-only (slower but works)
pip install vllm --extra-index-url https://download.pytorch.org/whl/cpu
```

### Start the Server

```bash
# GPU mode (recommended)
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/SmolLM-360M-bnb-4bit \
  --served-model-name smollm-360m \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --trust-remote-code \
  --max-model-len 2048 \
  --host 0.0.0.0 \
  --port 8000

# CPU mode (for machines without GPU)
python -m vllm.entrypoints.openai.api_server \
  --model HuggingFaceTB/SmolLM-360M \
  --served-model-name smollm-360m \
  --device cpu \
  --dtype float32 \
  --trust-remote-code \
  --max-model-len 1024 \
  --host 0.0.0.0 \
  --port 8000
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm-360m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Chat completion (streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "smollm-360m",
    "messages": [{"role": "user", "content": "Explain AI in simple terms."}],
    "stream": true,
    "max_tokens": 100
  }'
```

---

## Option 2: Docker vLLM

Run vLLM in a container for consistent environments.

### GPU Mode

```bash
docker run -d \
  --name smollm-vllm \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model unsloth/SmolLM-360M-bnb-4bit \
  --served-model-name smollm-360m \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --trust-remote-code \
  --max-model-len 2048 \
  --host 0.0.0.0 \
  --port 8000

# Verify it's running
docker logs -f smollm-vllm
```

### CPU Mode (No GPU)

```bash
docker run -d \
  --name smollm-vllm \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_NO_CUDA=1 \
  vllm/vllm-openai:latest \
  --model HuggingFaceTB/SmolLM-360M \
  --served-model-name smollm-360m \
  --device cpu \
  --dtype float32 \
  --trust-remote-code \
  --max-model-len 1024 \
  --host 0.0.0.0 \
  --port 8000
```

### Cleanup

```bash
docker stop smollm-vllm && docker rm smollm-vllm
```

---

## Option 3: Full Stack Docker Compose

Test the complete auth transformer + vLLM stack locally.

### Create `docker-compose.local.yaml`

Create this file in the `deployments/` directory:

```yaml
version: '3.8'

services:
  # External service stubs (ABAC, Classification, Audit, Observability)
  external-stubs:
    build:
      context: ./external-service-stubs
    ports:
      - "8081:8080"
    environment:
      - SERVICE_TYPE=all
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Auth Transformer
  auth-transformer:
    build:
      context: ./auth-transformer
    ports:
      - "8080:8080"
    environment:
      - MODEL_CLASSIFICATION=INTERNAL
      - PREDICTOR_HOST=vllm:8000
      - ABAC_SERVICE_URL=http://external-stubs:8080
      - CLASSIFICATION_SERVICE_URL=http://external-stubs:8080
      - AUDIT_SERVICE_URL=http://external-stubs:8080
      - OBSERVABILITY_SERVICE_URL=http://external-stubs:8080
    depends_on:
      external-stubs:
        condition: service_healthy
      vllm:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  # vLLM with SmolLM
  vllm:
    image: vllm/vllm-openai:latest
    # Uncomment for GPU support:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - VLLM_NO_CUDA=1  # Remove for GPU mode
    command: >
      --model HuggingFaceTB/SmolLM-360M
      --served-model-name smollm-360m
      --device cpu
      --dtype float32
      --trust-remote-code
      --max-model-len 1024
      --host 0.0.0.0
      --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 120s
```

### Start the Stack

```bash
cd deployments

# Build images
docker compose -f docker-compose.local.yaml build

# Start all services
docker compose -f docker-compose.local.yaml up -d

# Watch logs
docker compose -f docker-compose.local.yaml logs -f
```

### Test Through Auth Transformer

```bash
# Health check
curl http://localhost:8080/health

# Chat completion with auth headers (through transformer)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-client-dn: CN=testuser,OU=engineering,O=example,C=AU" \
  -H "x-data-classification: INTERNAL" \
  -d '{
    "model": "smollm-360m",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "max_tokens": 100
  }'

# Streaming with auth
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-client-dn: CN=testuser,OU=engineering,O=example,C=AU" \
  -H "x-data-classification: INTERNAL" \
  -N \
  -d '{
    "model": "smollm-360m",
    "messages": [{"role": "user", "content": "Explain gravity."}],
    "stream": true,
    "max_tokens": 100
  }'
```

### View Audit Logs

```bash
# Check audit events logged by the stubs
curl http://localhost:8081/audit-events

# Check metrics
curl http://localhost:8081/metrics/summary
```

### Cleanup

```bash
docker compose -f docker-compose.local.yaml down
```

---

## Option 4: Lightweight Transformers Testing

Test the model without vLLM using HuggingFace transformers directly.

### Install Dependencies

```bash
pip install transformers accelerate bitsandbytes torch
```

### Python Script

```python
#!/usr/bin/env python3
"""Local test script for SmolLM-360M using transformers."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_id = "HuggingFaceTB/SmolLM-360M"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)

if not torch.cuda.is_available():
    model = model.to("cpu")

# Generate text
prompt = "The meaning of life is"
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = inputs.to("cuda")
else:
    inputs = inputs.to("cpu")

print("\nGenerating...")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nResponse:\n{response}")
```

### Run the Script

```bash
python test_smollm.py
```

---

## Testing the Auth Transformer Separately

Test just the auth transformer without a real model backend.

### Start External Stubs

```bash
cd deployments/external-service-stubs
pip install -r requirements.txt
python stubs.py
# Running on http://localhost:8080
```

### Start Auth Transformer (Mock Predictor)

In another terminal:

```bash
cd deployments/auth-transformer
pip install -r requirements.txt

# Point to stubs for external services, and a mock predictor
export ABAC_SERVICE_URL=http://localhost:8080
export CLASSIFICATION_SERVICE_URL=http://localhost:8080
export AUDIT_SERVICE_URL=http://localhost:8080
export OBSERVABILITY_SERVICE_URL=http://localhost:8080
export PREDICTOR_HOST=localhost:9999  # Fake - will fail but auth flow will work
export MODEL_CLASSIFICATION=INTERNAL

python transformer.py
# Running on http://localhost:8080 (use different port if stubs on 8080)
```

### Test Authorization Flow Only

```bash
# This will authorize but fail at predictor stage (expected)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-client-dn: CN=testuser,OU=engineering,O=example,C=AU" \
  -H "x-data-classification: INTERNAL" \
  -d '{"model": "smollm-360m", "messages": [{"role": "user", "content": "test"}]}'

# Check that audit events were logged
curl http://localhost:8080/audit-events
```

---

## Python Client Example

A complete Python client for testing:

```python
#!/usr/bin/env python3
"""Python client for testing SmolLM-360M API."""

import httpx
import json

BASE_URL = "http://localhost:8080"  # Auth transformer
# BASE_URL = "http://localhost:8000"  # Direct vLLM

def chat_completion(prompt: str, stream: bool = False):
    """Send a chat completion request."""
    headers = {
        "Content-Type": "application/json",
        "x-client-dn": "CN=testuser,OU=engineering,O=example,C=AU",
        "x-data-classification": "INTERNAL",
    }
    
    payload = {
        "model": "smollm-360m",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 100,
    }
    
    if stream:
        with httpx.Client() as client:
            with client.stream("POST", f"{BASE_URL}/v1/chat/completions", 
                             headers=headers, json=payload) as response:
                print(f"Classification: {response.headers.get('x-aggregated-classification', 'N/A')}")
                print("\nResponse: ", end="")
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk["choices"][0]["delta"].get("content", "")
                            print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                print()
    else:
        response = httpx.post(f"{BASE_URL}/v1/chat/completions", 
                             headers=headers, json=payload)
        result = response.json()
        print(f"Classification: {response.headers.get('x-aggregated-classification', 'N/A')}")
        print(f"\nResponse: {result['choices'][0]['message']['content']}")


if __name__ == "__main__":
    print("=== Non-Streaming ===")
    chat_completion("What is Python?", stream=False)
    
    print("\n=== Streaming ===")
    chat_completion("Explain machine learning briefly.", stream=True)
```

---

## Troubleshooting

### Model Download Issues

```bash
# Pre-download the model
huggingface-cli download HuggingFaceTB/SmolLM-360M

# OR with authentication for gated models
huggingface-cli login
huggingface-cli download unsloth/SmolLM-360M-bnb-4bit
```

### CUDA/GPU Errors

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check vLLM GPU support
python -c "from vllm import LLM; print('vLLM imported successfully')"
```

### Out of Memory (OOM)

Reduce model length or use CPU mode:

```bash
# Use shorter context
--max-model-len 512

# Or switch to CPU (slower but stable)
--device cpu --dtype float32
```

### Port Conflicts

```bash
# Check what's using port 8000
lsof -i :8000

# Use a different port
--port 8001
```

### Connection Refused Errors

```bash
# Ensure vLLM is ready (takes 1-2 minutes to load model)
curl http://localhost:8000/health

# Check container logs
docker logs smollm-vllm
```

---

## Expected Responses

### Health Check Response

```json
{"status": "healthy"}
```

### Models List Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "smollm-360m",
      "object": "model",
      "created": 1706601600,
      "owned_by": "vllm"
    }
  ]
}
```

### Chat Completion Response (Non-Streaming)

```json
{
  "id": "cmpl-xxx",
  "object": "chat.completion",
  "created": 1706601600,
  "model": "smollm-360m",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### Streaming Response (Server-Sent Events)

```
data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Performance Expectations

| Hardware | Load Time | Tokens/sec | Notes |
|----------|-----------|------------|-------|
| Apple M1/M2 (CPU) | 30-60s | 5-10 | Use `--device cpu` |
| NVIDIA T4 | 15-30s | 30-50 | Quantized model recommended |
| NVIDIA A10 | 10-20s | 50-100 | Full precision works well |
| Intel CPU | 60-120s | 2-5 | Slowest but functional |

---

## Related Documentation

- [Main README](../../README.md) - Deployment overview
- [Architecture Plan](../../../../plans/ml-inference-deployment-architecture.md) - System design
- [Auth Transformer](../../auth-transformer/) - Authorization layer
- [External Stubs](../../external-service-stubs/) - Test service stubs
