# Local Testing Guide: SmolLM2-135M

This guide covers multiple approaches to test the SmolLM2-135M model locally, from running the HuggingFace transformers server directly to full Docker Compose setups with the auth transformer.

## Prerequisites

- **Python 3.11+** with pip
- **Docker** (optional, for containerized testing)
- **GPU** (optional, CPU mode available and works well for SmolLM2-135M)
- **HuggingFace account** (optional, for model caching)

## Quick Reference

| Testing Method | Complexity | GPU Required | Auth Stack |
|----------------|------------|--------------|------------|
| [Direct Server](#option-1-direct-smollm2-server) | Simple | Optional | No |
| [Docker Server](#option-2-docker-smollm2) | Simple | Optional | No |
| [Full Stack Docker](#option-3-full-stack-docker-compose) | Medium | Optional | Yes |
| [Transformers Only](#option-4-lightweight-transformers-testing) | Minimal | Optional | No |

---

## Option 1: Direct SmolLM2 Server

Run the SmolLM2-135M server directly with pip - fastest way to test the model.

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
cd deployments/mock-vllm
pip install -r requirements.txt
```

### Start the Server

```bash
cd deployments/mock-vllm

# CPU mode (default, works on any machine)
python mock_vllm.py

# Or with environment variables
MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct \
DEVICE=cpu \
MAX_MODEL_LENGTH=2048 \
PORT=8000 \
python mock_vllm.py

# For GPU (CUDA) mode
DEVICE=cuda python mock_vllm.py

# For Apple Silicon (MPS) mode
DEVICE=mps python mock_vllm.py
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
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Chat completion (streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Explain AI in simple terms."}],
    "stream": true,
    "max_tokens": 100
  }'
```

---

## Option 2: Docker SmolLM2

Run SmolLM2-135M in a container for consistent environments.

### Build and Run

```bash
cd deployments/mock-vllm

# Build the image
docker build -t smollm2-server .

# Run with HuggingFace cache mounted
docker run -d \
  --name smollm2 \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/app/.cache/huggingface \
  -e DEVICE=cpu \
  smollm2-server

# Verify it's running (may take 1-2 minutes for model download)
docker logs -f smollm2
```

### GPU Mode (NVIDIA)

```bash
docker run -d \
  --name smollm2 \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/app/.cache/huggingface \
  -e DEVICE=cuda \
  smollm2-server
```

### Cleanup

```bash
docker stop smollm2 && docker rm smollm2
```

---

## Option 3: Full Stack Docker Compose

Test the complete auth transformer + SmolLM2 stack locally.

### Start the Stack

```bash
cd deployments

# Build images
docker compose -f docker-compose.local.yaml build

# Start all services
docker compose -f docker-compose.local.yaml up -d

# Watch logs (model download takes 1-2 minutes on first run)
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
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
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
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
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

Test the model without a server using HuggingFace transformers directly.

### Install Dependencies

```bash
pip install transformers accelerate torch
```

### Python Script

```python
#!/usr/bin/env python3
"""Local test script for SmolLM2-135M using transformers."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

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

# Format as chat
messages = [
    {"role": "user", "content": "What is Python?"}
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print(f"\nPrompt:\n{prompt}")

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
    top_p=0.95,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nResponse:\n{response}")
```

### Run the Script

```bash
python test_smollm2.py
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

### Start Auth Transformer

In another terminal:

```bash
cd deployments/auth-transformer
pip install -r requirements.txt

# Point to stubs for external services
export ABAC_SERVICE_URL=http://localhost:8080
export CLASSIFICATION_SERVICE_URL=http://localhost:8080
export AUDIT_SERVICE_URL=http://localhost:8080
export OBSERVABILITY_SERVICE_URL=http://localhost:8080
export PREDICTOR_HOST=localhost:8000
export MODEL_CLASSIFICATION=INTERNAL

# Start on a different port since stubs use 8080
PORT=8082 python transformer.py
```

### Start SmolLM2 Server

In a third terminal:

```bash
cd deployments/mock-vllm
pip install -r requirements.txt
python mock_vllm.py
# Running on http://localhost:8000
```

### Test the Full Flow

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-client-dn: CN=testuser,OU=engineering,O=example,C=AU" \
  -H "x-data-classification: INTERNAL" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Check that audit events were logged
curl http://localhost:8080/audit-events
```

---

## Python Client Example

A complete Python client for testing:

```python
#!/usr/bin/env python3
"""Python client for testing SmolLM2-135M API."""

import httpx
import json

BASE_URL = "http://localhost:8080"  # Auth transformer
# BASE_URL = "http://localhost:8000"  # Direct SmolLM2 server

def chat_completion(prompt: str, stream: bool = False):
    """Send a chat completion request."""
    headers = {
        "Content-Type": "application/json",
        "x-client-dn": "CN=testuser,OU=engineering,O=example,C=AU",
        "x-data-classification": "INTERNAL",
    }
    
    payload = {
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
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
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct

# OR with authentication for gated models
huggingface-cli login
```

### CUDA/GPU Errors

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check MPS (Apple Silicon) availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Out of Memory (OOM)

Reduce max tokens or use CPU mode:

```bash
# Use smaller context
MAX_MODEL_LENGTH=1024 python mock_vllm.py

# Or force CPU mode
DEVICE=cpu python mock_vllm.py
```

### Port Conflicts

```bash
# Check what's using port 8000
lsof -i :8000

# Use a different port
PORT=8001 python mock_vllm.py
```

### Connection Refused Errors

```bash
# Ensure server is ready (takes 1-2 minutes to load model on first run)
curl http://localhost:8000/health

# Check container logs
docker logs smollm2
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
      "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
      "object": "model",
      "created": 1706601600,
      "owned_by": "huggingface"
    }
  ]
}
```

### Chat Completion Response (Non-Streaming)

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1706601600,
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
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
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

### Streaming Response (Server-Sent Events)

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Performance Expectations

| Hardware | Load Time | Tokens/sec | Notes |
|----------|-----------|------------|-------|
| Apple M1/M2 (CPU) | 15-30s | 10-20 | Uses `--device cpu` |
| Apple M1/M2 (MPS) | 15-30s | 20-40 | Uses `--device mps` |
| NVIDIA T4 | 10-20s | 40-80 | Uses `--device cuda` |
| NVIDIA A10 | 5-15s | 80-150 | Uses `--device cuda` |
| Intel CPU | 30-60s | 5-10 | Slowest but functional |

SmolLM2-135M is specifically designed to be lightweight and runs efficiently even on CPU.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `HuggingFaceTB/SmolLM2-135M-Instruct` | HuggingFace model ID |
| `DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `MAX_MODEL_LENGTH` | `2048` | Maximum context length |
| `RESPONSE_DELAY_MS` | `10` | Delay between streaming tokens |
| `PORT` | `8000` | Server port |

---

## Related Documentation

- [Main README](../../README.md) - Deployment overview
- [Architecture Plan](../../../../plans/ml-inference-deployment-architecture.md) - System design
- [Auth Transformer](../../auth-transformer/) - Authorization layer
- [External Stubs](../../external-service-stubs/) - Test service stubs
