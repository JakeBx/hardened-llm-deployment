# ML Inference Deployment (Streaming-Enabled)

This directory contains the implementation of the secure ML inference deployment pattern described in [`plans/ml-inference-deployment-architecture.md`](../plans/ml-inference-deployment-architecture.md).

**Key Features:**
- Token-by-token streaming for LLM responses
- Pre-computed classification (headers set before first token)
- OpenAI-compatible API via vLLM
- ABAC authorization with classification aggregation

## Directory Structure

```
deployments/
├── auth-transformer/           # Auth Transformer container
│   ├── Dockerfile
│   ├── requirements.txt
│   └── transformer.py
├── external-service-stubs/     # Stub implementations for external services
│   ├── Dockerfile
│   ├── requirements.txt
│   └── stubs.py
├── mt-orchestrator/            # Machine Translation Orchestrator
│   ├── Dockerfile
│   ├── requirements.txt
│   └── orchestrator.py
├── lid-fasttext/               # Language Identification Service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── lid_service.py
├── manifests/
│   ├── base/                   # Shared infrastructure
│   │   ├── namespace.yaml
│   │   ├── storage-class.yaml
│   │   ├── external-services-configmap.yaml
│   │   ├── external-service-stubs-deployment.yaml
│   │   └── kustomization.yaml
│   └── models/
│       ├── kimi-k2.5/          # Kimi K2.5 VLM deployment
│       │   ├── pvc.yaml
│       │   ├── inference-service.yaml
│       │   └── kustomization.yaml
│       ├── smollm-360m/        # SmolLM-360M deployment
│       │   ├── pvc.yaml
│       │   ├── inference-service.yaml
│       │   └── kustomization.yaml
│       └── machine-translation/ # Machine Translation InferenceGraph
│           ├── inference-graph.yaml
│           ├── translation-models.yaml
│           ├── pvcs.yaml
│           ├── kustomization.yaml
│           └── README.md
└── README.md
```

## Prerequisites

1. **Kubernetes Cluster** with GPU support (NVIDIA driver + device plugin)
2. **KServe** installed (v0.11+)
3. **Istio** service mesh (for mTLS)
4. **cert-manager** (for certificate management)
5. **kubectl** and **kustomize** CLI tools

## Quick Start

### 1. Build Container Images

```bash
# Build Auth Transformer
cd deployments/auth-transformer
docker build -t auth-transformer:latest .

# Build External Service Stubs
cd deployments/external-service-stubs
docker build -t external-service-stubs:latest .

# Push to your registry (adjust registry URL)
docker tag auth-transformer:latest your-registry/auth-transformer:latest
docker tag external-service-stubs:latest your-registry/external-service-stubs:latest
docker push your-registry/auth-transformer:latest
docker push your-registry/external-service-stubs:latest
```

### 2. Deploy Base Infrastructure

```bash
# Deploy namespace, storage class, and external service stubs
kubectl apply -k deployments/manifests/base
```

### 3. Deploy Models

#### Option A: Deploy SmolLM-360M (lightweight, CPU-compatible)

```bash
# Deploy SmolLM-360M-bnb-4bit
kubectl apply -k deployments/manifests/models/smollm-360m
```

#### Option B: Deploy Kimi K2.5 (requires 4x A100 GPUs)

```bash
# Deploy Kimi K2.5 VLM
kubectl apply -k deployments/manifests/models/kimi-k2.5
```

#### Option C: Deploy Machine Translation Pipeline (InferenceGraph)

The Machine Translation deployment uses a KServe InferenceGraph to orchestrate multiple translation models with intelligent routing based on language and text characteristics.

```bash
# Build MT-specific images first
cd deployments/mt-orchestrator
docker build -t your-registry/mt-orchestrator:latest .
docker push your-registry/mt-orchestrator:latest

cd ../lid-fasttext
docker build -t your-registry/lid-fasttext:latest .
docker push your-registry/lid-fasttext:latest

# Deploy the Machine Translation InferenceGraph
kubectl apply -k deployments/manifests/models/machine-translation
```

See [`deployments/manifests/models/machine-translation/README.md`](manifests/models/machine-translation/README.md) for full documentation.

### 4. Verify Deployment

```bash
# Check InferenceService status
kubectl get inferenceservices -n ml-inference

# Check pods
kubectl get pods -n ml-inference

# Check services
kubectl get svc -n ml-inference
```

## Model Configurations

### Kimi K2.5 (VLM)

| Property | Value |
|----------|-------|
| Model ID | `moonshotai/Kimi-K2.5` |
| Classification | CONFIDENTIAL |
| GPU Requirements | 4x A100 (80GB) |
| Memory | 128-256 GB |
| Storage | 500 GB SSD |
| Use Case | Vision-Language tasks, multimodal understanding |

### SmolLM-360M-bnb-4bit

| Property | Value |
|----------|-------|
| Model ID | `unsloth/SmolLM-360M-bnb-4bit` |
| Classification | INTERNAL |
| GPU Requirements | Optional (can run on CPU) |
| Memory | 2-4 GB |
| Storage | 5 GB SSD |
| Use Case | Lightweight text generation, edge deployment |

### Machine Translation InferenceGraph

A multi-model orchestrated translation service using KServe InferenceGraph. See [`machine-translation/README.md`](manifests/models/machine-translation/README.md) for full details.

**Pipeline Architecture:**
```
Detection → Characterization → Pre-processing → Model Selection → Translation
```

**Translation Models:**

| Model | Specialization | GPU Requirement |
|-------|----------------|-----------------|
| `Qwen/Qwen-MT-7B` | Chinese, Japanese, Korean | 1x A100 (40GB) |
| `facebook/nllb-200-3.3B` | 200+ languages, Russian, Hindi, Bengali | 1x A100 (40GB) |
| `meta-llama/Llama-3.1-8B-Instruct` | Spanish, literary content | 1x A100 (40GB) |
| `mistralai/Mistral-Nemo-12B` | French, long documents (128k context) | 2x A100 (40GB) |
| `Gemma-X2-28-9B` | Arabic, 28 most spoken languages | 2x A100 (40GB) |
| `Qwen2-1.5B-Instruct` | Fast/real-time translation | 1x T4/A10 |
| `NLLB-200-distilled-1.3B` | Technical documentation | 1x T4/A10 |

**Quick API Example:**

```bash
MT_URL=$(kubectl get inferencegraph machine-translation -n ml-inference -o jsonpath='{.status.url}')

curl -X POST "${MT_URL}/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour, comment allez-vous?",
    "target_lang": "en"
  }'
```

## Testing

### Get the Inference URL

```bash
# Get the inference service URL
INFERENCE_URL=$(kubectl get inferenceservice smollm-360m -n ml-inference -o jsonpath='{.status.url}')

# For Kimi K2.5
INFERENCE_URL=$(kubectl get inferenceservice kimi-k2-5 -n ml-inference -o jsonpath='{.status.url}')
```

### OpenAI-Compatible Chat Completions (Streaming)

The models are served via vLLM with OpenAI-compatible API. Use the `/v1/chat/completions` endpoint with `stream: true` for token-by-token streaming:

```bash
# Streaming chat completion (tokens arrive one at a time)
curl -X POST "${INFERENCE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-client-dn: CN=testuser,OU=engineering,O=example,C=AU" \
  -H "x-data-classification: INTERNAL" \
  --cert /path/to/client.crt \
  --key /path/to/client.key \
  -N \
  -d '{
    "model": "smollm-360m",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "stream": true,
    "max_tokens": 256
  }'
```

**Streaming Output Format** (Server-Sent Events):
```
data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Quantum"},"index":0}]}

data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" computing"},"index":0}]}

data: {"id":"cmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" uses"},"index":0}]}

data: [DONE]
```

### Non-Streaming Chat Completion

```bash
# Non-streaming (full response at once)
curl -X POST "${INFERENCE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-client-dn: CN=testuser,OU=engineering,O=example,C=AU" \
  -H "x-data-classification: INTERNAL" \
  --cert /path/to/client.crt \
  --key /path/to/client.key \
  -d '{
    "model": "smollm-360m",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false,
    "max_tokens": 50
  }'
```

### Kimi K2.5 with Vision (Multimodal)

```bash
# Vision-Language request with image URL
curl -X POST "${INFERENCE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-client-dn: CN=analyst,OU=research,O=example,C=AU" \
  -H "x-data-classification: CONFIDENTIAL" \
  --cert /path/to/client.crt \
  --key /path/to/client.key \
  -N \
  -d '{
    "model": "kimi-k2-5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ],
    "stream": true,
    "max_tokens": 512
  }'
```

### Python Client Example

```python
import httpx

INFERENCE_URL = "https://smollm-360m.ml-inference.example.com"

# Streaming with httpx
with httpx.Client(
    cert=("/path/to/client.crt", "/path/to/client.key"),
    verify="/path/to/ca.crt"
) as client:
    with client.stream(
        "POST",
        f"{INFERENCE_URL}/v1/chat/completions",
        headers={
            "x-client-dn": "CN=testuser,OU=engineering,O=example,C=AU",
            "x-data-classification": "INTERNAL"
        },
        json={
            "model": "smollm-360m",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True
        }
    ) as response:
        # Classification header is available immediately
        print(f"Classification: {response.headers.get('x-aggregated-classification')}")
        
        # Stream tokens as they arrive
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                import json
                chunk = json.loads(line[6:])
                content = chunk["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
```

### Check Response Headers

All responses (streaming and non-streaming) include:

| Header | Description | Example |
|--------|-------------|---------|
| `x-aggregated-classification` | Combined classification level | `CONFIDENTIAL` |
| `Content-Type` | Response format | `text/event-stream` (streaming) or `application/json` |

**Note:** For streaming responses, headers are sent **before** the first token arrives. This is enabled by pre-computing the classification during the request phase.

## External Service Stubs

The stubs provide development/testing implementations of:

| Service | Endpoint | Behavior |
|---------|----------|----------|
| ABAC Authorization | `POST /authorize` | Always returns `authorized: true` |
| Classification Combining | `POST /combine` | Returns higher of the two classifications |
| Audit Logging | `POST /audit-events` | Logs to console, stores in memory |
| Observability | `POST /metrics` | Logs to console |

### View Audit Logs

```bash
# Port-forward to the stubs service
kubectl port-forward svc/external-service-stubs -n ml-inference 8080:8080

# View audit events
curl http://localhost:8080/audit-events

# View metrics summary
curl http://localhost:8080/metrics/summary
```

## Production Considerations

### Replace Stub Services

For production, update `external-services-configmap.yaml` to point to real services:

```yaml
data:
  ABAC_SERVICE_URL: "https://abac.your-org.com"
  CLASSIFICATION_SERVICE_URL: "https://classification.your-org.com"
  AUDIT_SERVICE_URL: "https://audit.your-org.com"
  OBSERVABILITY_SERVICE_URL: "https://observability.your-org.com"
```

### Configure HuggingFace Token

For gated models, update the secret:

```bash
kubectl create secret generic huggingface-credentials \
  --from-literal=token=your-hf-token \
  -n ml-inference
```

### Update Storage Class

Configure the storage class for your cloud provider:

- **GKE**: `kubernetes.io/gce-pd` with `type: pd-ssd`
- **EKS**: `kubernetes.io/aws-ebs` with `type: gp3`
- **AKS**: `kubernetes.io/azure-disk` with `storageaccounttype: Premium_LRS`

### Enable Istio mTLS

Configure Istio Gateway for client certificate validation (see architecture document for details).

## Troubleshooting

### Model Fails to Load

```bash
# Check pod logs
kubectl logs -l serving.kserve.io/inferenceservice=smollm-360m -n ml-inference -c kserve-container
```

### Authorization Denied

```bash
# Check transformer logs
kubectl logs -l serving.kserve.io/inferenceservice=smollm-360m -n ml-inference -c auth-transformer
```

### GPU Issues

```bash
# Verify GPU availability
kubectl describe nodes | grep -A5 nvidia.com/gpu
```
