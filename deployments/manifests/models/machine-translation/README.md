# Machine Translation InferenceGraph

This directory contains the KServe InferenceGraph deployment for an orchestrated machine translation service with intelligent model routing.

## Architecture

The translation pipeline follows a four-stage architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Machine Translation Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │    Stage A   │    │    Stage B   │    │    Stage C   │    │  Stage D  │ │
│  │  Language    │───▶│    Text      │───▶│    Pre-      │───▶│   Model   │ │
│  │ Identification│    │Characterize │    │ processing   │    │ Selection │ │
│  │   (LID)      │    │              │    │              │    │           │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────┬─────┘ │
│                                                                     │       │
│  ┌─────────────────────────────────────────────────────────────────┼───────┤
│  │                     Translation Models                          │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────▼──────┐│
│  │  │ Qwen-MT-7B  │  │ NLLB-200-3B │  │ Llama-3.1-8B│  │                 ││
│  │  │ (Chinese/   │  │ (Generalist │  │ (Spanish/   │  │   Router        ││
│  │  │  Japanese/  │  │  200+ langs │  │  Literary)  │  │   (Orchestrator)││
│  │  │  Korean)    │  │  Russian/   │  │             │  │                 ││
│  │  └─────────────┘  │  Hindi)     │  └─────────────┘  └─────────────────┘│
│  │                   └─────────────┘                                      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  │Mistral-Nemo │  │Gemma-X2-28  │  │ Qwen2-1.5B  │  │NLLB-distill │   │
│  │  │   -12B      │  │    -9B      │  │ (Fast/RT)   │  │   -1.3B     │   │
│  │  │ (French/    │  │ (Arabic)    │  │             │  │ (Technical) │   │
│  │  │  Long Docs) │  │             │  │             │  │             │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│  └──────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `inference-graph.yaml` | KServe InferenceGraph definition with orchestrator and LID services |
| `translation-models.yaml` | All 7 translation model InferenceServices |
| `pvcs.yaml` | Persistent Volume Claims for model storage |
| `kustomization.yaml` | Kustomize configuration for deployment |

## Translation Models

### Language-Specific Models

| Language | Model | Why This Model |
|----------|-------|----------------|
| Chinese (zh) | `Qwen/Qwen-MT-7B` | Specifically tuned for EN-ZH; outperforms GPT-4 in fluency |
| Russian (ru) | `facebook/nllb-200-3.3B` | Highly robust for Cyrillic nuances and complex grammar |
| Spanish (es) | `meta-llama/Llama-3.1-8B-Instruct` | Exceptional at capturing regional dialects and formal/informal tone |
| French (fr) | `mistralai/Mistral-Nemo-12B` | Strong native performance in European languages; very high BLEU scores |
| Arabic (ar) | `Gemma-X2-28-9B` | Optimized for the 28 most spoken languages, excellent Arabic support |
| Hindi (hi) | `facebook/nllb-200-3.3B` | Purpose-built for South Asian languages and low-resource scripts |
| Bengali (bn) | `facebook/nllb-200-3.3B` | Unmatched accuracy in Bengali script/morphology |

### Characterization-Based Models

| Category | Model | Use Case |
|----------|-------|----------|
| Long Documents | `Mistral-Nemo-12B` | 128k context window for 50+ page PDFs |
| Short/Informal | `Qwen2-1.5B-Instruct` | Real-time chat, tweets, social media |
| Technical/Formal | `NLLB-200-distilled-1.3B` | API docs, technical manuals, faithful translations |
| Literary/Nuanced | `Llama-3.1-8B-Instruct` | Marketing copy, creative writing |

## Text Categories

The orchestrator characterizes input text into four categories:

1. **short_informal**: Real-time chat, tweets, social media (< 100 words, informal language)
2. **long_formal**: Legal documents, medical papers, technical manuals (> 500 words, formal register)
3. **literary_nuanced**: Marketing copy, creative writing (variable length, nuanced content)
4. **technical**: Code documentation, API specs, technical terminology

## GPU Requirements

| Model | GPU Requirement | Memory |
|-------|-----------------|--------|
| LID FastText | CPU only | 2 GB |
| MT Orchestrator | CPU only | 1 GB |
| Qwen-MT-7B | 1x A100 (40GB) | 32-64 GB |
| NLLB-200-3.3B | 1x A100 (40GB) | 32-64 GB |
| Llama-3.1-8B-Instruct | 1x A100 (40GB) | 32-64 GB |
| Mistral-Nemo-12B | 2x A100 (40GB) | 64-128 GB |
| Gemma-X2-28-9B | 2x A100 (40GB) | 48-96 GB |
| Qwen2-1.5B-Instruct | 1x T4/A10 | 8-16 GB |
| NLLB-distilled-1.3B | 1x T4/A10 | 8-16 GB |

**Total Cluster Requirement**: Minimum 10x A100 (40GB) GPUs for full deployment

## Deployment

### Prerequisites

1. KServe v0.11+ installed with InferenceGraph support
2. GPU nodes with NVIDIA drivers
3. HuggingFace credentials for gated models (Llama, Gemma)

### Build Container Images

```bash
# Build MT Orchestrator
cd deployments/mt-orchestrator
docker build -t your-registry/mt-orchestrator:latest .
docker push your-registry/mt-orchestrator:latest

# Build LID FastText service
cd deployments/lid-fasttext
docker build -t your-registry/lid-fasttext:latest .
docker push your-registry/lid-fasttext:latest
```

### Configure HuggingFace Token

```bash
kubectl create secret generic huggingface-credentials \
  --from-literal=token=hf_your_token_here \
  -n ml-inference
```

### Deploy

```bash
# Deploy base infrastructure first
kubectl apply -k deployments/manifests/base

# Deploy machine translation pipeline
kubectl apply -k deployments/manifests/models/machine-translation
```

### Verify Deployment

```bash
# Check InferenceGraph status
kubectl get inferencegraph machine-translation -n ml-inference

# Check all InferenceServices
kubectl get inferenceservices -n ml-inference -l app.kubernetes.io/component=translation-model

# Check pods
kubectl get pods -n ml-inference -l app.kubernetes.io/part-of=ml-inference-platform
```

## API Usage

### Basic Translation

```bash
# Get the InferenceGraph URL
MT_URL=$(kubectl get inferencegraph machine-translation -n ml-inference -o jsonpath='{.status.url}')

# Translate text (auto-detect source language)
curl -X POST "${MT_URL}/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour, comment allez-vous?",
    "target_lang": "en"
  }'
```

### Response

```json
{
  "translated_text": "Hello, how are you?",
  "source_lang": "fr",
  "source_lang_confidence": 0.99,
  "target_lang": "en",
  "model_used": "mistral-nemo-12b",
  "text_category": "short_informal",
  "preprocessing_applied": [
    "lid_detection",
    "characterized_as_short_informal",
    "text_cleaned",
    "routed_to_mistral",
    "postprocessed"
  ],
  "latency_ms": 245.3
}
```

### Streaming Translation

```bash
curl -X POST "${MT_URL}/v1/translate" \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "text": "这是一个很长的中文文档...",
    "target_lang": "en",
    "stream": true
  }'
```

### With Custom Glossary

```bash
curl -X POST "${MT_URL}/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "El sistema KServe gestiona los modelos de ML.",
    "target_lang": "en",
    "glossary": {
      "KServe": "KServe",
      "ML": "machine learning"
    }
  }'
```

### Batch Translation

```bash
curl -X POST "${MT_URL}/v1/translate/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "Hola mundo", "target_lang": "en"},
    {"text": "Bonjour le monde", "target_lang": "en"},
    {"text": "你好世界", "target_lang": "en"}
  ]'
```

### Language Identification Only

```bash
curl -X POST "${MT_URL}/v1/identify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет, как дела?"}'
```

### Text Characterization

```bash
curl -X POST "${MT_URL}/v1/characterize" \
  -H "Content-Type: application/json" \
  -d '{"text": "This API endpoint returns the user profile data including name, email, and preferences..."}'
```

### List Available Models

```bash
curl "${MT_URL}/v1/models"
```

## Pre-processing Pipeline

The orchestrator applies the following pre-processing steps:

1. **De-noising**
   - Remove HTML tags (using BeautifulSoup)
   - Normalize whitespace
   - Preserve URLs and email addresses

2. **Sentence Splitting**
   - Uses PySBD (Python Sentence Boundary Disambiguation)
   - Handles abbreviations and edge cases correctly

3. **Entity Preservation**
   - Extract URLs, emails, quoted terms
   - Extract code/technical terms in backticks
   - Preserve during translation

4. **Glossary Injection**
   - Apply custom terminology mappings
   - Constrained generation for specific terms

## Model Routing Logic

The orchestrator selects models using this priority order:

1. **Language-specific routing**: If source language matches a specialized model
2. **Text length shortcut**: Very short informal text → Qwen2-1.5B (fast)
3. **Category-based routing**: Based on characterization results
4. **Default fallback**: NLLB-200-3.3B (broadest language support)

### Routing Table

```
Source Lang   | Category         | Selected Model
--------------|------------------|------------------
zh, ja, ko    | *                | Qwen-MT-7B
ru, uk, bg    | *                | NLLB-200-3.3B
es            | *                | Llama-3.1-8B
fr            | *                | Mistral-Nemo-12B
ar            | *                | Gemma-X2-28-9B
hi, bn, ta    | *                | NLLB-200-3.3B
*             | short_informal   | Qwen2-1.5B
*             | technical        | NLLB-distilled-1.3B
*             | long_formal      | Mistral-Nemo-12B
*             | literary_nuanced | Llama-3.1-8B
*             | *                | NLLB-200-3.3B
```

## Scaling Configuration

Each model has independent autoscaling:

| Model | Min Replicas | Max Replicas | Scale Target |
|-------|--------------|--------------|--------------|
| MT Orchestrator | 1 | 10 | 10 concurrent |
| LID FastText | 1 | 5 | 20 concurrent |
| Qwen-MT-7B | 0 | 3 | 2 concurrent |
| NLLB-200-3.3B | 1 | 5 | 3 concurrent |
| Llama-3.1-8B | 0 | 3 | 2 concurrent |
| Mistral-Nemo-12B | 0 | 2 | 1 concurrent |
| Gemma-X2-28-9B | 0 | 2 | 1 concurrent |
| Qwen2-1.5B | 1 | 10 | 10 concurrent |
| NLLB-distilled | 1 | 5 | 5 concurrent |

## Troubleshooting

### Model Fails to Load

```bash
# Check pod logs for specific model
kubectl logs -l serving.kserve.io/inferenceservice=qwen-mt-7b -n ml-inference -c vllm

# Check if HF token is configured
kubectl get secret huggingface-credentials -n ml-inference -o jsonpath='{.data.token}' | base64 -d
```

### Routing Issues

```bash
# Check orchestrator logs
kubectl logs -l app=mt-orchestrator -n ml-inference

# Test language detection directly
curl -X POST "http://lid-fasttext.ml-inference.svc.cluster.local:8080/v1/identify" \
  -H "Content-Type: application/json" \
  -d '{"text": "test text"}'
```

### High Latency

1. Check if models are scaled up (cold start can take 2-3 minutes)
2. Verify GPU utilization: `kubectl top pods -n ml-inference`
3. Consider increasing `minReplicas` for frequently used models

### Memory OOM

1. Reduce `--max-model-len` in vLLM args
2. Reduce `--gpu-memory-utilization` (default 0.9)
3. Use quantized model variants where available
