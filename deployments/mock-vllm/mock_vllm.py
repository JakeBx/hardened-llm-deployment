"""
HuggingFace SmolLM2-135M OpenAI-Compatible Server for Local Testing

This server implements the OpenAI API endpoints using the HuggingFace
SmolLM2-135M model for actual inference, suitable for local development
and testing without requiring large GPU resources.

Implements:
- GET /health - Health check
- GET /v1/models - List models
- POST /v1/chat/completions - Chat completions (streaming and non-streaming)
- POST /v1/completions - Text completions

Uses the SmolLM2-135M model from HuggingFace for real inference.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")
RESPONSE_DELAY_MS = int(os.getenv("RESPONSE_DELAY_MS", "10"))
MAX_MODEL_LENGTH = int(os.getenv("MAX_MODEL_LENGTH", "2048"))
DEVICE = os.getenv("DEVICE", "auto")  # "auto", "cpu", "cuda", "mps"

app = FastAPI(title="SmolLM2-135M Server", version="1.0.0")

# Global model and tokenizer
model = None
tokenizer = None
model_loaded = False


# =============================================================================
# Models
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 256
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    n: Optional[int] = 1
    stream: Optional[bool] = False
    echo: Optional[bool] = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "huggingface"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# Model Loading
# =============================================================================

def get_device():
    """Determine the best available device."""
    if DEVICE != "auto":
        return DEVICE
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model():
    """Load the SmolLM2-135M model and tokenizer."""
    global model, tokenizer, model_loaded
    
    device = get_device()
    logger.info(f"Loading model {MODEL_NAME} on device: {device}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings for the device
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(device)
        
        model.eval()
        model_loaded = True
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    load_model()


# =============================================================================
# Text Generation
# =============================================================================

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages into a prompt string for the model."""
    # SmolLM2-Instruct uses a specific chat format
    formatted_messages = []
    for msg in messages:
        if msg.role == "system":
            formatted_messages.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif msg.role == "user":
            formatted_messages.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif msg.role == "assistant":
            formatted_messages.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    
    # Add the assistant prompt to start generation
    formatted_messages.append("<|im_start|>assistant\n")
    
    return "\n".join(formatted_messages)


def generate_response(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> tuple[str, int, int]:
    """
    Generate a response using the loaded model.
    
    Returns:
        Tuple of (generated_text, prompt_tokens, completion_tokens)
    """
    global model, tokenizer
    
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_LENGTH - max_tokens
    ).to(device)
    
    prompt_tokens = inputs["input_ids"].shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract only the generated tokens (excluding the prompt)
    generated_tokens = outputs[0][prompt_tokens:]
    completion_tokens = len(generated_tokens)
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up any trailing special tokens or markers
    if "<|im_end|>" in generated_text:
        generated_text = generated_text.split("<|im_end|>")[0]
    
    return generated_text.strip(), prompt_tokens, completion_tokens


async def generate_streaming_response(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """
    Generate tokens one at a time for streaming.
    
    Yields:
        Tuples of (token_text, is_finished)
    """
    global model, tokenizer
    
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_LENGTH - max_tokens
    ).to(device)
    
    generated_ids = inputs["input_ids"].clone()
    
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                yield "", True
                return
            
            # Decode the token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            
            # Check for special end markers
            if "<|im_end|>" in token_text:
                yield "", True
                return
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            yield token_text, False
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(RESPONSE_DELAY_MS / 1000)
    
    yield "", True


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint (vLLM compatible)."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI API compatible)."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id=MODEL_NAME,
                created=int(time.time()) - 86400,  # Created "yesterday"
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint (OpenAI API compatible).
    
    Supports both streaming and non-streaming responses.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"Chat completion request: model={request.model}, stream={request.stream}")
    
    # Format the prompt
    prompt = format_chat_prompt(request.messages)
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    if request.stream:
        # Streaming response
        async def generate_stream():
            # First chunk with role
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"
            
            # Stream tokens
            async for token_text, is_finished in generate_streaming_response(
                prompt,
                max_tokens=request.max_tokens or 256,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.95,
            ):
                if is_finished:
                    break
                    
                if token_text:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final chunk with finish_reason
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    else:
        # Non-streaming response
        response_text, prompt_tokens, completion_tokens = generate_response(
            prompt,
            max_tokens=request.max_tokens or 256,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
        )
        
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    Text completions endpoint (OpenAI API compatible).
    
    This is the older completion API, less commonly used with chat models.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"Completion request: model={request.model}")
    
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    # Generate completion
    response_text, prompt_tokens, completion_tokens = generate_response(
        request.prompt,
        max_tokens=request.max_tokens or 256,
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.95,
    )
    
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "text": response_text,
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting SmolLM2-135M server on port {port}")
    logger.info(f"Model: {MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=port)
