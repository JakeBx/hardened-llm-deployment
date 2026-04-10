"""
FastAPI Application for Machine Translation Orchestrator.

KServe InferenceGraph orchestrator for multi-model machine translation.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import TranslationEngine
from .lid import LanguageIdentifier
from .models import TranslationRequest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup: Initialize HTTP client and translation engine
    http_client = httpx.AsyncClient(timeout=60.0)
    app.state.http_client = http_client
    app.state.translation_engine = TranslationEngine(http_client)
    logger.info("Machine Translation Orchestrator started")
    yield
    # Shutdown: Clean up resources
    await http_client.aclose()


app = FastAPI(
    title="Machine Translation Orchestrator",
    description="KServe InferenceGraph orchestrator for multi-model machine translation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    return {"status": "ready"}


@app.post("/v1/translate")
async def translate(request: Request, body: TranslationRequest):
    """
    Translate text using the optimal model based on language and text characteristics.
    
    The orchestrator will:
    1. Detect the source language (if not provided)
    2. Characterize the text (informal, formal, technical, etc.)
    3. Pre-process the text (de-noise, segment sentences)
    4. Route to the optimal translation model
    5. Post-process the result
    """
    translation_engine = request.app.state.translation_engine
    
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if body.stream:
        # Return streaming response
        return StreamingResponse(
            translation_engine.translate_stream(body),
            media_type="text/event-stream"
        )
    
    return await translation_engine.translate(body)


@app.post("/v1/translate/batch")
async def translate_batch(request: Request, body: List[TranslationRequest]):
    """
    Batch translate multiple texts concurrently.
    
    Optimal for processing multiple documents or segments.
    """
    translation_engine = request.app.state.translation_engine
    
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Execute all translations concurrently
    tasks = [translation_engine.translate(req) for req in body]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to error responses
    responses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            responses.append({
                "index": i,
                "error": str(result),
                "success": False
            })
        else:
            responses.append({
                "index": i,
                "result": result.model_dump(),
                "success": True
            })
    
    return {"results": responses}


@app.post("/v1/identify")
async def identify_language(request: Request, text: str):
    """
    Identify the language of the given text.
    
    Uses FastText for accurate detection across 200+ languages.
    """
    translation_engine = request.app.state.translation_engine
    http_client = request.app.state.http_client
    
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    lid = LanguageIdentifier(http_client)
    result = await lid.identify(text)
    return {
        "language": result.language_code,
        "confidence": result.confidence,
        "script": result.script,
        "language_family": LanguageIdentifier.get_language_family(result.language_code).value
    }


@app.post("/v1/characterize")
async def characterize_text(request: Request, text: str):
    """
    Characterize text to determine its category and properties.
    
    Returns:
    - category: short_informal, long_formal, literary_nuanced, or technical
    - word_count: Number of words
    - formality_score: 0.0 (informal) to 1.0 (formal)
    - technical_density: Ratio of technical terms
    """
    translation_engine = request.app.state.translation_engine
    
    if not translation_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = translation_engine.characterizer.characterize(text)
    return {
        "category": result.category.value,
        "word_count": result.word_count,
        "avg_sentence_length": result.avg_sentence_length,
        "formality_score": result.formality_score,
        "technical_density": result.technical_density
    }


@app.get("/v1/models")
async def list_models():
    """List available translation models and their specializations."""
    return {
        "models": [
            {
                "id": "qwen-mt-7b",
                "name": "Qwen/Qwen-MT-7B",
                "specialization": "Chinese, Japanese, Korean",
                "category": "asian_specialist"
            },
            {
                "id": "nllb-200-3b",
                "name": "facebook/nllb-200-3.3B",
                "specialization": "200+ languages, Russian, Hindi, Bengali",
                "category": "generalist"
            },
            {
                "id": "llama-3-8b-instruct",
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "specialization": "Spanish, literary/nuanced content",
                "category": "high_quality_llm"
            },
            {
                "id": "mistral-nemo-12b",
                "name": "mistralai/Mistral-Nemo-12B",
                "specialization": "French, long documents (128k context)",
                "category": "high_quality_llm"
            },
            {
                "id": "gemma-x2-28-9b",
                "name": "Gemma-X2-28-9B",
                "specialization": "Arabic, 28 most spoken languages",
                "category": "multilingual"
            },
            {
                "id": "qwen2-1-5b-instruct",
                "name": "Qwen2-1.5B-Instruct",
                "specialization": "Fast/real-time, informal content",
                "category": "fast"
            },
            {
                "id": "nllb-200-distilled-1b",
                "name": "facebook/nllb-200-distilled-1.3B",
                "specialization": "Technical documentation",
                "category": "technical"
            }
        ]
    }
