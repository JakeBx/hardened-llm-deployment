"""
Auth Transformer for KServe InferenceService - Streaming Enabled

This transformer:
1. Request path: Calls ABAC service for authorization (parallel)
2. Request path: Pre-computes aggregated classification (parallel)
3. Sets classification header BEFORE streaming begins
4. Response path: Streams tokens directly without buffering
5. Async: Sends audit events and observability data after completion

Key Design: Pre-computed Classification for Streaming
- Classification is computed BEFORE inference starts
- Headers are set before first token is streamed
- No response buffering required

Environment Variables:
- MODEL_CLASSIFICATION: Classification level of the model (e.g., SECRET, CONFIDENTIAL)
- ABAC_SERVICE_URL: URL of the ABAC authorization service
- CLASSIFICATION_SERVICE_URL: URL of the classification combining service
- AUDIT_SERVICE_URL: URL of the audit log service
- OBSERVABILITY_SERVICE_URL: URL of the observability service
- PREDICTOR_HOST: Hostname of the predictor service (set by KServe)
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
MODEL_CLASSIFICATION = os.getenv("MODEL_CLASSIFICATION", "INTERNAL")
ABAC_SERVICE_URL = os.getenv("ABAC_SERVICE_URL", "http://abac-stub:8080")
CLASSIFICATION_SERVICE_URL = os.getenv("CLASSIFICATION_SERVICE_URL", "http://classification-stub:8080")
AUDIT_SERVICE_URL = os.getenv("AUDIT_SERVICE_URL", "http://audit-stub:8080")
OBSERVABILITY_SERVICE_URL = os.getenv("OBSERVABILITY_SERVICE_URL", "http://observability-stub:8080")
PREDICTOR_HOST = os.getenv("PREDICTOR_HOST", "localhost:8081")

app = FastAPI(title="Auth Transformer (Streaming)", version="2.0.0")

# HTTP client for external services
http_client: Optional[httpx.AsyncClient] = None


class AuthRequest(BaseModel):
    user_dn: str
    data_classification: str
    model_classification: str


class ClassificationRequest(BaseModel):
    request_classification: str
    model_classification: str


class AuditEvent(BaseModel):
    event_type: str
    user_dn: str
    model: str
    data_classification: str
    model_classification: str
    aggregated_classification: Optional[str] = None
    authorized: Optional[bool] = None
    denial_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: float


@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    logger.info(f"Streaming Transformer started with MODEL_CLASSIFICATION={MODEL_CLASSIFICATION}")


@app.on_event("shutdown")
async def shutdown():
    global http_client
    if http_client:
        await http_client.aclose()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    return {"status": "ready"}


async def call_abac_service(user_dn: str, data_classification: str) -> tuple[bool, Optional[str]]:
    """Call the ABAC authorization service"""
    try:
        response = await http_client.post(
            f"{ABAC_SERVICE_URL}/authorize",
            json={
                "user_dn": user_dn,
                "data_classification": data_classification,
                "model_classification": MODEL_CLASSIFICATION,
            },
        )
        response.raise_for_status()
        result = response.json()
        return result.get("authorized", False), result.get("reason")
    except Exception as e:
        logger.error(f"ABAC service call failed: {e}")
        # Fail closed - deny if ABAC service is unavailable
        return False, f"ABAC service unavailable: {e}"


async def call_classification_service(data_classification: str) -> str:
    """Call the classification combining service (pre-compute before inference)"""
    try:
        response = await http_client.post(
            f"{CLASSIFICATION_SERVICE_URL}/combine",
            json={
                "request_classification": data_classification,
                "model_classification": MODEL_CLASSIFICATION,
            },
        )
        response.raise_for_status()
        result = response.json()
        return result.get("aggregated_classification", MODEL_CLASSIFICATION)
    except Exception as e:
        logger.error(f"Classification service call failed: {e}")
        # Return the higher classification as fallback
        return MODEL_CLASSIFICATION


async def send_audit_event(event: AuditEvent):
    """Send audit event asynchronously (fire and forget)"""
    try:
        await http_client.post(
            f"{AUDIT_SERVICE_URL}/audit-events",
            json=event.model_dump(),
        )
    except Exception as e:
        logger.warning(f"Failed to send audit event: {e}")


async def send_observability_data(metrics: Dict[str, Any]):
    """Send observability data asynchronously (fire and forget)"""
    try:
        await http_client.post(
            f"{OBSERVABILITY_SERVICE_URL}/metrics",
            json=metrics,
        )
    except Exception as e:
        logger.warning(f"Failed to send observability data: {e}")


async def stream_from_predictor(
    request: Request, 
    body: bytes
) -> AsyncGenerator[bytes, None]:
    """Stream response from predictor without buffering"""
    predictor_url = f"http://{PREDICTOR_HOST}{request.url.path}"
    
    # Forward headers (excluding host)
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    
    async with http_client.stream(
        method=request.method,
        url=predictor_url,
        headers=headers,
        content=body,
    ) as response:
        async for chunk in response.aiter_bytes():
            yield chunk


async def forward_to_predictor_non_streaming(request: Request, body: bytes) -> httpx.Response:
    """Forward the request to the predictor service (non-streaming fallback)"""
    predictor_url = f"http://{PREDICTOR_HOST}{request.url.path}"
    
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    
    response = await http_client.request(
        method=request.method,
        url=predictor_url,
        headers=headers,
        content=body,
    )
    return response


def should_stream(request: Request) -> bool:
    """Determine if the request should be streamed based on Accept header or path"""
    accept = request.headers.get("accept", "")
    path = request.url.path.lower()
    
    # Stream for SSE, text-event-stream, or chat completions endpoints
    if "text/event-stream" in accept:
        return True
    if "chat/completions" in path or "generate_stream" in path:
        return True
    # Check for stream=true in query params
    if request.query_params.get("stream", "").lower() == "true":
        return True
    return False


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def transform(request: Request, path: str):
    """
    Main transformer endpoint that handles all inference requests with streaming support.
    
    Flow:
    1. Extract user DN and data classification from headers
    2. PARALLEL: Call ABAC service AND Classification service
    3. If denied, return 403
    4. Set x-aggregated-classification header
    5. Stream tokens from predictor to client (no buffering)
    6. After completion, send async audit event
    """
    start_time = time.time()
    
    # Skip health/ready endpoints
    if path in ("health", "ready"):
        return await health() if path == "health" else await ready()
    
    # Extract headers
    user_dn = request.headers.get("x-client-dn", "unknown")
    data_classification = request.headers.get("x-data-classification", "INTERNAL")
    
    logger.info(f"Processing request: user_dn={user_dn}, data_classification={data_classification}")
    
    # =========================================================================
    # PRE-INFERENCE PHASE: ABAC + Classification in PARALLEL
    # This enables streaming by computing classification before inference starts
    # =========================================================================
    
    abac_task = asyncio.create_task(call_abac_service(user_dn, data_classification))
    classification_task = asyncio.create_task(call_classification_service(data_classification))
    
    # Wait for both to complete
    (authorized, denial_reason), aggregated_classification = await asyncio.gather(
        abac_task, classification_task
    )
    
    if not authorized:
        # Send denial audit event (async)
        asyncio.create_task(send_audit_event(AuditEvent(
            event_type="AUTH_DENIED",
            user_dn=user_dn,
            model=MODEL_CLASSIFICATION,
            data_classification=data_classification,
            model_classification=MODEL_CLASSIFICATION,
            authorized=False,
            denial_reason=denial_reason or "ABAC authorization denied",
            timestamp=time.time(),
        )))
        
        raise HTTPException(status_code=403, detail="Authorization denied")
    
    # Send auth success audit event (async)
    asyncio.create_task(send_audit_event(AuditEvent(
        event_type="AUTH_SUCCESS",
        user_dn=user_dn,
        model=MODEL_CLASSIFICATION,
        data_classification=data_classification,
        model_classification=MODEL_CLASSIFICATION,
        aggregated_classification=aggregated_classification,
        authorized=True,
        timestamp=time.time(),
    )))
    
    logger.info(f"Authorized, aggregated_classification={aggregated_classification}")
    
    # Read request body
    body = await request.body()
    
    # =========================================================================
    # STREAMING INFERENCE PHASE
    # Classification header is set BEFORE streaming begins
    # =========================================================================
    
    # Headers to include in response (set before streaming)
    response_headers = {
        "x-aggregated-classification": aggregated_classification,
        "x-model-classification": MODEL_CLASSIFICATION,
    }
    
    if should_stream(request):
        # Streaming response - tokens flow directly through
        logger.info("Using streaming response")
        
        async def stream_with_audit():
            """Wrapper to stream and send audit after completion"""
            token_count = 0
            try:
                async for chunk in stream_from_predictor(request, body):
                    token_count += 1
                    yield chunk
            finally:
                # After stream completes, send completion audit
                latency_ms = (time.time() - start_time) * 1000
                asyncio.create_task(send_audit_event(AuditEvent(
                    event_type="INFERENCE_COMPLETE",
                    user_dn=user_dn,
                    model=MODEL_CLASSIFICATION,
                    data_classification=data_classification,
                    model_classification=MODEL_CLASSIFICATION,
                    aggregated_classification=aggregated_classification,
                    latency_ms=latency_ms,
                    timestamp=time.time(),
                )))
                asyncio.create_task(send_observability_data({
                    "request_count": 1,
                    "latency_ms": latency_ms,
                    "model_classification": MODEL_CLASSIFICATION,
                    "data_classification": data_classification,
                    "streamed": True,
                    "token_chunks": token_count,
                }))
        
        return StreamingResponse(
            stream_with_audit(),
            headers=response_headers,
            media_type="text/event-stream",
        )
    
    else:
        # Non-streaming response (fallback for non-LLM models or batch requests)
        logger.info("Using non-streaming response")
        
        predictor_response = await forward_to_predictor_non_streaming(request, body)
        
        # Calculate latency and send audit
        latency_ms = (time.time() - start_time) * 1000
        
        asyncio.create_task(send_audit_event(AuditEvent(
            event_type="INFERENCE_COMPLETE",
            user_dn=user_dn,
            model=MODEL_CLASSIFICATION,
            data_classification=data_classification,
            model_classification=MODEL_CLASSIFICATION,
            aggregated_classification=aggregated_classification,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )))
        
        asyncio.create_task(send_observability_data({
            "request_count": 1,
            "latency_ms": latency_ms,
            "model_classification": MODEL_CLASSIFICATION,
            "data_classification": data_classification,
            "streamed": False,
        }))
        
        # Merge response headers
        final_headers = dict(predictor_response.headers)
        final_headers.update(response_headers)
        
        from fastapi import Response
        return Response(
            content=predictor_response.content,
            status_code=predictor_response.status_code,
            headers=final_headers,
            media_type=predictor_response.headers.get("content-type"),
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
