"""
External Service Stubs for Development/Testing

These stubs implement the external service APIs and always return success/true.
In production, these would be replaced with actual external services.

Services implemented:
1. ABAC Authorization Service - POST /authorize
2. Classification Combining Service - POST /combine
3. Audit Log Service - POST /audit-events
4. Observability Service - POST /metrics, POST /traces

Run mode is selected via SERVICE_TYPE environment variable:
- abac: ABAC Authorization Service
- classification: Classification Combining Service
- audit: Audit Log Service
- observability: Observability Service
- all: All services on different paths (default for testing)
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICE_TYPE = os.getenv("SERVICE_TYPE", "all")

app = FastAPI(title=f"External Service Stub ({SERVICE_TYPE})", version="1.0.0")


# ============================================================================
# ABAC Authorization Service Models and Endpoints
# ============================================================================

class AuthorizeRequest(BaseModel):
    user_dn: str
    data_classification: str
    model_classification: str


class AuthorizeResponse(BaseModel):
    authorized: bool
    reason: Optional[str] = None


@app.post("/authorize", response_model=AuthorizeResponse)
async def authorize(request: AuthorizeRequest):
    """
    ABAC Authorization endpoint - STUB always returns authorized: true
    
    In production, this would:
    - Validate user DN against directory service
    - Check user's clearance level
    - Compare against data and model classifications
    - Apply ABAC policies
    """
    logger.info(f"ABAC Authorization request: user_dn={request.user_dn}, "
                f"data_classification={request.data_classification}, "
                f"model_classification={request.model_classification}")
    
    return AuthorizeResponse(
        authorized=True,
        reason="STUB: All requests authorized for development/testing"
    )


# ============================================================================
# Classification Combining Service Models and Endpoints
# ============================================================================

class CombineRequest(BaseModel):
    request_classification: str
    model_classification: str


class CombineResponse(BaseModel):
    aggregated_classification: str


# Classification hierarchy (higher index = more sensitive)
CLASSIFICATION_HIERARCHY = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET"]


def get_higher_classification(class1: str, class2: str) -> str:
    """Return the higher (more sensitive) classification"""
    try:
        idx1 = CLASSIFICATION_HIERARCHY.index(class1.upper())
        idx2 = CLASSIFICATION_HIERARCHY.index(class2.upper())
        return CLASSIFICATION_HIERARCHY[max(idx1, idx2)]
    except ValueError:
        # If unknown classification, return the model classification as default
        return class2


@app.post("/combine", response_model=CombineResponse)
async def combine_classifications(request: CombineRequest):
    """
    Classification Combining endpoint - returns the higher classification
    
    In production, this might implement more complex rules based on
    organizational policies (e.g., certain combinations might result
    in specific override classifications).
    """
    logger.info(f"Classification combine request: "
                f"request_classification={request.request_classification}, "
                f"model_classification={request.model_classification}")
    
    aggregated = get_higher_classification(
        request.request_classification,
        request.model_classification
    )
    
    logger.info(f"Aggregated classification: {aggregated}")
    
    return CombineResponse(aggregated_classification=aggregated)


# ============================================================================
# Audit Log Service Models and Endpoints
# ============================================================================

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


class AuditResponse(BaseModel):
    status: str
    event_id: str


# In-memory audit log for debugging
audit_log: List[Dict[str, Any]] = []


@app.post("/audit-events", response_model=AuditResponse)
async def log_audit_event(event: AuditEvent):
    """
    Audit Log endpoint - STUB logs to console and memory
    
    In production, this would:
    - Persist to a secure audit store
    - Ensure tamper-proof logging
    - Support compliance queries
    """
    event_dict = event.model_dump()
    event_id = f"audit-{len(audit_log) + 1:06d}"
    event_dict["event_id"] = event_id
    
    audit_log.append(event_dict)
    
    logger.info(f"Audit event [{event.event_type}]: user={event.user_dn}, "
                f"model={event.model}, authorized={event.authorized}")
    
    return AuditResponse(status="logged", event_id=event_id)


@app.get("/audit-events")
async def get_audit_events(limit: int = 100):
    """Get recent audit events (for debugging)"""
    return {"events": audit_log[-limit:], "total": len(audit_log)}


# ============================================================================
# Observability Service Models and Endpoints
# ============================================================================

class MetricsData(BaseModel):
    request_count: Optional[int] = None
    latency_ms: Optional[float] = None
    model_classification: Optional[str] = None
    data_classification: Optional[str] = None


class MetricsResponse(BaseModel):
    status: str


class TracesData(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None


# In-memory metrics for debugging
metrics_buffer: List[Dict[str, Any]] = []


@app.post("/metrics", response_model=MetricsResponse)
async def receive_metrics(metrics: MetricsData):
    """
    Metrics endpoint - STUB logs to console
    
    In production, this would:
    - Forward to Prometheus/OTLP collector
    - Aggregate metrics
    - Support alerting
    """
    metrics_dict = metrics.model_dump()
    metrics_dict["received_at"] = datetime.utcnow().isoformat()
    metrics_buffer.append(metrics_dict)
    
    logger.info(f"Metrics received: latency={metrics.latency_ms}ms, "
                f"model={metrics.model_classification}")
    
    return MetricsResponse(status="received")


@app.post("/traces", response_model=MetricsResponse)
async def receive_traces(traces: TracesData):
    """
    Traces endpoint - STUB logs to console
    
    In production, this would:
    - Forward to Jaeger/Zipkin/OTLP collector
    - Store for distributed tracing
    """
    logger.info(f"Trace received: trace_id={traces.trace_id}, "
                f"operation={traces.operation}")
    
    return MetricsResponse(status="received")


@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get metrics summary (for debugging)"""
    if not metrics_buffer:
        return {"count": 0, "avg_latency_ms": 0}
    
    latencies = [m.get("latency_ms", 0) for m in metrics_buffer if m.get("latency_ms")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    return {
        "count": len(metrics_buffer),
        "avg_latency_ms": avg_latency,
        "recent": metrics_buffer[-10:]
    }


# ============================================================================
# Health and Ready Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service_type": SERVICE_TYPE}


@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    return {"status": "ready", "service_type": SERVICE_TYPE}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting {SERVICE_TYPE} service stub on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
