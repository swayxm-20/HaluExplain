"""
main.py
-------
HaluExplain — FastAPI Orchestration Layer
==========================================
Wires together the four core modules into a single REST endpoint.

Endpoints
---------
GET  /health          → liveness check + model status
POST /verify_claim    → full hallucination detection pipeline

Startup behaviour
-----------------
All four heavy models are loaded **once** during application startup using
FastAPI's lifespan context manager (recommended replacement for deprecated
@app.on_event("startup") in FastAPI ≥ 0.93).
This approach ensures O(1) cold-start per worker process.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config import API_DESCRIPTION, API_TITLE, API_VERSION, TOP_K_PASSAGES
from app.consistency import ConsistencyAnalyser
from app.entailment import EntailmentEngine
from app.explanation import ExplanationGenerator
from app.models import HealthResponse, VerifyClaimRequest, VerifyClaimResponse
from app.retriever import HybridRetriever

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Global model registry (populated during lifespan startup)
# ══════════════════════════════════════════════════════════════════════════════

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all ML models once on startup and release on shutdown."""
    logger.info("═══════════════════════════════════════════════")
    logger.info("  HaluExplain — loading models…               ")
    logger.info("═══════════════════════════════════════════════")

    try:
        _models["consistency"] = ConsistencyAnalyser()
        _models["retriever"] = HybridRetriever()
        _models["entailment"] = EntailmentEngine()
        _models["explanation"] = ExplanationGenerator()
        _models["ready"] = True
        logger.info("All models loaded successfully ✓")
    except Exception as exc:
        logger.critical("Model loading failed: %s", exc, exc_info=True)
        _models["ready"] = False

    yield  # ← application runs here

    logger.info("Shutting down HaluExplain engine…")
    _models.clear()


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness check",
    tags=["Meta"],
)
async def health_check() -> HealthResponse:
    """Returns 200 if the service is alive and models are loaded."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        models_loaded=_models.get("ready", False),
    )


@app.post(
    "/verify_claim",
    response_model=VerifyClaimResponse,
    summary="Verify an AI-generated claim for hallucinations",
    tags=["Verification"],
    status_code=status.HTTP_200_OK,
)
async def verify_claim(request: VerifyClaimRequest) -> VerifyClaimResponse:
    """
    Full hallucination detection pipeline.

    1. **Self-Consistency Analysis** — generates 5 diverse LLM responses
       for the claim/prompt and measures inter-response agreement.
    2. **Hybrid Evidence Retrieval** — retrieves Top-K supporting/contradicting
       passages via RRF(SBERT + BM25).
    3. **NLI Entailment** — DeBERTa-v3-large scores each (evidence, claim) pair.
    4. **Explanation Generation** — BART summarises the verdict into prose.

    Returns all intermediate results plus a final human-readable explanation.
    """
    if not _models.get("ready"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading or failed to initialise. Please retry shortly.",
        )

    claim = request.claim.strip()
    top_k = request.top_k or TOP_K_PASSAGES
    start_time = time.perf_counter()

    try:
        # ── Module 1: Self-Consistency ─────────────────────────────────
        logger.info("[1/4] Self-consistency analysis for claim: %r", claim[:80])
        consistency_result = _models["consistency"].analyse(prompt=claim)

        # ── Module 2: Hybrid Evidence Retrieval ────────────────────────
        logger.info("[2/4] Hybrid retrieval (top_k=%d)…", top_k)
        evidence_list = _models["retriever"].retrieve(
            query=claim,
            extra_docs=request.context_docs,
            top_k=top_k,
        )

        # ── Module 3: Entailment / NLI ─────────────────────────────────
        logger.info("[3/4] Entailment verification…")
        entailment_result = _models["entailment"].verify(
            claim=claim,
            evidence_passages=evidence_list,
        )

        # ── Module 4: Explanation Generation ──────────────────────────
        logger.info("[4/4] Generating explanation…")
        explanation_text = _models["explanation"].generate(
            claim=claim,
            entailment=entailment_result,
            evidence_passages=evidence_list,
            consistency=consistency_result,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline error: {exc}",
        ) from exc

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info("Pipeline complete in %.1f ms | verdict=%s", elapsed_ms, entailment_result.verdict)

    return VerifyClaimResponse(
        claim=claim,
        consistency=consistency_result,
        evidence=evidence_list,
        entailment=entailment_result,
        explanation=explanation_text,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dev runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
