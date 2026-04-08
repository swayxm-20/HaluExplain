"""
app/models.py
-------------
Pydantic v2 schemas for all request / response payloads used by the
HaluExplain FastAPI application.

These schemas serve as both validation contracts and auto-generated
OpenAPI documentation objects.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# Shared sub-models
# ══════════════════════════════════════════════════════════════════════════════


class EvidencePassage(BaseModel):
    """A single retrieved passage with its hybrid relevance score."""

    text: str = Field(..., description="The raw evidence text passage.")
    rrf_score: float = Field(
        ..., ge=0.0, description="Reciprocal Rank Fusion score (higher = more relevant)."
    )
    source: Optional[str] = Field(
        None, description="Optional source label / document ID."
    )


class ConsistencyResult(BaseModel):
    """Output of the self-consistency analysis module."""

    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Consistency score: 1.0 = fully consistent, 0.0 = maximal conflict."
    )
    conflicting_statements: List[str] = Field(
        default_factory=list,
        description="Sentences identified as outliers / contradicting the consensus."
    )
    total_sentences: int = Field(
        ..., ge=0,
        description="Total number of sentences analysed across all LLM responses."
    )
    consensus_sentences: int = Field(
        ..., ge=0,
        description="Number of sentences that belong to a consensus cluster."
    )


class EntailmentResult(BaseModel):
    """Output of the NLI entailment engine."""

    verdict: str = Field(
        ...,
        description="One of: ENTAILMENT, CONTRADICTION, NEUTRAL, or UNCERTAIN."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of the winning label (0–1)."
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Full label → probability map: contradiction / neutral / entailment."
    )


# ══════════════════════════════════════════════════════════════════════════════
# API Request & Response
# ══════════════════════════════════════════════════════════════════════════════


class VerifyClaimRequest(BaseModel):
    """Payload for POST /verify_claim."""

    claim: str = Field(
        ...,
        min_length=10,
        description="The AI-generated claim to verify.",
        examples=["The Great Wall of China is visible from space with the naked eye."],
    )
    context_docs: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional additional text passages to include in the knowledge base "
            "alongside the built-in corpus. Useful for domain-specific verification."
        ),
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20,
        description="Override the default number of retrieved evidence passages (default: 5)."
    )


class VerifyClaimResponse(BaseModel):
    """Full JSON response from POST /verify_claim."""

    claim: str = Field(..., description="The original claim that was evaluated.")

    # ── Module 1 output ────────────────────────────────────────────────────
    consistency: ConsistencyResult

    # ── Module 2 output ────────────────────────────────────────────────────
    evidence: List[EvidencePassage] = Field(
        ..., description="Top-K ranked evidence passages from the hybrid retriever."
    )

    # ── Module 3 output ────────────────────────────────────────────────────
    entailment: EntailmentResult

    # ── Module 4 output ────────────────────────────────────────────────────
    explanation: str = Field(
        ..., description="Human-readable natural language explanation of the verdict."
    )

    # ── Meta ───────────────────────────────────────────────────────────────
    processing_time_ms: float = Field(
        ..., ge=0.0, description="Total end-to-end latency in milliseconds."
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: bool
