"""
config.py
---------
Central configuration for the HaluExplain engine.
All hyperparameters, model names, and thresholds live here
so the rest of the codebase stays clean and easy to tune.
"""

from __future__ import annotations


# ──────────────────────────────────────────────
# Model identifiers
# ──────────────────────────────────────────────

# Sentence-BERT for dense embeddings (consistency + retrieval)
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

# NLI cross-encoder for entailment verification
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-large"

# BART model for abstractive explanation generation
EXPLANATION_MODEL: str = "facebook/bart-large-cnn"


# ──────────────────────────────────────────────
# Consistency module
# ──────────────────────────────────────────────

# Number of mock LLM responses to generate / accept
NUM_LLM_RESPONSES: int = 5

# AgglomerativeClustering distance threshold (cosine space, 0–2)
# Lower = tighter clusters (more sensitive to contradictions)
CLUSTERING_DISTANCE_THRESHOLD: float = 0.4

# A cluster is "consensus" if it contains at least this many sentences
CONSENSUS_MIN_CLUSTER_SIZE: int = 2


# ──────────────────────────────────────────────
# Retrieval module
# ──────────────────────────────────────────────

# Number of top passages returned by the hybrid retriever
TOP_K_PASSAGES: int = 5

# RRF constant — typical value is 60 (Robertson et al., 2009)
RRF_K: int = 60

# Weight of dense retriever score in weighted fusion (alternative to RRF)
DENSE_WEIGHT: float = 0.6
SPARSE_WEIGHT: float = 0.4

# Path to the knowledge-base JSON file
KNOWLEDGE_BASE_PATH: str = "data/knowledge_base.json"

# External search configuration
ENABLE_EXTERNAL_SEARCH: bool = True
TAVILY_API_KEY: str = None  # Set this in environment or override in code


# ──────────────────────────────────────────────
# Entailment module
# ──────────────────────────────────────────────

# Logit order for DeBERTa-v3-large NLI labels
# HuggingFace cross-encoder returns [contradiction, neutral, entailment]
NLI_LABEL_ORDER: list[str] = ["contradiction", "neutral", "entailment"]

# Confidence threshold below which we report the verdict as UNCERTAIN
UNCERTAINTY_THRESHOLD: float = 0.50


# ──────────────────────────────────────────────
# Explanation module
# ──────────────────────────────────────────────

EXPLANATION_MAX_LENGTH: int = 220
EXPLANATION_MIN_LENGTH: int = 60
EXPLANATION_NUM_BEAMS: int = 4


# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────

API_TITLE: str = "HaluExplain — Hallucination Detection Engine"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "Detects AI hallucinations using self-consistency analysis, "
    "hybrid evidence retrieval, DeBERTa NLI entailment, and BART explanation."
)
