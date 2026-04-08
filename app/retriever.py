"""
app/retriever.py
----------------
Hybrid Evidence Retrieval Module
==================================
Combines a **dense** (Sentence-BERT) retriever with a **sparse** (BM25Okapi)
retriever and merges their ranked lists via **Reciprocal Rank Fusion (RRF)**.

Knowledge Base
--------------
Passages are loaded from ``data/knowledge_base.json`` at initialisation.
Optional caller-supplied documents are appended on each query so the
retriever can accept domain-specific context at request time.

Retrieval Pipeline
------------------
1. Dense  : embed query → cosine similarity → rank passages
2. Sparse : tokenise query → BM25Okapi scores → rank passages
3. Fusion : RRF(dense_rank, sparse_rank) → unified score → Top-K

Returns
-------
List[EvidencePassage] sorted by descending RRF score.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from config import (
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
    RRF_K,
    SPARSE_WEIGHT,
    TOP_K_PASSAGES,
)
from app.models import EvidencePassage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_knowledge_base(path: str) -> List[dict]:
    """
    Load passages from a JSON file.

    Expected format::

        [
            {"text": "...", "source": "optional label"},
            ...
        ]

    Falls back to an empty list + warning if the file is missing.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        logger.warning("Knowledge base not found at '%s'. Retriever will use only caller-supplied docs.", abs_path)
        return []
    with open(abs_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d passages from knowledge base.", len(data))
    return data  # type: ignore[return-value]


def _reciprocal_rank_fusion(
    dense_ranking: List[int],
    sparse_ranking: List[int],
    k: int = RRF_K,
) -> dict[int, float]:
    """
    Compute RRF scores for each document index.

    RRF(d) = Σ_{r ∈ rankings} 1 / (k + rank(d, r))

    Parameters
    ----------
    dense_ranking  : passage indices sorted by dense score (best first)
    sparse_ranking : passage indices sorted by sparse score (best first)
    k              : smoothing constant (default 60)

    Returns
    -------
    dict mapping passage index → RRF score
    """
    rrf_scores: dict[int, float] = {}

    for rank, idx in enumerate(dense_ranking, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    for rank, idx in enumerate(sparse_ranking, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    return rrf_scores


# ──────────────────────────────────────────────────────────────────────────────
# Main retriever class
# ──────────────────────────────────────────────────────────────────────────────


class HybridRetriever:
    """
    Stateful hybrid retriever.  The knowledge-base embeddings and BM25 index
    are pre-computed at construction time; per-query cost is just one
    forward-pass through the encoder + BM25 scoring.

    Parameters
    ----------
    model_name          : Sentence-Transformers model for dense retrieval.
    knowledge_base_path : Path to the JSON knowledge-base file.
    top_k               : Default number of passages to return.
    rrf_k               : RRF constant.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
        top_k: int = TOP_K_PASSAGES,
        rrf_k: int = RRF_K,
    ) -> None:
        logger.info("Initialising HybridRetriever…")
        self._embedder = SentenceTransformer(model_name)
        self._top_k = top_k
        self._rrf_k = rrf_k

        # Load and index the static knowledge base
        raw_docs = _load_knowledge_base(knowledge_base_path)
        self._base_texts: List[str] = [d["text"] for d in raw_docs]
        self._base_sources: List[Optional[str]] = [d.get("source") for d in raw_docs]

        # Pre-compute dense embeddings for the static KB
        if self._base_texts:
            logger.info("Pre-computing dense embeddings for %d KB passages…", len(self._base_texts))
            self._base_embeddings: np.ndarray = normalize(
                self._embedder.encode(self._base_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True),
                norm="l2",
            )
        else:
            self._base_embeddings = np.empty((0, 768))

        logger.info("HybridRetriever ready.")

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        extra_docs: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[EvidencePassage]:
        """
        Retrieve Top-K evidence passages for *query*.

        Parameters
        ----------
        query      : The claim / question to retrieve evidence for.
        extra_docs : Additional passages to merge into the search corpus
                     for this request only (request-scoped documents).
        top_k      : Override default Top-K.

        Returns
        -------
        List[EvidencePassage] sorted by descending RRF score.
        """
        k = top_k if top_k is not None else self._top_k

        # ── Merge static KB with request-scoped docs ───────────────────
        texts: List[str] = list(self._base_texts)
        sources: List[Optional[str]] = list(self._base_sources)

        if extra_docs:
            texts.extend(extra_docs)
            sources.extend([None] * len(extra_docs))

        if not texts:
            logger.warning("No passages available for retrieval.")
            return []

        # ── Dense embeddings for the full corpus ───────────────────────
        if extra_docs:
            extra_embs: np.ndarray = normalize(
                self._embedder.encode(extra_docs, batch_size=32, show_progress_bar=False, convert_to_numpy=True),
                norm="l2",
            )
            corpus_embeddings: np.ndarray = np.vstack([self._base_embeddings, extra_embs])
        else:
            corpus_embeddings = self._base_embeddings

        # ── Query embedding ────────────────────────────────────────────
        query_emb: np.ndarray = normalize(
            self._embedder.encode([query], show_progress_bar=False, convert_to_numpy=True),
            norm="l2",
        )  # shape (1, D)

        # ── Dense ranking (cosine similarity, descending) ──────────────
        dense_scores: np.ndarray = (corpus_embeddings @ query_emb.T).squeeze()  # (N,)
        dense_ranking: List[int] = np.argsort(-dense_scores).tolist()

        # ── Sparse BM25 ranking ────────────────────────────────────────
        tokenised_corpus = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenised_corpus)
        sparse_scores: np.ndarray = bm25.get_scores(query.lower().split())
        sparse_ranking: List[int] = np.argsort(-sparse_scores).tolist()

        # ── RRF fusion ─────────────────────────────────────────────────
        rrf_scores = _reciprocal_rank_fusion(dense_ranking, sparse_ranking, k=self._rrf_k)

        # Sort by RRF score descending, take Top-K
        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:k]

        results: List[EvidencePassage] = [
            EvidencePassage(
                text=texts[i],
                rrf_score=round(rrf_scores[i], 6),
                source=sources[i],
            )
            for i in sorted_indices
        ]

        logger.info("Retrieval complete: returned %d/%d passages.", len(results), k)
        return results
