"""
app/consistency.py
------------------
Self-Consistency Analysis Module
=================================
Accepts a user prompt and (optionally) a list of pre-generated LLM responses.
If no responses are provided, a built-in mock LLM generates 5 diverse
paraphrases so the module is fully self-contained without an external API key.

Algorithm
---------
1. Tokenise every response into sentences (NLTK punkt tokeniser).
2. Embed all sentences with Sentence-BERT (all-mpnet-base-v2).
3. Compute pairwise cosine distances on unit-normalised embeddings.
4. Run AgglomerativeClustering with the cosine distance matrix.
5. Label clusters:
   - Consensus  → size ≥ CONSENSUS_MIN_CLUSTER_SIZE
   - Outlier    → size == 1  (conflicting / hallucinated statement)
6. consistency_score = consensus_sentence_count / total_sentence_count

Returns
-------
ConsistencyResult: score (0–1), conflicting_statements, totals.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from config import (
    CLUSTERING_DISTANCE_THRESHOLD,
    CONSENSUS_MIN_CLUSTER_SIZE,
    EMBEDDING_MODEL,
    NUM_LLM_RESPONSES,
)
from app.models import ConsistencyResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Mock LLM — used when real LLM responses are not provided
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_RESPONSE_TEMPLATES: List[str] = [
    (
        "{prompt} "
        "This is supported by substantial scientific evidence. "
        "Researchers have consistently observed this phenomenon across multiple studies. "
        "The consensus in the academic community strongly supports this view."
    ),
    (
        "Regarding the topic of {prompt}, "
        "numerous peer-reviewed publications confirm this claim. "
        "The empirical data leaves little room for doubt. "
        "Experts widely agree on this matter."
    ),
    (
        "Studies examining {prompt} have found clear confirmation. "
        "Statistical analyses reveal high confidence in this assertion. "
        "The scientific literature corroborates the claim thoroughly."
    ),
    (
        "While {prompt} is generally accepted, "
        "some minority viewpoints challenge this interpretation. "
        "However, the predominant evidence still supports the original claim. "
        "Contradictory reports remain outliers and lack replication."  # slight outlier
    ),
    (
        "There is absolutely no scientific evidence that {prompt}. "  # intentional contradiction
        "In fact, multiple independent studies have refuted this idea entirely. "
        "The claim is considered a common misconception among experts."
    ),
]


def _generate_mock_responses(prompt: str, n: int = NUM_LLM_RESPONSES) -> List[str]:
    """Generate *n* diverse mock LLM responses by filling prompt templates."""
    short_prompt = prompt.strip().rstrip("?.!")
    return [
        tmpl.format(prompt=short_prompt) for tmpl in _MOCK_RESPONSE_TEMPLATES[:n]
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Sentence tokeniser (lightweight regex fallback if NLTK data is unavailable)
# ──────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split *text* into sentences.

    Tries NLTK punkt tokeniser first; falls back to a simple regex split on
    '. ', '! ', '? ' boundaries so the module works without NLTK data files.
    """
    try:
        import nltk  # type: ignore

        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            logger.info("NLTK punkt data not found; downloading…")
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    except ImportError:
        # Regex fallback
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 10]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


class ConsistencyAnalyser:
    """
    Stateful analyser that loads the embedding model once and reuses it across
    multiple calls — critical for production latency.

    Parameters
    ----------
    model_name : str
        Sentence-Transformers model identifier (default from config).
    distance_threshold : float
        AgglomerativeClustering cut-off in cosine distance space (0–2).
    consensus_min_size : int
        Minimum cluster size to be considered a consensus cluster.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        distance_threshold: float = CLUSTERING_DISTANCE_THRESHOLD,
        consensus_min_size: int = CONSENSUS_MIN_CLUSTER_SIZE,
    ) -> None:
        logger.info("Loading sentence embedding model: %s", model_name)
        self._embedder = SentenceTransformer(model_name)
        self._distance_threshold = distance_threshold
        self._consensus_min_size = consensus_min_size

    # ------------------------------------------------------------------
    def analyse(
        self,
        prompt: str,
        llm_responses: Optional[List[str]] = None,
    ) -> ConsistencyResult:
        """
        Run self-consistency analysis.

        Parameters
        ----------
        prompt : str
            The original user prompt / claim.
        llm_responses : list[str] | None
            Up to NUM_LLM_RESPONSES pre-generated responses.
            If None or empty, mock responses are generated automatically.

        Returns
        -------
        ConsistencyResult
        """
        if not llm_responses:
            logger.info("No LLM responses provided — using mock generator.")
            llm_responses = _generate_mock_responses(prompt)

        # ── 1. Sentence tokenisation ───────────────────────────────────
        all_sentences: List[str] = []
        for response in llm_responses:
            all_sentences.extend(_split_sentences(response))

        total = len(all_sentences)
        if total == 0:
            logger.warning("No sentences extracted from LLM responses.")
            return ConsistencyResult(
                score=1.0,
                conflicting_statements=[],
                total_sentences=0,
                consensus_sentences=0,
            )

        # ── 2. Embed sentences ─────────────────────────────────────────
        logger.debug("Embedding %d sentences…", total)
        embeddings: np.ndarray = self._embedder.encode(
            all_sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        # Unit-normalise so dot-product == cosine similarity
        embeddings = normalize(embeddings, norm="l2")

        # ── 3. Cosine distance matrix ──────────────────────────────────
        # cosine_distance = 1 − cosine_similarity; range [0, 2] for unit vectors
        cosine_sim: np.ndarray = embeddings @ embeddings.T
        distance_matrix: np.ndarray = np.clip(1.0 - cosine_sim, 0.0, 2.0)

        # ── 4. Agglomerative clustering ────────────────────────────────
        if total == 1:
            # Edge case: single sentence — treat as consensus
            return ConsistencyResult(
                score=1.0,
                conflicting_statements=[],
                total_sentences=1,
                consensus_sentences=1,
            )

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=self._distance_threshold,
        )
        labels: np.ndarray = clustering.fit_predict(distance_matrix)

        # ── 5. Identify consensus vs. outlier clusters ─────────────────
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes: dict[int, int] = dict(zip(unique_labels.tolist(), counts.tolist()))

        consensus_sentences: int = 0
        conflicting: List[str] = []

        for idx, label in enumerate(labels):
            if cluster_sizes[label] >= self._consensus_min_size:
                consensus_sentences += 1
            else:
                # Singleton cluster → outlier / conflicting statement
                conflicting.append(all_sentences[idx])

        # ── 6. Consistency score ───────────────────────────────────────
        score: float = consensus_sentences / total if total > 0 else 1.0
        score = float(np.clip(score, 0.0, 1.0))

        logger.info(
            "Consistency analysis done: score=%.3f, total=%d, consensus=%d, outliers=%d",
            score, total, consensus_sentences, len(conflicting),
        )

        return ConsistencyResult(
            score=score,
            conflicting_statements=conflicting,
            total_sentences=total,
            consensus_sentences=consensus_sentences,
        )
