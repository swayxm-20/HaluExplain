"""
app/entailment.py
-----------------
Verification & Entailment Engine
==================================
Uses the DeBERTa-v3-large cross-encoder fine-tuned on NLI to determine
whether a set of retrieved evidence passages *supports*, *contradicts*,
or is *neutral* toward the user's claim.

Strategy
--------
Rather than scoring each passage independently and picking the best, we:
1. Score every (passage, claim) pair independently.
2. Aggregate via *max-pool* on each label dimension so a single strong
   entailment or contradiction anywhere in the evidence dominates the verdict.
   This mimics how a human fact-checker would reason: one authoritative
   source confirming or denying a claim is sufficient.

Label mapping (HuggingFace cross-encoder/nli-deberta-v3-large)
---------------------------------------------------------------
The model was trained on MNLI / SNLI and outputs logits in order:
    [contradiction, neutral, entailment]

Returns
-------
EntailmentResult: verdict, confidence, full probabilities dict.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import CrossEncoder  # type: ignore

from config import NLI_LABEL_ORDER, NLI_MODEL, UNCERTAINTY_THRESHOLD
from app.models import EntailmentResult, EvidencePassage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1-D logit array."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Entailment engine
# ──────────────────────────────────────────────────────────────────────────────


class EntailmentEngine:
    """
    Cross-encoder NLI engine backed by DeBERTa-v3-large.

    Parameters
    ----------
    model_name          : HuggingFace model identifier.
    uncertainty_threshold : If max-probability < threshold, verdict = UNCERTAIN.
    """

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
    ) -> None:
        logger.info("Loading NLI cross-encoder: %s", model_name)
        # num_labels=3 → raw logit output (no built-in softmax)
        self._model = CrossEncoder(model_name, num_labels=3, max_length=512)
        self._threshold = uncertainty_threshold
        logger.info("EntailmentEngine ready.")

    # ------------------------------------------------------------------
    def verify(
        self,
        claim: str,
        evidence_passages: List[EvidencePassage],
    ) -> EntailmentResult:
        """
        Verify *claim* against a list of evidence passages.

        Parameters
        ----------
        claim            : The hypothesis (user's AI-generated claim).
        evidence_passages: Retrieved evidence to use as premises.

        Returns
        -------
        EntailmentResult
        """
        if not evidence_passages:
            logger.warning("No evidence passages supplied; returning UNCERTAIN verdict.")
            return EntailmentResult(
                verdict="UNCERTAIN",
                confidence=0.0,
                probabilities={"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0},
            )

        # ── Build (premise, hypothesis) pairs ─────────────────────────
        # Format recommended by the cross-encoder/nli-* model card
        pairs = [
            (f"Premise: {ep.text}", f"Hypothesis: {claim}")
            for ep in evidence_passages
        ]

        # ── Run inference ──────────────────────────────────────────────
        logger.debug("Running NLI cross-encoder on %d pairs…", len(pairs))
        raw_logits: np.ndarray = self._model.predict(pairs, convert_to_numpy=True)
        # shape: (num_pairs, 3)  [contradiction, neutral, entailment]

        if raw_logits.ndim == 1:
            # Single pair edge case
            raw_logits = raw_logits[None, :]

        # ── Softmax per pair ───────────────────────────────────────────
        probs_per_pair: np.ndarray = np.array([_softmax(row) for row in raw_logits])
        # shape: (num_pairs, 3)

        # ── Max-pool aggregation across all evidence ───────────────────
        # Each label gets the highest probability any passage produces
        aggregated_probs: np.ndarray = probs_per_pair.max(axis=0)  # (3,)

        # ── Derive verdict ─────────────────────────────────────────────
        label_to_prob: dict[str, float] = {
            label: float(aggregated_probs[i])
            for i, label in enumerate(NLI_LABEL_ORDER)
        }

        winning_label: str = NLI_LABEL_ORDER[int(aggregated_probs.argmax())]
        confidence: float = float(aggregated_probs.max())

        if confidence < self._threshold:
            verdict = "UNCERTAIN"
        else:
            verdict = winning_label.upper()

        logger.info(
            "Entailment verdict: %s (confidence=%.3f) | probs=%s",
            verdict, confidence, label_to_prob,
        )

        return EntailmentResult(
            verdict=verdict,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in label_to_prob.items()},
        )
