"""
app/explanation.py
------------------
Explanation Generator
======================
Uses BART-large-cnn to synthesise the original claim, the entailment verdict,
consistency findings, and retrieved evidence into a clear, human-readable
paragraph that follows the prescribed output format.

Output format
-------------
"The claim '[Claim]' is classified as [VERDICT] (confidence: [X]%).
 Evidence [confirms/contradicts/does not clearly address] that [key fact].
 [Consistency note if score < 0.8.]
 Suggested correction: '[correction / N/A if claim is supported].'"

Design notes
------------
* We construct a structured input document and ask BART to summarise it.
  This leverages BART's abstractive capability while giving us control over
  the information that must appear in the output.
* A deterministic template is used as a reliable fallback if the model
  output is shorter than expected (e.g., GPU OOM on low-resource systems).
"""

from __future__ import annotations

import logging
import textwrap
from typing import List

from transformers import pipeline  # type: ignore

from config import (
    EXPLANATION_MAX_LENGTH,
    EXPLANATION_MIN_LENGTH,
    EXPLANATION_MODEL,
    EXPLANATION_NUM_BEAMS,
)
from app.models import ConsistencyResult, EntailmentResult, EvidencePassage

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Verdict → human-readable phrase mappings
# ──────────────────────────────────────────────────────────────────────────────

_VERDICT_PHRASE: dict[str, str] = {
    "ENTAILMENT": "supported by evidence",
    "CONTRADICTION": "contradicted by evidence",
    "NEUTRAL": "not clearly addressed by available evidence",
    "UNCERTAIN": "unverifiable with the available evidence",
}

_EVIDENCE_VERB: dict[str, str] = {
    "ENTAILMENT": "confirms",
    "CONTRADICTION": "contradicts",
    "NEUTRAL": "does not clearly address",
    "UNCERTAIN": "does not conclusively address",
}

_CORRECTION_PREFIX: dict[str, str] = {
    "ENTAILMENT": "No correction needed — the claim is factually supported.",
    "CONTRADICTION": "Consider revising the claim to align with the evidence.",
    "NEUTRAL": "More research may be needed to confirm or deny this claim.",
    "UNCERTAIN": "Insufficient evidence was found; treat this claim with caution.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Explanation generator
# ──────────────────────────────────────────────────────────────────────────────


class ExplanationGenerator:
    """
    BART-based abstractive explanation generator.

    Falls back to a deterministic template when the generated text is
    suspiciously short (< 30 characters) to ensure the API always returns
    something meaningful.

    Parameters
    ----------
    model_name : HuggingFace model identifier (default: facebook/bart-large-cnn).
    """

    def __init__(self, model_name: str = EXPLANATION_MODEL) -> None:
        logger.info("Loading explanation model: %s", model_name)
        self._summariser = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # CPU; set to 0 for GPU
        )
        logger.info("ExplanationGenerator ready.")

    # ------------------------------------------------------------------
    def _build_input_document(
        self,
        claim: str,
        verdict: str,
        confidence: float,
        evidence_passages: List[EvidencePassage],
        consistency: ConsistencyResult,
    ) -> str:
        """
        Construct the document that BART will summarise into an explanation.
        """
        top_evidence = evidence_passages[0].text if evidence_passages else "No evidence retrieved."

        consistency_note = ""
        if consistency.score < 0.8 and consistency.conflicting_statements:
            snippet = consistency.conflicting_statements[0][:120]
            consistency_note = (
                f"Note: Self-consistency analysis detected conflicting statements "
                f"(score={consistency.score:.2f}), for example: \"{snippet}\". "
            )

        doc = textwrap.dedent(f"""
            Claim under evaluation: "{claim}"

            Verdict: {verdict} ({_VERDICT_PHRASE.get(verdict, verdict)}).
            Confidence: {confidence * 100:.1f}%.

            Most relevant evidence: {top_evidence}

            {consistency_note}

            Correction guidance: {_CORRECTION_PREFIX.get(verdict, "")}

            Please summarise the above into a single clear paragraph explaining
            whether the claim is factually accurate and why.
        """).strip()

        return doc

    # ------------------------------------------------------------------
    def generate(
        self,
        claim: str,
        entailment: EntailmentResult,
        evidence_passages: List[EvidencePassage],
        consistency: ConsistencyResult,
    ) -> str:
        """
        Generate a human-readable explanation paragraph.

        Parameters
        ----------
        claim             : Original user claim.
        entailment        : Result from EntailmentEngine.verify().
        evidence_passages : Top-K passages from HybridRetriever.retrieve().
        consistency       : Result from ConsistencyAnalyser.analyse().

        Returns
        -------
        str — formatted explanation paragraph.
        """
        verdict = entailment.verdict
        confidence = entailment.confidence
        confidence_pct = f"{confidence * 100:.1f}"
        top_evidence = evidence_passages[0].text if evidence_passages else "No evidence found."
        evidence_verb = _EVIDENCE_VERB.get(verdict, "does not address")
        correction = _CORRECTION_PREFIX.get(verdict, "")

        # ── Try BART summarisation ─────────────────────────────────────
        try:
            input_doc = self._build_input_document(
                claim, verdict, confidence, evidence_passages, consistency
            )

            result = self._summariser(
                input_doc,
                max_length=EXPLANATION_MAX_LENGTH,
                min_length=EXPLANATION_MIN_LENGTH,
                num_beams=EXPLANATION_NUM_BEAMS,
                early_stopping=True,
                truncation=True,
            )
            generated_text: str = result[0]["summary_text"].strip()  # type: ignore[index]

            if len(generated_text) >= 30:
                # Prepend a templated header to ensure claim & verdict always appear
                header = (
                    f"The claim '{claim}' is classified as {verdict} "
                    f"(confidence: {confidence_pct}%). "
                )
                if not generated_text.startswith("The claim"):
                    generated_text = header + generated_text

                logger.info("Explanation generated via BART (%d chars).", len(generated_text))
                return generated_text

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("BART generation failed: %s — using template fallback.", exc)

        # ── Deterministic template fallback ────────────────────────────
        consistency_note = ""
        if consistency.score < 0.8 and consistency.conflicting_statements:
            consistency_note = (
                f" Self-consistency analysis also flagged potential conflicts "
                f"(score={consistency.score:.2f}), suggesting LLM responses were "
                f"partially inconsistent."
            )

        explanation = (
            f"The claim '{claim}' is classified as {verdict} "
            f"(confidence: {confidence_pct}%). "
            f"Evidence {evidence_verb} that {top_evidence[:200].rstrip('.')}. "
            f"{consistency_note} "
            f"Suggested correction: '{correction}'"
        ).strip()

        logger.info("Explanation generated via template fallback (%d chars).", len(explanation))
        return explanation
