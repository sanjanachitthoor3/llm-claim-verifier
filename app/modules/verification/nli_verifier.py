"""
nli_verifier.py
Module 5 - Claim Verification via Natural Language Inference
LLM Claim Verification System

Responsibility:
    Given a claim and a list of retrieved evidence sentences, determine
    whether the claim is SUPPORTED, CONTRADICTED, or NOT ENOUGH INFO.

Pipeline position:
    faiss_retriever.py  →  nli_verifier.py  →  scoring module
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

_MODEL_NAME = "facebook/bart-large-mnli"

# Verdict constants — used by the scoring module downstream.
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_ENOUGH_INFO = "NOT ENOUGH INFO"


class NLIVerifier:
    """
    Wraps facebook/bart-large-mnli for proper NLI-based claim verification.

    Each evidence sentence is treated as the NLI *premise*; the claim is
    treated as the *hypothesis*.  Per-sentence predictions are aggregated
    into a single verdict using the priority rule:
        entailment (any)  →  SUPPORTED
        contradiction (any, no entailment)  →  CONTRADICTED
        otherwise  →  NOT ENOUGH INFO

    Usage:
        verifier = NLIVerifier()
        result = verifier.verify(claim, evidence_sentences)
    """

    # bart-large-mnli logit order as defined by its label2id config.
    _LABELS = ["contradiction", "neutral", "entailment"]

    def __init__(self):
        """Load the tokenizer and NLI model once at construction time."""
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, claim: str, evidence_sentences: List[str]) -> str:
        """
        Verify a claim against a list of evidence sentences.

        Args:
            claim:              The factual claim to verify (hypothesis).
            evidence_sentences: Retrieved sentences from Wikipedia (premises).

        Returns:
            One of: "SUPPORTED", "CONTRADICTED", "NOT ENOUGH INFO".
            Returns NOT_ENOUGH_INFO immediately if either input is empty.
        """
        if not claim or not claim.strip():
            return NOT_ENOUGH_INFO
        if not evidence_sentences:
            return NOT_ENOUGH_INFO

        labels = [
            self._classify(premise, claim)
            for premise in evidence_sentences
            if premise and premise.strip()
        ]

        return self._aggregate(labels)

    def verify_detailed(
        self, claim: str, evidence_sentences: List[str]
    ) -> dict:
        """
        Return per-sentence predictions alongside the aggregated verdict.

        Useful for the evidence-based output feature (displaying which
        sentence supported or contradicted the claim in the UI).

        Args:
            claim:              The factual claim to verify.
            evidence_sentences: Retrieved sentences from Wikipedia.

        Returns:
            A dict with keys:
                "verdict"  -> str  (SUPPORTED / CONTRADICTED / NOT ENOUGH INFO)
                "details"  -> list of {"sentence": str, "label": str}
        """
        if not claim or not claim.strip() or not evidence_sentences:
            return {"verdict": NOT_ENOUGH_INFO, "details": []}

        details = []
        for sentence in evidence_sentences:
            if not sentence or not sentence.strip():
                continue
            label = self._classify(sentence, claim)
            details.append({"sentence": sentence, "label": label})

        verdict = self._aggregate([d["label"] for d in details])
        return {"verdict": verdict, "details": details}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(self, premise: str, hypothesis: str) -> str:
        """
        Run the NLI model for a single (premise, hypothesis) pair.

        Tokenizes the premise and hypothesis together as a sequence pair,
        runs a forward pass through bart-large-mnli, and returns the
        highest-scoring label.

        bart-large-mnli's label2id maps logit positions as:
            index 0 → contradiction
            index 1 → neutral
            index 2 → entailment

        Args:
            premise:    One evidence sentence (what we know).
            hypothesis: The claim being verified (what we want to test).

        Returns:
            One of: "entailment", "contradiction", "neutral".
        """
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits  # shape: (1, 3)

        predicted_index = logits.argmax(dim=-1).item()
        return self._LABELS[predicted_index]

    @staticmethod
    def _aggregate(labels: List[str]) -> str:
        """
        Collapse per-sentence NLI labels into a single verdict.

        Priority:
            1. Any "entailment"    -> SUPPORTED
            2. Any "contradiction" (and no entailment) -> CONTRADICTED
            3. Otherwise           -> NOT ENOUGH INFO

        Args:
            labels: List of NLI label strings for each evidence sentence.

        Returns:
            A verdict constant string.
        """
        if not labels:
            return NOT_ENOUGH_INFO
        if "entailment" in labels:
            return SUPPORTED
        if "contradiction" in labels:
            return CONTRADICTED
        return NOT_ENOUGH_INFO
