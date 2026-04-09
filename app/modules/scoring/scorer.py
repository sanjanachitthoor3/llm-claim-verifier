"""
scorer.py
Module 6 - Hallucination Risk Scoring
LLM Claim Verification System

Responsibility:
    Given a list of per-claim NLI verdicts, compute a trust score and a
    hallucination risk score for the full LLM response.

Pipeline position:
    nli_verifier.py  →  scorer.py  →  FastAPI response
"""

from typing import List

# Verdict constants — must match those defined in nli_verifier.py.
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_ENOUGH_INFO = "NOT ENOUGH INFO"


def compute_score(verdicts: List[str]) -> dict:
    """
    Compute a hallucination risk score from a list of claim verdicts.

    Scoring formula (from the v1 spec):
        trust_score        = supported_claims / total_claims
        hallucination_risk = 1 - trust_score

    A trust_score of 1.0 means every verifiable claim was supported by
    Wikipedia evidence.  A score of 0.0 means no claim was supported.

    Args:
        verdicts: List of verdict strings, each one of:
                  "SUPPORTED", "CONTRADICTED", or "NOT ENOUGH INFO".

    Returns:
        A dict with keys:
            "trust_score"        (float) proportion of supported claims [0.0, 1.0]
            "hallucination_risk" (float) 1 - trust_score               [0.0, 1.0]
            "supported"          (int)   count of SUPPORTED verdicts
            "contradicted"       (int)   count of CONTRADICTED verdicts
            "not_enough_info"    (int)   count of NOT ENOUGH INFO verdicts
            "total"              (int)   total number of claims evaluated

        If verdicts is empty, all numeric fields are 0 and both scores
        are 0.0 to signal that no evaluation was possible.
    """
    if not verdicts:
        return _build_result(
            supported=0,
            contradicted=0,
            not_enough_info=0,
            total=0,
            trust_score=0.0,
            hallucination_risk=0.0,
        )

    supported = verdicts.count(SUPPORTED)
    contradicted = verdicts.count(CONTRADICTED)
    not_enough_info = verdicts.count(NOT_ENOUGH_INFO)
    total = len(verdicts)

    trust_score = supported / total
    hallucination_risk = 1.0 - trust_score

    return _build_result(
        supported=supported,
        contradicted=contradicted,
        not_enough_info=not_enough_info,
        total=total,
        trust_score=round(trust_score, 4),
        hallucination_risk=round(hallucination_risk, 4),
    )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _build_result(
    supported: int,
    contradicted: int,
    not_enough_info: int,
    total: int,
    trust_score: float,
    hallucination_risk: float,
) -> dict:
    """
    Assemble the standardised result dict returned to the FastAPI layer.

    Keeping construction in one place ensures the response schema never
    drifts between the happy path and the empty-input guard.

    Args:
        supported:          Count of SUPPORTED verdicts.
        contradicted:       Count of CONTRADICTED verdicts.
        not_enough_info:    Count of NOT ENOUGH INFO verdicts.
        total:              Total number of claims evaluated.
        trust_score:        Proportion of supported claims (0.0–1.0).
        hallucination_risk: 1 - trust_score (0.0–1.0).

    Returns:
        Standardised scoring result dict.
    """
    return {
        "trust_score": trust_score,
        "hallucination_risk": hallucination_risk,
        "supported": supported,
        "contradicted": contradicted,
        "not_enough_info": not_enough_info,
        "total": total,
    }