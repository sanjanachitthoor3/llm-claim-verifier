"""
verify.py
Module 7 - FastAPI Endpoint
LLM Claim Verification System

Responsibility:
    Expose the full verification pipeline as a single REST endpoint.
    POST /verify accepts an LLM response, runs it through every module,
    and returns structured claim-level results plus an aggregate score.

Pipeline:
    text  →  ClaimExtractor  →  WikiFetcher  →  split_into_sentences
          →  get_top_k_sentences  →  NLIVerifier  →  compute_score
          →  JSON response
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from app.modules.claim_extraction.extractor import ClaimExtractor
from app.modules.wikipedia.wiki_fetcher import WikiFetcher
from app.utils.text_preprocessing import split_into_sentences
from app.modules.embeddings.embedder import get_top_k_sentences
from app.modules.verification.nli_verifier import NLIVerifier
from app.modules.scoring.scorer import compute_score

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Claim Verification System",
    description="Detects and locates hallucinated claims in LLM responses.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Module initialisation (once at startup — heavy models load here)
# ---------------------------------------------------------------------------

_claim_extractor = ClaimExtractor()
_wiki_fetcher = WikiFetcher()
_nli_verifier = NLIVerifier()

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

TOP_K_SENTENCES = 3          # evidence sentences retrieved per claim
MIN_SENTENCE_LENGTH = 20     # chars; filters out headings and stub sentences

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class VerifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The LLM-generated response to verify.",
        examples=["Albert Einstein was born in 1879 in Ulm, Germany."],
    )


class ClaimResult(BaseModel):
    claim: str
    verdict: str
    evidence: List[str]


class ScoreResult(BaseModel):
    trust_score: float
    hallucination_risk: float
    supported: int
    contradicted: int
    not_enough_info: int
    total: int


class VerifyResponse(BaseModel):
    claims: List[ClaimResult]
    score: ScoreResult

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest) -> VerifyResponse:
    """
    Verify an LLM-generated response for hallucinated claims.

    Steps:
        1. Extract individual factual claims from the input text.
        2. For each claim:
            a. Search and fetch the most relevant Wikipedia article.
            b. Split the article into sentences.
            c. Filter out very short sentences (headings, stubs).
            d. Retrieve the top-k most semantically similar sentences.
            e. Run NLI verification to produce a verdict.
        3. Aggregate all verdicts into a hallucination risk score.
        4. Return per-claim results and the aggregate score.

    Raises:
        HTTPException 422: if the request body fails validation.
        HTTPException 500: if an unexpected error occurs mid-pipeline.
    """
    try:
        # ----------------------------------------------------------------
        # Step 1: Extract claims
        # ----------------------------------------------------------------
        claims: List[str] = _claim_extractor.extract_claims(request.text)

        if not claims:
            return VerifyResponse(
                claims=[],
                score=ScoreResult(**compute_score([])),
            )

        # ----------------------------------------------------------------
        # Step 2: Verify each claim
        # ----------------------------------------------------------------
        claim_results: List[ClaimResult] = []
        verdicts: List[str] = []

        for claim in claims:
            evidence, verdict = _verify_single_claim(claim)
            verdicts.append(verdict)
            claim_results.append(
                ClaimResult(claim=claim, verdict=verdict, evidence=evidence)
            )

        # ----------------------------------------------------------------
        # Step 3: Compute aggregate score
        # ----------------------------------------------------------------
        score_data = compute_score(verdicts)

        return VerifyResponse(
            claims=claim_results,
            score=ScoreResult(**score_data),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Internal pipeline helper
# ---------------------------------------------------------------------------


def _verify_single_claim(claim: str) -> tuple[List[str], str]:
    """
    Run the retrieval and NLI sub-pipeline for one claim.

    Args:
        claim: A single factual claim string.

    Returns:
        A (evidence_sentences, verdict) tuple where:
            evidence_sentences — top-k sentences used for verification.
            verdict            — "SUPPORTED", "CONTRADICTED", or
                                 "NOT ENOUGH INFO".

    Notes:
        Returns ("NOT ENOUGH INFO", []) if Wikipedia retrieval fails,
        so the pipeline continues rather than aborting the whole request.
    """
    # Step 2a: Fetch the Wikipedia article for the most relevant entity.
    article: str | None = _wiki_fetcher.get_article_for_claim(claim)
    if not article:
        return [], "NOT ENOUGH INFO"

    # Step 2b: Split article into sentences.
    sentences: List[str] = split_into_sentences(article)

    # Step 2c: Filter out stubs, headings, and very short fragments.
    sentences = [s for s in sentences if len(s) >= MIN_SENTENCE_LENGTH]

    if not sentences:
        return [], "NOT ENOUGH INFO"

    # Step 2d: Retrieve top-k sentences by semantic similarity to the claim.
    evidence: List[str] = get_top_k_sentences(claim, sentences, k=TOP_K_SENTENCES)

    # Step 2e: Run NLI to determine the verdict.
    verdict: str = _nli_verifier.verify(claim, evidence)

    return evidence, verdict


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Lightweight liveness probe for deployment environments."""
    return {"status": "ok"}