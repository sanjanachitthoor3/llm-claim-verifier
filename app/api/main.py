from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from app.modules.claim_extraction.extractor import ClaimExtractor
from app.modules.wikipedia.wiki_fetcher import WikiFetcher
from app.utils.text_preprocessing import split_into_sentences
from app.modules.retrieval.faiss_search import FAISSRetriever
from app.modules.verification.nli_verifier import NLIVerifier
from app.modules.scoring.scorer import compute_score

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LLM Claim Verification System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Request Model --------------------

class VerifyRequest(BaseModel):
    text: str


# -------------------- Response Models --------------------

class ClaimResult(BaseModel):
    claim: str
    verdict: str
    evidence: List[str]


class ScoreResponse(BaseModel):
    trust_score: float
    hallucination_risk: float
    supported: int
    contradicted: int
    not_enough_info: int
    total: int


class VerifyResponse(BaseModel):
    claims: List[ClaimResult]
    score: ScoreResponse


# -------------------- Endpoint --------------------

@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest):

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' field must not be empty.")

    # ✅ Load once (important fix)
    extractor = ClaimExtractor()
    fetcher = WikiFetcher()
    verifier = NLIVerifier()

    claims = extractor.extract_claims(request.text)

    results = []
    all_verdicts = []

    for c in claims:
        article = fetcher.get_article_for_claim(c)

        if article:
            sentences = split_into_sentences(article)

            # FAISS per claim (index changes)
            retriever = FAISSRetriever()
            retriever.build_index(sentences)
            top_k = retriever.search(c, k=3)

            verdict_result = verifier.verify(c, top_k)

            verdict = verdict_result["verdict"]
            evidence = verdict_result["evidence"]
        else:
            verdict = "NOT ENOUGH INFO"
            evidence = []

        results.append(ClaimResult(claim=c, verdict=verdict, evidence=evidence))
        all_verdicts.append(verdict)

    score = compute_score(all_verdicts)

    return VerifyResponse(claims=results, score=score)