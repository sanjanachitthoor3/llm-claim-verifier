#module 3: embedding
# claim + many sentences from wikipedia -> embed to capture meaning -> return top-k relevant sentences

'''
INPUT:
claim (string)
sentences (list)

PROCESS:
1. Encode claim → vector
2. Encode sentences → vectors
3. Compute similarity scores
4. Sort
5. Pick top-k

OUTPUT:
[top_k_sentences]
'''

from sentence_transformers import SentenceTransformer, util
from typing import List

# Load model once at module level to avoid reloading on every call.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_MODEL_NAME)


def get_top_k_sentences(claim: str, sentences: List[str], k: int = 3) -> List[str]:
    """
    Return the top-k sentences most semantically similar to the claim.

    Args:
        claim:     A factual claim string to verify.
        sentences: Candidate sentences from a Wikipedia article.
        k:         Number of top results to return (default: 3).

    Returns:
        A list of up to k sentences, ranked by cosine similarity (descending).
        Returns an empty list if sentences is empty or k < 1.
    """
    if not sentences or k < 1:
        return []

    # Step 1: Encode the claim into a single embedding vector.
    claim_embedding = _model.encode(claim, convert_to_tensor=True)

    # Step 2: Encode all candidate sentences into embedding vectors.
    sentence_embeddings = _model.encode(sentences, convert_to_tensor=True)

    # Step 3: Compute cosine similarity between claim and each sentence.
    cosine_scores = util.cos_sim(claim_embedding, sentence_embeddings)[0]

    # Step 4: Rank by similarity score and return the top-k sentences.
    top_k_indices = cosine_scores.argsort(descending=True)[:k]
    return [sentences[i] for i in top_k_indices]