"""
faiss_retriever.py
Module 4 - FAISS-Based Semantic Retrieval
LLM Claim Verification System

Responsibility:
    Build a FAISS index over Wikipedia sentence embeddings and retrieve the
    top-k sentences most semantically similar to a given claim.

Pipeline position:
    embedder.py  →  faiss_retriever.py  →  NLI verification module
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

_MODEL_NAME = "all-MiniLM-L6-v2"


class FAISSRetriever:
    """
    Encapsulates a FAISS flat inner-product index for sentence retrieval.

    Usage pattern:
        retriever = FAISSRetriever()
        retriever.build_index(sentences)
        top_sentences = retriever.search(claim, k=3)
    """

    def __init__(self):
        """Initialise the retriever with a shared sentence-transformer model."""
        self._model = SentenceTransformer(_MODEL_NAME)
        self._index: faiss.IndexFlatIP | None = None
        self._sentences: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, sentences: List[str]) -> None:
        """
        Encode sentences and populate the FAISS index.

        Steps:
            1. Encode all sentences into L2-normalised embedding vectors.
            2. Create a FAISS IndexFlatIP (inner product) index.
            3. Add the normalised vectors to the index.

        After normalisation, inner product == cosine similarity, so the index
        ranks results by semantic closeness without any extra distance math.

        Args:
            sentences: List of plain-text sentences to index.
                       Silently resets any previously built index.
        """
        self._sentences = []
        self._index = None

        if not sentences:
            return

        self._sentences = sentences

        # Encode and normalise.
        embeddings = self._encode_and_normalise(sentences)

        # IndexFlatIP: exact (non-approximate) inner-product search.
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

    def search(self, claim: str, k: int = 3) -> List[str]:
        """
        Retrieve the top-k sentences most semantically similar to the claim.

        Args:
            claim: A factual claim string to search against the index.
            k:     Number of results to return (default: 3).
                   Clamped to the number of indexed sentences if smaller.

        Returns:
            An ordered list of up to k sentences (highest similarity first).
            Returns an empty list if the index has not been built or is empty.
        """
        if self._index is None or not self._sentences or k < 1:
            return []

        # Clamp k so FAISS never requests more rows than exist in the index.
        k = min(k, len(self._sentences))

        # Encode and normalise the claim query vector.
        query = self._encode_and_normalise([claim])  # shape: (1, dim)

        # FAISS search returns (scores, indices) arrays of shape (1, k).
        _, indices = self._index.search(query, k)

        return [self._sentences[i] for i in indices[0] if i != -1]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_and_normalise(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings and L2-normalise each vector in-place.

        Args:
            texts: List of strings to encode.

        Returns:
            A float32 numpy array of shape (len(texts), embedding_dim),
            where each row has unit L2 norm.
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings