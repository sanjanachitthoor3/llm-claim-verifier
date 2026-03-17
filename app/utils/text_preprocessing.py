"""
text_processing.py
Module 2B - Sentence Processing
LLM Claim Verification System

Responsibility:
    Take a raw Wikipedia article string and return a clean, flat list of
    sentences ready for embedding generation and FAISS indexing.

Pipeline position:
    wiki_fetcher.py  →  text_processing.py  →  embedding module
"""

import re
import nltk
from typing import List

# Ensure the Punkt tokenizer data is available.
# Downloads only if not already present on the system.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def split_into_sentences(text: str) -> List[str]:
    """
    Split a Wikipedia article string into a list of clean sentences.

    Process:
        1. Tokenize the raw text into sentences using NLTK's Punkt tokenizer.
        2. Clean each sentence via clean_sentence().
        3. Discard any sentences that are empty after cleaning.

    Args:
        text: Full Wikipedia article content as a single string.

    Returns:
        A list of clean, non-empty sentence strings.

    Example:
        >>> sentences = split_into_sentences(
        ...     "Albert Einstein (1879) was a physicist. He developed relativity."
        ... )
        >>> sentences
        ['Albert Einstein (1879) was a physicist.', 'He developed relativity.']
    """
    if not text or not text.strip():
        return []

    raw_sentences = nltk.sent_tokenize(text)

    cleaned = [clean_sentence(s) for s in raw_sentences]

    # Drop sentences that are empty after cleaning.
    return [s for s in cleaned if s]


def clean_sentence(sentence: str) -> str:
    """
    Normalise a single sentence string for downstream NLP use.

    Cleaning steps applied (in order):
        1. Replace newline and carriage-return characters with a space.
        2. Collapse runs of whitespace (spaces, tabs) into a single space.
        3. Strip leading and trailing whitespace.

    Important: No words, numbers, punctuation, or factual content are
    removed.  The function is purely a whitespace normaliser so that
    embeddings and NLI models receive consistent input.

    Args:
        sentence: A raw sentence string, possibly containing newlines or
                  irregular spacing from Wikipedia article formatting.

    Returns:
        A normalised sentence string, or an empty string if the input
        contained only whitespace.

    Example:
        >>> clean_sentence("  He was born\\n  in Ulm,  Germany.  ")
        'He was born in Ulm, Germany.'
    """
    # Step 1: Replace newline / carriage-return characters with a space.
    sentence = re.sub(r"[\r\n]+", " ", sentence)

    # Step 2: Collapse multiple spaces or tabs into a single space.
    sentence = re.sub(r"[ \t]+", " ", sentence)

    # Step 3: Strip leading and trailing whitespace.
    sentence = sentence.strip()

    return sentence
