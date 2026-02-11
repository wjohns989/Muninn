"""
Muninn BM25 Index
-----------------
In-memory BM25 keyword search for precision matching.
Complements vector search by catching exact terms that
embedding models might miss (e.g., version numbers, UUIDs, paths).
"""

import math
import re
import logging
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

logger = logging.getLogger("Muninn.BM25")

# BM25 tuning constants
K1 = 1.2   # Term frequency saturation
B = 0.75   # Length normalization

# Stopwords for English (minimal set — avoids external deps)
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "was", "are", "be",
    "this", "that", "which", "has", "have", "had", "not", "no", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "i", "you", "he", "she", "we", "they", "me", "my", "your",
    "its", "our", "their", "what", "who", "how", "when", "where", "so",
    "if", "then", "than", "just", "about", "into", "also", "more",
    "been", "being", "some", "any", "all", "each", "very",
}

# Pattern for tokenization — split on non-alphanumeric except hyphens/dots/underscores
TOKENIZE_PATTERN = re.compile(r"[^a-zA-Z0-9_.\-/]+")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase terms, removing stopwords."""
    tokens = TOKENIZE_PATTERN.split(text.lower())
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """
    In-memory BM25 inverted index for fast keyword search.

    Designed for the memory corpus scale (hundreds to low thousands of documents).
    Rebuilds are fast enough to do periodically rather than maintaining
    complex incremental updates.
    """

    def __init__(self):
        # Document store: id → token list
        self._docs: Dict[str, List[str]] = {}
        # Inverted index: term → set of doc ids
        self._inverted: Dict[str, Set[str]] = defaultdict(set)
        # Document frequencies: term → count of docs containing it
        self._df: Dict[str, int] = defaultdict(int)
        # Document lengths
        self._doc_lengths: Dict[str, int] = {}
        # Average document length
        self._avg_dl: float = 0.0
        # Total documents
        self._n: int = 0

    def add(self, doc_id: str, text: str) -> None:
        """Add or update a document in the index."""
        # Remove old version if exists
        if doc_id in self._docs:
            self.remove(doc_id)

        tokens = tokenize(text)
        self._docs[doc_id] = tokens
        self._doc_lengths[doc_id] = len(tokens)

        # Update inverted index and document frequencies
        seen_terms: Set[str] = set()
        for token in tokens:
            self._inverted[token].add(doc_id)
            if token not in seen_terms:
                self._df[token] += 1
                seen_terms.add(token)

        self._n += 1
        self._recompute_avg_dl()

    def remove(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self._docs:
            return

        tokens = self._docs[doc_id]
        seen_terms: Set[str] = set()
        for token in tokens:
            if token in self._inverted:
                self._inverted[token].discard(doc_id)
                if not self._inverted[token]:
                    del self._inverted[token]
            if token not in seen_terms:
                self._df[token] -= 1
                if self._df[token] <= 0:
                    del self._df[token]
                seen_terms.add(token)

        del self._docs[doc_id]
        del self._doc_lengths[doc_id]
        self._n -= 1
        self._recompute_avg_dl()

    def search(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        """
        Search the index using BM25 scoring.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        if self._n == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self._df:
                continue

            df = self._df[term]
            # IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id in self._inverted.get(term, set()):
                # Term frequency in this document
                tf = self._docs[doc_id].count(term)
                dl = self._doc_lengths[doc_id]

                # BM25 term score
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / self._avg_dl)
                scores[doc_id] += idf * numerator / denominator

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

    def clear(self) -> None:
        """Remove all documents from the index."""
        self._docs.clear()
        self._inverted.clear()
        self._df.clear()
        self._doc_lengths.clear()
        self._avg_dl = 0.0
        self._n = 0

    def rebuild(self, documents: Dict[str, str]) -> None:
        """
        Rebuild the entire index from a dict of {id: text}.
        More efficient than individual adds for large batches.
        """
        self.clear()
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            self._docs[doc_id] = tokens
            self._doc_lengths[doc_id] = len(tokens)

            seen_terms: Set[str] = set()
            for token in tokens:
                self._inverted[token].add(doc_id)
                if token not in seen_terms:
                    self._df[token] += 1
                    seen_terms.add(token)

            self._n += 1

        self._recompute_avg_dl()
        logger.info("BM25 index rebuilt: %d documents, %d unique terms",
                     self._n, len(self._df))

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return self._n

    def _recompute_avg_dl(self) -> None:
        """Recompute average document length."""
        if self._n > 0:
            self._avg_dl = sum(self._doc_lengths.values()) / self._n
        else:
            self._avg_dl = 0.0
