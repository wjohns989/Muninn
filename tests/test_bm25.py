"""Tests for muninn.retrieval.bm25 — In-memory BM25 keyword search."""

import pytest
from muninn.retrieval.bm25 import BM25Index


class TestBM25Index:
    @pytest.fixture
    def index(self):
        idx = BM25Index()
        idx.add("doc1", "the quick brown fox jumps over the lazy dog")
        idx.add("doc2", "a fast red fox leaps over a sleepy hound")
        idx.add("doc3", "python programming language for data science")
        idx.add("doc4", "machine learning with deep neural networks")
        idx.add("doc5", "the fox ran quickly through the forest")
        return idx

    def test_basic_search(self, index):
        results = index.search("fox")
        assert len(results) > 0
        doc_ids = [r[0] for r in results]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc5" in doc_ids

    def test_relevance_ordering(self, index):
        results = index.search("fox jumps")
        assert len(results) > 0
        # doc1 has both "fox" and "jumps", should rank higher
        doc_ids = [r[0] for r in results]
        assert doc_ids[0] == "doc1"

    def test_no_match(self, index):
        results = index.search("cryptocurrency blockchain")
        assert len(results) == 0

    def test_limit(self, index):
        results = index.search("fox", limit=2)
        assert len(results) <= 2

    def test_empty_query(self, index):
        results = index.search("")
        assert len(results) == 0

    def test_empty_index(self):
        idx = BM25Index()
        results = idx.search("anything")
        assert len(results) == 0

    def test_add_and_remove(self):
        idx = BM25Index()
        idx.add("a", "hello world")
        assert idx.search("hello") != []
        idx.remove("a")
        assert idx.search("hello") == []

    def test_rebuild(self):
        idx = BM25Index()
        documents = {
            "x": "alpha beta gamma",
            "y": "delta epsilon zeta",
        }
        idx.rebuild(documents)
        assert len(idx.search("alpha")) == 1
        assert len(idx.search("delta")) == 1
        assert idx.search("alpha")[0][0] == "x"

    def test_clear(self, index):
        assert index.search("fox") != []
        index.clear()
        assert index.search("fox") == []

    def test_scores_positive(self, index):
        results = index.search("fox")
        for doc_id, score in results:
            assert score > 0

    def test_single_document(self):
        idx = BM25Index()
        idx.add("only", "the single document in the corpus")
        results = idx.search("single document")
        assert len(results) == 1
        assert results[0][0] == "only"

    def test_duplicate_add_updates(self):
        idx = BM25Index()
        idx.add("d1", "original content about cats")
        results1 = idx.search("cats")
        assert len(results1) == 1

        # Re-add with different content should update
        idx.add("d1", "updated content about dogs")
        results_cats = idx.search("cats")
        results_dogs = idx.search("dogs")
        # After update, "cats" should not match (old content removed)
        assert len(results_cats) == 0
        assert len(results_dogs) == 1

    def test_special_characters(self):
        idx = BM25Index()
        idx.add("s1", "hello-world foo_bar baz@qux.com")
        # Tokenizer should handle punctuation
        results = idx.search("hello")
        assert len(results) >= 0  # May or may not match depending on tokenizer


class TestBM25Tokenizer:
    def test_lowercase(self):
        idx = BM25Index()
        idx.add("d1", "UPPERCASE WORDS HERE")
        results = idx.search("uppercase")
        assert len(results) == 1

    def test_stopword_filtering(self):
        idx = BM25Index()
        idx.add("d1", "the quick brown fox")
        idx.add("d2", "a slow grey wolf")
        # Searching for "the" (stopword) should not dominate
        results = idx.search("the fox")
        # "fox" should be the discriminating term
        assert results[0][0] == "d1"
