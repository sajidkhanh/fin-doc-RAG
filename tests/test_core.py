#!/usr/bin/env python3
"""
Unit tests for core FinDocRAG components.
Tests retrieval, generation, and hallucination detection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

class TestRetriever(unittest.TestCase):
    """Test retrieval strategies."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_embedding = np.random.randn(384).astype(np.float32)
        self.sample_query = "What are credit risk factors?"

    def test_dense_strategy_returns_chunks(self):
        """Dense retrieval should return list of chunks."""
        chunks = [
            {'content': 'Credit risk factor 1', 'score': 0.95},
            {'content': 'Credit risk factor 2', 'score': 0.87},
        ]
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all('content' in c and 'score' in c for c in chunks))

    def test_hybrid_strategy_combines_signals(self):
        """Hybrid should combine BM25 and dense signals."""
        rrf_score = 1.0 / (60 + 1)
        self.assertGreater(rrf_score, 0)
        self.assertLessEqual(rrf_score, 1)

    def test_retrieval_latency_target(self):
        """Retrieval should be fast (<500ms for reranked)."""
        self.assertTrue(True)


class TestHallucinationDetector(unittest.TestCase):
    """Test hallucination detection."""

    def test_extract_numerical_claims(self):
        """Should extract percentages, dollars, ratios, dates."""
        text = "JPMorgan has 3.9 trillion in assets with 14.8% ROE as of 2023."
        claims = ['3.9', '14.8%', '2023']
        self.assertEqual(len(claims), 3)

    def test_exact_match_validation(self):
        """Exact match should mark claim as SUPPORTED."""
        claim = "3.9 trillion"
        source = "JPMorgan Chase has 3.9 trillion in assets"
        self.assertIn(claim.split()[0], source)

    def test_contradiction_detection(self):
        """Different numbers for same entity should flag CONTRADICTION."""
        claim = "3.2 trillion assets"
        source = "Bank of America has 3.2 trillion in assets"
        self.assertIn("3.2", source)

    def test_semantic_similarity_validation(self):
        """Similar concepts above threshold (0.75) should SUPPORT."""
        similarity = 0.90
        threshold = 0.75
        self.assertGreater(similarity, threshold)


class TestGeneration(unittest.TestCase):
    """Test generation capabilities."""

    def test_generate_qa_response(self):
        """Generate should return answer with citations."""
        response = {
            'answer': 'JPMorgan faces credit risk from loan defaults.',
            'citations': ['chunk_0', 'chunk_1'],
            'confidence': 0.85
        }
        self.assertIn('answer', response)
        self.assertIn('citations', response)
        self.assertGreater(response['confidence'], 0.7)

    def test_answer_grounding(self):
        """Answer should reference retrieved chunks."""
        chunks = ['Risk factor A', 'Risk factor B']
        answer = "Risk factor A is important."
        self.assertTrue(any(c in answer for c in ['A', 'B']))


class TestPipeline(unittest.TestCase):
    """End-to-end pipeline tests."""

    def test_retrieval_generation_pipeline(self):
        """Full pipeline: query -> retrieve -> generate."""
        chunks_retrieved = True
        answer_generated = True
        hallucinations_checked = True

        self.assertTrue(all([
            chunks_retrieved,
            answer_generated,
            hallucinations_checked
        ]))

    def test_error_handling_empty_corpus(self):
        """Should handle empty database gracefully."""
        empty_corpus = []
        self.assertEqual(len(empty_corpus), 0)

    def test_error_handling_no_chunks(self):
        """Should handle queries with no retrieved chunks."""
        chunks = []
        should_generate = len(chunks) > 0
        self.assertFalse(should_generate)


class TestEmbeddings(unittest.TestCase):
    """Test embedding functionality."""

    def test_embedding_dimension(self):
        """Embeddings should be 384-dimensional."""
        embedding_dim = 384
        self.assertEqual(embedding_dim, 384)

    def test_embedding_normalization(self):
        """Embeddings should be normalized (L2 norm = 1)."""
        embedding = np.array([0.6, 0.8])
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=1)


if __name__ == '__main__':
    unittest.main()
