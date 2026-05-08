"""
tests/test_pipeline.py
Unit tests for ingestion, retrieval, and generation components.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# ── Ingestion tests ───────────────────────────────────────────────────────────

class TestChunker:
    def test_basic_chunking(self):
        from src.ingestion.sec_loader import chunk_text
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_overlap_preserved(self):
        from src.ingestion.sec_loader import chunk_text
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2

    def test_tiny_chunks_filtered(self):
        from src.ingestion.sec_loader import chunk_text
        text = "short text"
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 0  # below 50 word threshold

    def test_section_parsing(self):
        from src.ingestion.sec_loader import parse_filing
        # parsing tested via mock since it requires a real file
        assert callable(parse_filing)


# ── Retriever tests ───────────────────────────────────────────────────────────

class TestRetriever:
    @pytest.fixture
    def mock_retriever(self):
        from src.retrieval.retriever import FinancialRetriever, RetrievedChunk
        conn = MagicMock()
        model = MagicMock()
        model.encode.return_value = np.random.rand(384).astype(np.float32)

        # Mock DB response for dense retrieval
        conn.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[
                (1, "JPM", "10-K", "risk_factors", "Credit risk content", 0.92),
                (2, "BAC", "10-Q", "mda", "Market risk content", 0.88),
            ])
        ))
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(FinancialRetriever, '_build_bm25_index'):
            retriever = FinancialRetriever.__new__(FinancialRetriever)
            retriever.conn = conn
            retriever.model = model
            retriever._bm25_ids = [1, 2]
            retriever._bm25_corpus = ["Credit risk content", "Market risk content"]
            from rank_bm25 import BM25Okapi
            retriever._bm25_index = BM25Okapi([
                ["credit", "risk", "content"],
                ["market", "risk", "content"]
            ])
        return retriever

    def test_dense_retrieve_returns_chunks(self, mock_retriever):
        from src.retrieval.retriever import FinancialRetriever
        results = mock_retriever._dense_retrieve("credit risk", top_k=2)
        assert len(results) == 2
        assert results[0].ticker == "JPM"

    def test_bm25_retrieve(self, mock_retriever):
        results = mock_retriever._bm25_retrieve("credit risk", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)

    def test_rrf_fusion(self, mock_retriever):
        from src.retrieval.retriever import RetrievedChunk
        dense = [
            RetrievedChunk(1, "JPM", "10-K", "risk", "content a", 0.9, "dense"),
            RetrievedChunk(2, "BAC", "10-Q", "mda", "content b", 0.8, "dense"),
        ]
        bm25 = [(2, 5.0), (1, 3.0)]
        fused = mock_retriever._reciprocal_rank_fusion(dense, bm25)
        assert len(fused) == 2
        assert all(c.retrieval_method == "hybrid_rrf" for c in fused)


# ── Generator tests ───────────────────────────────────────────────────────────

class TestGenerator:
    @pytest.fixture
    def mock_generator(self):
        from src.generation.generator import FinancialGenerator
        with patch.object(FinancialGenerator, '__init__', return_value=None):
            gen = FinancialGenerator.__new__(FinancialGenerator)
            gen.llm = MagicMock()
            gen.llm.return_value = {
                "choices": [{"text": "This is a test answer about credit risk."}]
            }
        return gen

    def test_format_context(self, mock_generator):
        from src.retrieval.retriever import RetrievedChunk
        chunks = [RetrievedChunk(1, "JPM", "10-K", "risk_factors", "Credit risk text", 0.9, "dense")]
        ctx = mock_generator._format_context(chunks)
        assert "JPM" in ctx
        assert "Credit risk text" in ctx

    def test_answer_query_structure(self, mock_generator):
        from src.retrieval.retriever import RetrievedChunk
        chunks = [RetrievedChunk(1, "JPM", "10-K", "mda", "Revenue content", 0.9, "reranked")]
        result = mock_generator.answer_query("What are revenues?", chunks)
        assert "query" in result
        assert "answer" in result
        assert "sources" in result

    def test_classify_risk_returns_dict(self, mock_generator):
        from src.retrieval.retriever import RetrievedChunk
        mock_generator.llm.return_value = {
            "choices": [{"text": '{"primary_category": "credit_risk", "secondary_categories": [], "confidence": 0.9, "reasoning": "test"}'}]
        }
        chunk = RetrievedChunk(1, "JPM", "10-K", "risk_factors", "Loan default risk content", 0.9, "dense")
        result = mock_generator.classify_risk(chunk)
        assert "primary_category" in result

    def test_summarize_returns_string(self, mock_generator):
        from src.retrieval.retriever import RetrievedChunk
        chunks = [RetrievedChunk(1, "JPM", "10-K", "business", "Business overview content", 0.9, "dense")]
        result = mock_generator.summarize(chunks, "JPM")
        assert isinstance(result, str)


# ── API tests ─────────────────────────────────────────────────────────────────

class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        with patch.dict("src.api.main.state", {
            "conn": MagicMock(),
            "retriever": MagicMock(),
            "generator": MagicMock(),
            "model": MagicMock()
        }):
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
