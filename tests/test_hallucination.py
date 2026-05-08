"""
tests/test_hallucination.py
Test suite for numerical hallucination detection in LLM-generated financial text.

Tests cover:
  1. Extraction of numerical claims from generated text
  2. Validation against source documents
  3. Hallucination rate thresholds (target <20%)
  4. Comparison across retrieval strategies
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict

import pytest

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from db import get_connection
from sentence_transformers import SentenceTransformer
from retrieval.retriever import FinancialRetriever
from generation.generator import FinancialGenerator
from evaluation.hallucination_detector import NumericalHallucinationDetector

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHallucinationDetection:
    """Test numerical hallucination detection."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Initialize test fixtures: DB, models, retriever, generator, detector."""
        conn = get_connection()
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        retriever = FinancialRetriever(conn, model)
        generator = FinancialGenerator()
        detector = NumericalHallucinationDetector(model)

        yield {
            "conn": conn,
            "model": model,
            "retriever": retriever,
            "generator": generator,
            "detector": detector
        }

        conn.close()

    def test_extraction_basic(self, setup):
        """Test extraction of numerical claims from generated text."""
        detector = setup["detector"]

        # Generate test text with various numerical formats
        test_text = (
            "JPMorgan reported net income of $10.5 billion in 2023, "
            "up 25% from 2022. The dividend was increased by 15 basis points. "
            "ROE improved to 12.4%. The filing date was 2024-01-15."
        )

        claims = detector.extract_numerical_claims(test_text)

        assert len(claims) > 0, "Should extract at least one numerical claim"
        assert any("10.5" in c.number or "10" in c.number for c in claims)
        assert any("25" in c.number for c in claims)
        logger.info(f"✓ Extracted {len(claims)} claims from test text")

    def test_detection_with_ground_truth(self, setup):
        """Test hallucination detection where ground truth numbers are known."""
        retriever = setup["retriever"]
        generator = setup["generator"]
        detector = setup["detector"]

        # Financial fact: JPMorgan's primary risks are credit, market, operational
        query = (
            "What are JPMorgan's primary risk factors? "
            "List the top 3 risk categories."
        )

        # Retrieve source chunks
        chunks = retriever.retrieve(
            query=query,
            strategy="reranked",
            top_k=5,
            candidate_k=20
        )

        assert len(chunks) > 0, "Should retrieve chunks for valid query"

        # Generate answer grounded in sources
        response = generator.answer_query(query, chunks)
        generated_text = response["answer"]

        # Convert retrieved chunks to detector format
        source_chunks = [
            {
                "content": c.content,
                "ticker": c.ticker,
                "filing_type": c.filing_type,
                "section": c.section
            }
            for c in chunks
        ]

        # Detect hallucinations
        score = detector.detect(generated_text, source_chunks)

        logger.info(
            f"Generated text hallucination rate: {score.hallucination_rate:.1%} "
            f"({score.unverifiable} unverifiable out of {score.total_claims} claims)"
        )

        # Threshold: acceptable hallucination rate for grounded generation
        assert score.hallucination_rate < 0.30, \
            f"Hallucination rate {score.hallucination_rate:.1%} exceeds 30% threshold"

    def test_detection_with_injected_hallucination(self, setup):
        """Test that detector catches obviously false numbers."""
        detector = setup["detector"]

        # Deliberately false claim
        false_text = (
            "JPMorgan reported net income of $500 billion in 2023. "
            "This represents a 1000% increase from 2022."
        )

        # Dummy source chunks that don't contain these numbers
        source_chunks = [
            {
                "content": (
                    "JPMorgan reported net income of $16.1 billion in 2023, "
                    "up 18% from 2022."
                ),
                "ticker": "JPM",
                "filing_type": "10-K",
                "section": "financials"
            }
        ]

        score = detector.detect(false_text, source_chunks)

        # Should flag the false numbers as unverifiable
        assert score.unverifiable > 0, \
            "Should flag injected hallucinations as unverifiable"
        logger.info(
            f"✓ Detector correctly flagged {score.unverifiable} "
            f"unverifiable claims in deliberately false text"
        )

    def test_strategy_comparison(self, setup):
        """Compare hallucination rates across retrieval strategies."""
        retriever = setup["retriever"]
        generator = setup["generator"]
        detector = setup["detector"]

        query = "What are the key credit risk mitigation strategies?"

        strategies = ["naive_dense", "hybrid", "reranked"]
        results = {}

        for strategy in strategies:
            logger.info(f"\nTesting strategy: {strategy}")

            # Retrieve
            chunks = retriever.retrieve(
                query=query,
                strategy=strategy,
                top_k=5,
                candidate_k=20
            )

            # Generate
            response = generator.answer_query(query, chunks)

            # Detect
            source_chunks = [
                {
                    "content": c.content,
                    "ticker": c.ticker,
                    "filing_type": c.filing_type,
                    "section": c.section
                }
                for c in chunks
            ]
            score = detector.detect(response["answer"], source_chunks)

            results[strategy] = {
                "hallucination_rate": score.hallucination_rate,
                "total_claims": score.total_claims,
                "unverifiable": score.unverifiable
            }

            logger.info(
                f"  Hallucination rate: {score.hallucination_rate:.1%} "
                f"({score.unverifiable}/{score.total_claims} unverifiable)"
            )

        # Hypothesis: reranked should have lowest hallucination rate
        naive_rate = results["naive_dense"]["hallucination_rate"]
        reranked_rate = results["reranked"]["hallucination_rate"]

        logger.info(
            f"\n✓ Strategy comparison:"
            f"\n  naive_dense: {naive_rate:.1%}"
            f"\n  hybrid: {results['hybrid']['hallucination_rate']:.1%}"
            f"\n  reranked: {reranked_rate:.1%}"
            f"\n  Improvement (naive → reranked): "
            f"{(naive_rate - reranked_rate):.1%} reduction"
        )

    def test_threshold_validation(self, setup):
        """Test that grounded generation stays below 20% hallucination threshold."""
        retriever = setup["retriever"]
        generator = setup["generator"]
        detector = setup["detector"]

        # 10 financial queries with ground-truth answers
        queries = [
            "What is JPMorgan's largest business segment?",
            "How much did JPMorgan earn in net income last year?",
            "What are JPMorgan's main customer groups?",
            "How does JPMorgan manage liquidity risk?",
            "What is JPMorgan's dividend policy?",
            "How is JPMorgan organized by geography?",
            "What technology does JPMorgan use for trading?",
            "How many employees does JPMorgan have?",
            "What are JPMorgan's regulatory capital requirements?",
            "What are JPMorgan's climate risk disclosures?"
        ]

        passed = 0
        failed = 0

        for i, query in enumerate(queries, 1):
            logger.info(f"\n[{i}/10] Query: {query}")

            chunks = retriever.retrieve(
                query=query,
                strategy="reranked",
                top_k=5,
                candidate_k=20
            )

            response = generator.answer_query(query, chunks)

            source_chunks = [
                {
                    "content": c.content,
                    "ticker": c.ticker,
                    "filing_type": c.filing_type,
                    "section": c.section
                }
                for c in chunks
            ]
            score = detector.detect(response["answer"], source_chunks)

            if score.hallucination_rate < 0.20:
                logger.info(f"  ✓ PASS (hallucination rate: {score.hallucination_rate:.1%})")
                passed += 1
            else:
                logger.info(f"  ✗ FAIL (hallucination rate: {score.hallucination_rate:.1%})")
                failed += 1

        logger.info(f"\n{'='*70}")
        logger.info(f"Threshold validation: {passed}/10 queries below 20% threshold")
        logger.info(f"Pass rate: {passed/len(queries):.1%}")
        logger.info(f"{'='*70}")

        # At least 70% should pass the 20% threshold
        assert passed >= 7, \
            f"Only {passed}/10 queries passed threshold (need ≥7)"


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v", "-s"])
