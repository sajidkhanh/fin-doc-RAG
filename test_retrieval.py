"""
test_retrieval.py
Test script to verify all three retrieval strategies work correctly.
Run this after ingestion completes: python test_retrieval.py
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, "src")

from db import get_connection
from sentence_transformers import SentenceTransformer
from retrieval.retriever import FinancialRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def test_retrieval():
    """Test all three retrieval strategies."""
    logger.info("=" * 80)
    logger.info("FinRAG Retrieval Module Test")
    logger.info("=" * 80)

    # Connect to database
    try:
        conn = get_connection()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return

    # Load embedding model
    try:
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load embedding model: {e}")
        conn.close()
        return

    # Initialize retriever
    try:
        retriever = FinancialRetriever(conn, model)
        logger.info("Retriever initialized with BM25 index")
    except Exception as e:
        logger.error(f"✗ Retriever initialization failed: {e}", exc_info=True)
        conn.close()
        return

    # Test query
    query = "What are JPMorgan primary credit risk factors?"
    logger.info(f"\nTest Query: {query}")
    logger.info("=" * 80)

    strategies = ["naive_dense", "hybrid", "reranked"]

    for strategy in strategies:
        logger.info(f"\n[{strategy.upper()}]")
        try:
            results = retriever.retrieve(
                query=query,
                strategy=strategy,
                top_k=3,
                candidate_k=20
            )

            for i, chunk in enumerate(results, 1):
                logger.info(
                    f"  [{i}] {chunk.ticker:4} {chunk.section:20} "
                    f"(score: {chunk.score:.4f}) | method: {chunk.retrieval_method}"
                )
                logger.info(f"      {chunk.content[:100]}...")

        except Exception as e:
            logger.error(f"  ✗ {strategy} failed: {e}")

    conn.close()
    logger.info("\n" + "=" * 80)
    logger.info("Retrieval test complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_retrieval()
