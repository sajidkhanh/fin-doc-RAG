"""
test_generation.py
Test script to verify end-to-end retrieval + generation pipeline.
Run this after ingestion and retrieval tests: python test_generation.py
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, "src")

from db import get_connection
from sentence_transformers import SentenceTransformer
from retrieval.retriever import FinancialRetriever
from generation.generator import FinancialGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def test_generation():
    """Test retrieval + generation end-to-end."""
    logger.info("=" * 80)
    logger.info("FinRAG Generation Module Test (Retrieval + Generation)")
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
        logger.info("Retriever initialized")
    except Exception as e:
        logger.error(f"✗ Retriever initialization failed: {e}")
        conn.close()
        return

    # Initialize generator
    try:
        logger.info("Loading Mistral 7B Q4 (Metal backend)...")
        generator = FinancialGenerator()
        logger.info("Generator initialized with Mistral 7B")
    except Exception as e:
        logger.error(f"✗ Generator initialization failed: {e}")
        conn.close()
        return

    # Test query
    query = "What are JPMorgan primary credit risk factors?"
    logger.info(f"\nTest Query: {query}")
    logger.info("=" * 80)

    # Step 1: Retrieve
    logger.info("\n[RETRIEVAL] Using reranked strategy (best)...")
    try:
        chunks = retriever.retrieve(
            query=query,
            strategy="reranked",
            top_k=5,
            candidate_k=20
        )
        logger.info(f"Retrieved {len(chunks)} chunks")
        for i, c in enumerate(chunks, 1):
            logger.info(f"  [{i}] {c.ticker} {c.section} (score: {c.score:.4f})")
    except Exception as e:
        logger.error(f"✗ Retrieval failed: {e}")
        conn.close()
        return

    # Step 2: Generate Q&A response
    logger.info("\n[GENERATION] Q&A task...")
    try:
        response = generator.answer_query(query, chunks)
        logger.info(f"Answer:\n{response['answer']}")
        logger.info(f"\nSources cited: {len(response['sources'])}")
        for src in response['sources']:
            logger.info(
                f"  • {src['ticker']} {src['filing_type']} ({src['section']}) "
                f"via {src['method']}"
            )
    except Exception as e:
        logger.error(f"✗ Q&A generation failed: {e}")

    conn.close()
    logger.info("\n" + "=" * 80)
    logger.info("Generation test complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_generation()
