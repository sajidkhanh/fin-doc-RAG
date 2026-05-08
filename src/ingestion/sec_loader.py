"""
ingestion/sec_loader.py
Pulls 10-K and 10-Q filings from SEC EDGAR for target tickers.
Parses, chunks, and stores in PostgreSQL with pgvector embeddings.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Add parent directories to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from db import get_connection

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target tickers — major financial institutions
TARGET_TICKERS = ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC"]
FILING_TYPES = ["10-K", "10-Q"]
NUM_FILINGS = 3  # per ticker per type


def init_db(conn) -> None:
    """Create pgvector extension and filings table."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                id SERIAL PRIMARY KEY,
                ticker TEXT NOT NULL,
                filing_type TEXT NOT NULL,
                filing_date TEXT,
                section TEXT,
                chunk_id INTEGER,
                content TEXT NOT NULL,
                embedding vector(384),
                metadata JSONB
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS filings_embedding_idx
            ON filings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        conn.commit()
    logger.info("Database initialized.")


def download_filings(data_dir: Path) -> None:
    """Download SEC filings for all target tickers.

    SEC EDGAR blocks requests without a proper User-Agent header (returns 403).
    We monkey-patch requests.adapters.HTTPAdapter.send() to inject our User-Agent
    into every HTTP request at the lowest level, before requests are made.
    """
    import requests
    from requests.adapters import HTTPAdapter

    # WHY patch HTTPAdapter.send (not Session.__init__):
    # - This is the lowest level where all HTTP requests pass through
    # - Works regardless of when/how sec-edgar-downloader creates Sessions
    # - Ensures User-Agent is set on EVERY request, never overwritten
    original_send = HTTPAdapter.send

    def patched_send(self, request, *args, **kwargs):
        # WHY inject User-Agent header:
        # SEC EDGAR API returns 403 Forbidden without a meaningful User-Agent.
        # Default requests User-Agent is "python-requests/X.X.X" which looks like a bot.
        request.headers["User-Agent"] = "FinRAG Research sajidkhanhyder@gmail.com"
        return original_send(self, request, *args, **kwargs)

    HTTPAdapter.send = patched_send

    dl = Downloader(download_folder=str(data_dir))

    for ticker in TARGET_TICKERS:
        for filing_type in FILING_TYPES:
            try:
                # api.get(filing_type, ticker, amount=num_filings)
                # Returns the number of filings downloaded
                count = dl.get(filing_type, ticker, amount=NUM_FILINGS)
                logger.info(f"✓ {ticker:4} {filing_type:5} {count} filings")
            except Exception as e:
                logger.warning(f"✗ {ticker:4} {filing_type:5} failed: {type(e).__name__}")

    # Restore original HTTPAdapter (cleanup)
    HTTPAdapter.send = original_send


def parse_filing(filepath: Path) -> Dict[str, str]:
    """Extract key sections from SEC filing HTML.

    WHY section-based parsing:
    - Avoids ingesting non-content (headers, footers, boilerplate)
    - Allows section-specific retrieval (e.g., "risk_factors" queries)
    - Improves precision of retrieval (MDA is more relevant than cover pages)
    """
    sections = {}
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        text = soup.get_text(separator="\n")

        # Extract named sections using common 10-K/10-Q headers
        # WHY regex patterns: SEC filings follow Item numbering (1=Business, 1A=Risk, 7=MDA, 8=Financials)
        # These sections have the highest information density for financial analysis
        section_patterns = {
            "business": r"item\s*1[\.\s]*business",
            "risk_factors": r"item\s*1a[\.\s]*risk factors",
            "mda": r"item\s*7[\.\s]*management.*?discussion",
            "financials": r"item\s*8[\.\s]*financial statements",
            "quantitative_risk": r"item\s*7a[\.\s]*quantitative",
        }

        lines = text.split("\n")
        current_section = "preamble"
        sections[current_section] = []

        for line in lines:
            line_lower = line.lower().strip()
            for sec_name, pattern in section_patterns.items():
                if re.search(pattern, line_lower):
                    current_section = sec_name
                    sections.setdefault(current_section, [])
                    break
            sections[current_section].append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items() if v}
    except Exception as e:
        logger.warning(f"Parse error {filepath}: {e}")
        return {}


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Sliding window chunker with token-aware splitting.

    WHY sliding window with overlap:
    - Preserves semantic continuity (64-word overlap prevents context loss at boundaries)
    - 512 words ≈ 1700 chars, fits well within BAAI/bge-small model's 512-token limit
    - Overlap ensures entities/concepts split at boundaries remain retrievable

    WHY word-based not token-based:
    - Faster on CPU, sufficient for financial documents
    - Sentence-transformers handle tokenization consistently regardless of split points

    WHY minimum 50 chars:
    - Avoids storing noise (single words, numbers, empty sections)
    - Improves signal-to-noise in vector DB
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        # Only store chunks with meaningful content
        if len(chunk.strip()) > 50:
            chunks.append(chunk)

        # Advance by (chunk_size - overlap) to create sliding window
        start += chunk_size - overlap

    return chunks


def embed_chunks(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    """Batch embed text chunks using sentence transformer.

    WHY normalized embeddings:
    - pgvector uses vector_cosine_ops, which requires normalized vectors for correct similarity
    - Cosine similarity is invariant to magnitude, but normalization is expected by the index

    WHY batch_size=32:
    - Balances memory usage vs latency (larger batches on Apple M4 with Metal support)
    - BAAI/bge-small is lightweight (33M params), can handle larger batches on GPU

    WHY show_progress_bar=False:
    - Cleaner logging for production/CI environments
    """
    return model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True  # Required for cosine similarity in pgvector
    )


def ingest_all(data_dir: Path, conn, model: SentenceTransformer) -> int:
    """Walk downloaded filings, parse, chunk, embed, and store in pgvector.

    Pipeline:
      1. Parse filing HTML → extract key sections (business, risk, MDA, etc.)
      2. Chunk sections → sliding window (512 words, 64 overlap)
      3. Embed chunks → BAAI/bge-small-en-v1.5 (384-dim)
      4. Store in filings table with pgvector embeddings for fast retrieval

    WHY batch insert:
    - Single multi-value INSERT faster than N individual INSERTs
    - Reduces network round-trips (critical for 50k+ chunks)
    - execute_values() handles NULL/JSONB type casting
    """
    total_chunks = 0
    records = []

    # Walk through ticker directories
    for ticker in TARGET_TICKERS:
        ticker_dir = data_dir / "sec-edgar-filings" / ticker
        if not ticker_dir.exists():
            logger.warning(f"No directory for {ticker}, skipping")
            continue

        # Process each filing type (10-K, 10-Q)
        for filing_type in FILING_TYPES:
            filing_dir = ticker_dir / filing_type
            if not filing_dir.exists():
                continue

            # Get up to NUM_FILINGS per filing type per ticker
            filings = sorted(filing_dir.rglob("*.htm"))[:NUM_FILINGS]
            logger.info(f"{ticker} {filing_type}: found {len(filings)} filings")

            for filing_path in filings:
                sections = parse_filing(filing_path)
                if not sections:
                    logger.warning(f"No sections parsed from {filing_path}")
                    continue

                # Extract date from directory name (SEC EDGAR format: YYYY-MM-DD)
                filing_date = filing_path.parent.name

                # Process each section of the filing
                for section, content in sections.items():
                    chunks = chunk_text(content)
                    if not chunks:
                        continue

                    # Embed all chunks for this section at once (batched)
                    embeddings = embed_chunks(chunks, model)

                    # Build records for batch insert
                    # chunk_id tracks order within section (for reconstruction if needed)
                    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                        records.append((
                            ticker,
                            filing_type,
                            filing_date,
                            section,
                            i,  # chunk_id: position within section
                            chunk,
                            emb.tolist(),  # pgvector expects float array
                            {"source": str(filing_path)}  # track source for interpretability
                        ))
                        total_chunks += 1

    # Batch insert all records at once
    # WHY execute_values: handles type casting (vector, jsonb) correctly
    if records:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                    INSERT INTO filings
                        (ticker, filing_type, filing_date, section, chunk_id, content, embedding, metadata)
                    VALUES %s
                """,
                records,
                template="(%s,%s,%s,%s,%s,%s,%s::vector,%s::jsonb)"
            )
            conn.commit()
        logger.info(f"✓ Batch inserted {total_chunks} chunks into pgvector")
    else:
        logger.warning("No chunks to insert!")

    return total_chunks


if __name__ == "__main__":
    # Use relative path from project root for data storage
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get database connection using safe parameter-based approach
    # (avoids issues with special characters in passwords)
    conn = get_connection()

    # Load embedding model from environment or use default
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    logger.info(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    # Initialize database schema
    init_db(conn)
    logger.info("Database schema initialized.")

    # Step 1: Download filings from SEC EDGAR
    logger.info("Downloading SEC filings...")
    download_filings(data_dir)

    # Step 2: Ingest, chunk, embed, and store filings
    logger.info("Ingesting and embedding filings...")
    n = ingest_all(data_dir, conn, model)
    logger.info(f"✓ Done. {n} chunks stored in pgvector.")

    conn.close()
    logger.info("Database connection closed.")
