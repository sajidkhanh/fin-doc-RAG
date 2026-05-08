"""
retrieval/retriever.py
Three retrieval strategies for ablation benchmarking:
  1. Naive dense retrieval (baseline)
  2. Hybrid BM25 + dense (improved)
  3. Hybrid + Cohere reranker (best)
"""

import os
import logging
from typing import List, Dict, Tuple, Literal
from dataclasses import dataclass
from pathlib import Path

import cohere
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# WHY sys.path insert: retriever runs from src/retrieval/, need src/ in path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from db import get_connection

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: int
    ticker: str
    filing_type: str
    section: str
    content: str
    score: float
    retrieval_method: str


class FinancialRetriever:
    """
    Multi-strategy retriever enabling ablation study across:
    - naive_dense: cosine similarity over pgvector
    - hybrid: BM25 + dense score fusion (RRF)
    - reranked: hybrid candidates reranked via Cohere
    """

    def __init__(self, conn, embedding_model: SentenceTransformer):
        self.conn = conn
        self.model = embedding_model
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY", ""))
        self._bm25_index = None
        self._bm25_corpus = None
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build in-memory BM25 index over all stored chunks.

        WHY handle empty corpus:
        - If ingestion hasn't completed, filings table may be empty
        - BM25Okapi crashes with ZeroDivisionError on empty corpus
        - Initialize with dummy corpus if needed, flag to user
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, content FROM filings ORDER BY id;")
            rows = cur.fetchall()

        self._bm25_ids = [r[0] for r in rows]
        self._bm25_corpus = [r[1] for r in rows]

        if not self._bm25_corpus:
            # Graceful handling: empty corpus (ingestion not yet complete)
            logger.warning(
                "BM25 index built over 0 chunks. "
                "Ingestion may not be complete. "
                "Initialize with empty tokenized corpus to avoid crashes."
            )
            tokenized = [[]]  # Dummy corpus to avoid ZeroDivisionError
        else:
            tokenized = [doc.lower().split() for doc in self._bm25_corpus]

        self._bm25_index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(rows)} chunks.")

    def _dense_retrieve(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        """Retrieve top-k chunks via pgvector cosine similarity."""
        query_emb = self.model.encode(
            query, normalize_embeddings=True
        ).tolist()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, filing_type, section, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM filings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_emb, query_emb, top_k))
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                id=r[0], ticker=r[1], filing_type=r[2],
                section=r[3], content=r[4], score=float(r[5]),
                retrieval_method="dense"
            ) for r in rows
        ]

    def _bm25_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Retrieve top-k chunk IDs and BM25 scores."""
        tokens = query.lower().split()
        scores = self._bm25_index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._bm25_ids[i], float(scores[i])) for i in top_indices]

    def _reciprocal_rank_fusion(
        self,
        dense_chunks: List[RetrievedChunk],
        bm25_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[RetrievedChunk]:
        """Fuse dense and BM25 rankings via RRF scoring."""
        rrf_scores: Dict[int, float] = {}
        id_to_chunk: Dict[int, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_chunks):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + 1 / (k + rank + 1)
            id_to_chunk[chunk.id] = chunk

        bm25_id_to_content = {cid: score for cid, score in bm25_results}
        for rank, (chunk_id, _) in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            if chunk_id not in id_to_chunk:
                # Fetch missing chunk from DB
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, ticker, filing_type, section, content FROM filings WHERE id=%s",
                        (chunk_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        id_to_chunk[chunk_id] = RetrievedChunk(
                            id=row[0], ticker=row[1], filing_type=row[2],
                            section=row[3], content=row[4], score=0.0,
                            retrieval_method="bm25"
                        )

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        results = []
        for cid in sorted_ids:
            if cid in id_to_chunk:
                chunk = id_to_chunk[cid]
                chunk.score = rrf_scores[cid]
                chunk.retrieval_method = "hybrid_rrf"
                results.append(chunk)
        return results

    def retrieve(
        self,
        query: str,
        strategy: Literal["naive_dense", "hybrid", "reranked"] = "reranked",
        top_k: int = 5,
        candidate_k: int = 20
    ) -> List[RetrievedChunk]:
        """
        Unified retrieval interface.

        Args:
            query: Natural language query
            strategy: One of naive_dense | hybrid | reranked
            top_k: Final number of chunks to return
            candidate_k: Candidate pool size for hybrid/reranked
        """
        if strategy == "naive_dense":
            return self._dense_retrieve(query, top_k=top_k)

        dense_chunks = self._dense_retrieve(query, top_k=candidate_k)

        if strategy == "hybrid":
            bm25_results = self._bm25_retrieve(query, top_k=candidate_k)
            fused = self._reciprocal_rank_fusion(dense_chunks, bm25_results)
            return fused[:top_k]

        if strategy == "reranked":
            bm25_results = self._bm25_retrieve(query, top_k=candidate_k)
            candidates = self._reciprocal_rank_fusion(dense_chunks, bm25_results)[:candidate_k]
            # Cohere reranker
            try:
                response = self.cohere_client.rerank(
                    query=query,
                    documents=[c.content for c in candidates],
                    top_n=top_k,
                    model="rerank-english-v3.0"
                )
                reranked = []
                for hit in response.results:
                    chunk = candidates[hit.index]
                    chunk.score = hit.relevance_score
                    chunk.retrieval_method = "reranked"
                    reranked.append(chunk)
                return reranked
            except Exception as e:
                logger.warning(f"Cohere reranker failed, falling back to hybrid: {e}")
                return candidates[:top_k]

        raise ValueError(f"Unknown strategy: {strategy}")
