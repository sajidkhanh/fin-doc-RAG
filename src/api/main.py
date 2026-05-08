"""
api/main.py
FastAPI application exposing FinRAG endpoints.
"""

import os
import logging
from typing import Literal, Optional, List
from contextlib import asynccontextmanager

import psycopg2
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Global state
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, models on startup."""
    logger.info("Initializing FinRAG API...")
    state["conn"] = psycopg2.connect(os.getenv("POSTGRES_URL"))

    from retrieval.retriever import FinancialRetriever
    from generation.generator import FinancialGenerator

    embedding_path = os.getenv("FINETUNED_EMBEDDING_PATH", os.getenv("EMBEDDING_MODEL_NAME"))
    state["model"] = SentenceTransformer(embedding_path)
    state["retriever"] = FinancialRetriever(state["conn"], state["model"])
    state["generator"] = FinancialGenerator()
    logger.info("FinRAG API ready.")
    yield
    state["conn"].close()


app = FastAPI(
    title="FinRAG — Financial Document Intelligence API",
    description=(
        "RAG pipeline over SEC 10-K/10-Q filings supporting intelligent search, "
        "summarization, risk classification, and next-best-action recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    strategy: Literal["naive_dense", "hybrid", "reranked"] = "reranked"
    top_k: int = 5
    ticker_filter: Optional[str] = None


class SummarizeRequest(BaseModel):
    ticker: str
    filing_type: Literal["10-K", "10-Q"] = "10-K"
    strategy: Literal["naive_dense", "hybrid", "reranked"] = "reranked"


class ClassifyRequest(BaseModel):
    text: str


class NextBestActionRequest(BaseModel):
    query: str
    context_type: str = "operations"
    strategy: Literal["naive_dense", "hybrid", "reranked"] = "reranked"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "mistral-7b-q4", "vector_store": "pgvector"}


@app.post("/search", summary="Intelligent search over SEC filings")
def search(req: QueryRequest):
    """
    Grounded Q&A over SEC filings using the selected retrieval strategy.
    Returns answer with source citations and retrieval metadata.
    """
    try:
        chunks = state["retriever"].retrieve(
            req.query, strategy=req.strategy, top_k=req.top_k
        )
        result = state["generator"].answer_query(req.query, chunks)
        return result
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", summary="Summarize SEC filing for a ticker")
def summarize(req: SummarizeRequest):
    """
    Generate an executive summary of the most recent filing for a given ticker.
    """
    try:
        query = f"{req.ticker} {req.filing_type} business overview risk factors financial highlights"
        chunks = state["retriever"].retrieve(query, strategy=req.strategy, top_k=8)
        chunks = [c for c in chunks if c.ticker == req.ticker.upper()] or chunks
        summary = state["generator"].summarize(chunks, req.ticker)
        return {"ticker": req.ticker, "filing_type": req.filing_type, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", summary="Classify text into financial risk categories")
def classify(req: ClassifyRequest):
    """
    Classify a filing excerpt into financial risk categories
    (credit, market, operational, liquidity, regulatory, cyber, macro).
    """
    try:
        from retrieval.retriever import RetrievedChunk
        dummy_chunk = RetrievedChunk(
            id=0, ticker="", filing_type="", section="",
            content=req.text, score=1.0, retrieval_method="direct"
        )
        result = state["generator"].classify_risk(dummy_chunk)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/next-best-action", summary="Generate operational next-best-action recommendations")
def next_best_action(req: NextBestActionRequest):
    """
    Generate prioritized next-best-action recommendations for operational workflows
    grounded in SEC filing disclosures.
    """
    try:
        chunks = state["retriever"].retrieve(req.query, strategy=req.strategy, top_k=5)
        recommendation = state["generator"].next_best_action(
            req.query, chunks, req.context_type
        )
        return {
            "query": req.query,
            "recommendations": recommendation,
            "sources": [{"ticker": c.ticker, "section": c.section} for c in chunks]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickers", summary="List available tickers in the knowledge base")
def list_tickers():
    """Return all tickers with filing counts available in the vector store."""
    try:
        with state["conn"].cursor() as cur:
            cur.execute("""
                SELECT ticker, filing_type, COUNT(*) as chunks
                FROM filings
                GROUP BY ticker, filing_type
                ORDER BY ticker, filing_type;
            """)
            rows = cur.fetchall()
        return [{"ticker": r[0], "filing_type": r[1], "chunks": r[2]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
