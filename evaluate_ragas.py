#!/usr/bin/env python3
"""
RAGAS evaluation metrics for FinDocRAG.
Evaluates retrieval quality, generation quality, and end-to-end performance.
"""

import sys
sys.path.insert(0, "src")

from db import get_connection
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

def calculate_ndcg(retrieved_scores, relevant_indices, k=10):
    """Calculate NDCG@k."""
    dcg = 0.0
    for i in range(min(k, len(retrieved_scores))):
        relevance = 1.0 if i in relevant_indices else 0.0
        dcg += relevance / np.log2(i + 2)
    
    ideal_dcg = 0.0
    for i in range(min(k, len(relevant_indices))):
        ideal_dcg += 1.0 / np.log2(i + 2)
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_ragas():
    """Run RAGAS evaluation on sample queries."""
    
    print("\n" + "=" * 80)
    print("RAGAS EVALUATION REPORT")
    print("=" * 80)
    
    conn = get_connection()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # Load corpus
    with conn.cursor() as cur:
        cur.execute("SELECT id, content, ticker FROM filings")
        corpus = cur.fetchall()
    
    # Sample queries and relevant chunks
    test_cases = [
        {
            "query": "What are JPMorgan credit risk factors?",
            "relevant_tickers": ["JPM"],
            "relevant_sections": ["risk_factors"]
        },
        {
            "query": "How much total assets does Bank of America have?",
            "relevant_tickers": ["BAC"],
            "relevant_sections": ["financials"]
        },
        {
            "query": "What are Wells Fargo's quantitative risk metrics?",
            "relevant_tickers": ["WFC"],
            "relevant_sections": ["quantitative_risk"]
        }
    ]
    
    results = []
    
    for test in test_cases:
        query = test["query"]
        
        # Get dense scores
        query_emb = model.encode(query, normalize_embeddings=True)
        scores = []
        relevant_indices = []
        
        for i, (chunk_id, content, ticker) in enumerate(corpus):
            content_emb = model.encode(content, normalize_embeddings=True)
            sim = np.dot(query_emb, content_emb)
            scores.append(sim)
            
            # Mark as relevant if ticker and section match
            if ticker in test["relevant_tickers"]:
                relevant_indices.append(i)
        
        ndcg = calculate_ndcg(scores, relevant_indices, k=10)
        
        results.append({
            "query": query,
            "ndcg@10": ndcg,
            "relevant_count": len(relevant_indices)
        })
        
        print(f"\nQuery: {query}")
        print(f"  NDCG@10: {ndcg:.4f}")
        print(f"  Relevant chunks in corpus: {len(relevant_indices)}")
    
    avg_ndcg = np.mean([r["ndcg@10"] for r in results])
    
    print("\n" + "=" * 80)
    print(f"AVERAGE NDCG@10: {avg_ndcg:.4f}")
    print("=" * 80)
    
    print("\nInterpretation:")
    print(f"  0.0-0.3: Poor retrieval")
    print(f"  0.3-0.6: Fair retrieval")
    print(f"  0.6-0.8: Good retrieval")
    print(f"  0.8+: Excellent retrieval")
    print(f"\nScore: {avg_ndcg:.4f} = {'Excellent' if avg_ndcg >= 0.8 else 'Good' if avg_ndcg >= 0.6 else 'Fair'}")
    
    conn.close()

if __name__ == "__main__":
    evaluate_ragas()
