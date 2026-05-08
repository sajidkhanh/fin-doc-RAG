# FinRAG: Financial Document Retrieval & Generation System

A production-grade Retrieval-Augmented Generation (RAG) system for financial document analysis with semantic search, domain-adapted embeddings, and numerical hallucination detection.

## Overview

FinRAG combines multiple retrieval strategies with local LLM inference to analyze financial documents (SEC filings, annual reports, earnings calls). It's designed for accurate financial information extraction with built-in hallucination detection.

## Key Metrics

- 3 Retrieval Strategies: Dense semantic search, hybrid BM25+RRF, Cohere-reranked
- Retrieval latency: 50-500ms depending on strategy
- Generation speed: ~20 tokens/second on Apple Metal
- Model: Mistral 7B Q4 (4.1GB, auto-downloaded)
- Embedding: BAAI/bge-small-en-v1.5 (384-dimensional)
- Real financial data: 15 verified chunks from 2023 SEC 10-K filings
- Database: PostgreSQL + pgvector with IVFFlat indexing

## Quick Start

```bash
# Setup
bash setup.sh

# Load data
python load_real_data_simple.py

# Test
python test_retrieval.py
python test_generation.py
```

## Architecture

```
Ingestion: Documents → Extract sections → Chunk → Embed → Store in pgvector

Retrieval: Query → Dense + BM25 + RRF → [Optional Rerank] → Top-k chunks

Generation: Chunks → Mistral 7B → Answer with citations

Evaluation: Answer → Extract claims → Validate → Accuracy metrics
```

## Features

- Dense semantic search via pgvector cosine similarity
- Hybrid BM25 + dense vector retrieval with Reciprocal Rank Fusion
- Cohere cross-encoder reranking for precision optimization
- PEFT/LoRA domain-adapted embeddings on FinQA dataset
- Numerical hallucination detection with contradiction detection
- Local LLM inference on Apple Metal GPU (no API calls)
- Real verified financial data from SEC EDGAR filings

## Project Structure

```
finrag/
├── src/
│   ├── retrieval/retriever.py       # Retrieval strategies
│   ├── generation/generator.py      # Mistral 7B inference
│   ├── evaluation/hallucination_detector.py
│   ├── models/finetune_embeddings.py
│   └── ingestion/sec_loader.py
├── tests/
├── verified_real_chunks.json        # Real financial data
├── load_real_data_simple.py
├── test_retrieval.py
├── test_generation.py
├── generate_final_report.py
├── setup.sh
├── requirements.txt
└── README.md
```

## Data

Uses verified real financial data from 2023 SEC 10-K filings:
- JPMorgan Chase (3.9T assets)
- Bank of America (3.2T assets)
- Wells Fargo (2.0T assets)

All metrics are official and verifiable against SEC.gov.

## Installation

```bash
git clone https://github.com/yourusername/finrag.git
cd finrag

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Database
brew install postgresql
brew services start postgresql
createdb finrag
```

## Configuration

Edit module constants to customize:
- Chunk size and overlap
- Embedding model
- LLM parameters
- Hallucination detection thresholds

## Model Download

The Mistral 7B Q4 model (4.1GB) downloads automatically on first use. It is cached locally for fast subsequent runs.

Hardware requirements:
- macOS: Apple Silicon (M1+) with Metal GPU
- Linux/Windows: CPU inference supported
- Disk: 5GB free space

## License

MIT - see LICENSE file
  url={https://github.com/yourusername/finrag}
}
