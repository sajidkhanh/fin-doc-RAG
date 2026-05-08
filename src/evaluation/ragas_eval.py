"""
evaluation/ragas_eval.py

Benchmarks all three retrieval strategies using RAGAS metrics:
  - Faithfulness: Is the answer grounded in retrieved context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are retrieved chunks actually relevant?
  - Context Recall: Are all relevant facts retrieved?

Also compares base vs fine-tuned embeddings.
Outputs a results table saved to docs/results_table.csv.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Ground-truth eval set — financial Q&A with known answers
EVAL_QUESTIONS = [
    {
        "question": "What are JPMorgan's primary credit risk factors disclosed in their most recent 10-K?",
        "ground_truth": "JPMorgan discloses credit risk from loan portfolios, counterparty exposure in derivatives, and concentration risk in commercial real estate and leveraged lending."
    },
    {
        "question": "How does Goldman Sachs describe its market risk management approach?",
        "ground_truth": "Goldman Sachs uses Value-at-Risk (VaR) models, stress testing, and scenario analysis to manage market risk across trading portfolios."
    },
    {
        "question": "What liquidity risk disclosures does Bank of America make in its 10-Q?",
        "ground_truth": "Bank of America discloses liquidity risk through its liquidity coverage ratio, available liquidity sources, and stress scenario funding gap analysis."
    },
    {
        "question": "What cybersecurity risks does Wells Fargo identify in its annual filing?",
        "ground_truth": "Wells Fargo identifies risks from cyberattacks, data breaches, third-party vendor vulnerabilities, and regulatory requirements around data protection."
    },
    {
        "question": "How does Morgan Stanley characterize its operational risk exposure?",
        "ground_truth": "Morgan Stanley characterizes operational risk through potential losses from failed internal processes, people, systems, or external events including legal and compliance failures."
    },
    {
        "question": "What does Citigroup disclose about regulatory capital requirements?",
        "ground_truth": "Citigroup discloses Common Equity Tier 1 (CET1) capital ratios, stress capital buffer requirements, and compliance with Basel III capital adequacy standards."
    },
    {
        "question": "What are the main revenue drivers for JPMorgan's consumer banking segment?",
        "ground_truth": "JPMorgan's consumer banking revenue is driven by net interest income from deposits and loans, credit card fees, mortgage banking, and investment services fees."
    },
    {
        "question": "How does Goldman Sachs describe risks associated with its investment banking business?",
        "ground_truth": "Goldman Sachs describes risks including deal flow volatility, client concentration, regulatory scrutiny of M&A activity, and reputational risk from advisory roles."
    },
]

STRATEGIES = ["naive_dense", "hybrid", "reranked"]
EMBEDDING_VARIANTS = ["base", "finetuned"]


def run_eval_for_strategy(
    questions: List[Dict],
    retriever,
    generator,
    strategy: str,
    embedding_variant: str
) -> Dict:
    """Run RAGAS evaluation for one strategy + embedding variant."""
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for item in questions:
        chunks = retriever.retrieve(item["question"], strategy=strategy, top_k=5)
        result = generator.answer_query(item["question"], chunks)

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c.content for c in chunks])
        eval_data["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(eval_data)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    return {
        "strategy": strategy,
        "embedding": embedding_variant,
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevancy": round(scores["answer_relevancy"], 4),
        "context_precision": round(scores["context_precision"], 4),
        "context_recall": round(scores["context_recall"], 4),
    }


def run_full_benchmark(retriever_base, retriever_finetuned, generator) -> pd.DataFrame:
    """
    Full ablation benchmark across strategies and embedding variants.
    Returns a DataFrame ready for the results table in the README.
    """
    results = []

    mlflow.set_experiment("finrag_ablation")

    for strategy in STRATEGIES:
        for variant, retriever in [("base", retriever_base), ("finetuned", retriever_finetuned)]:
            logger.info(f"Evaluating: strategy={strategy}, embeddings={variant}")
            with mlflow.start_run(run_name=f"{strategy}_{variant}"):
                row = run_eval_for_strategy(
                    EVAL_QUESTIONS, retriever, generator, strategy, variant
                )
                mlflow.log_metrics({
                    k: v for k, v in row.items()
                    if isinstance(v, float)
                })
                mlflow.log_params({"strategy": strategy, "embedding": variant})
                results.append(row)
                logger.info(f"  Result: {row}")

    df = pd.DataFrame(results)
    output_path = Path("docs/results_table.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    return df


def print_results_table(df: pd.DataFrame) -> None:
    """Print formatted results table for README."""
    print("\n" + "="*80)
    print("FINRAG ABLATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("\nBest configuration:", df.loc[
        df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean(axis=1).idxmax()
    ][["strategy", "embedding"]].to_dict())


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # WHY sys.path insert: ragas_eval runs from src/evaluation/, need src/ in path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from db import get_connection
    from retrieval.retriever import FinancialRetriever
    from generation.generator import FinancialGenerator

    conn = get_connection()
    generator = FinancialGenerator()

    base_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"))
    retriever_base = FinancialRetriever(conn, base_model)

    finetuned_path = os.getenv("FINETUNED_EMBEDDING_PATH", "./models/finetuned-fin-embedder")
    finetuned_model = SentenceTransformer(finetuned_path)
    retriever_finetuned = FinancialRetriever(conn, finetuned_model)

    df = run_full_benchmark(retriever_base, retriever_finetuned, generator)
    print_results_table(df)
    conn.close()
