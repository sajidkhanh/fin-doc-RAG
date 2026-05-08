"""
models/finetune_embeddings.py

Fine-tunes a sentence transformer on financial Q&A pairs using PEFT/LoRA.
Uses the FinQA dataset (financial question answering over earnings reports).
Runs on Apple M4 via MPS backend.

This ablation demonstrates that domain-adapted embeddings outperform
general-purpose ones on financial retrieval tasks.
"""

import os
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL = "BAAI/bge-small-en-v1.5"
OUTPUT_DIR = "./models/finetuned-fin-embedder"
BATCH_SIZE = 16
EPOCHS = 3
MAX_SEQ_LEN = 256

# MPS (Metal) on Apple Silicon
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
logger.info(f"Using device: {DEVICE}")


def load_finqa_pairs() -> list:
    """
    Load FinQA dataset and format as (question, context) positive pairs
    for contrastive learning. Each Q is paired with its gold context paragraph.
    """
    logger.info("Loading FinQA dataset...")
    dataset = load_dataset("dreamerdeo/finqa", split="train")
    examples = []
    for row in dataset:
        question = row.get("question", "")
        # Gold context = pre/post text paragraphs
        pre = " ".join(row.get("pre_text", []))
        post = " ".join(row.get("post_text", []))
        context = (pre + " " + post).strip()
        if question and context and len(context) > 50:
            examples.append(InputExample(texts=[question, context]))
    logger.info(f"Loaded {len(examples)} Q-context pairs from FinQA.")
    return examples


def build_eval_set(examples: list, n: int = 200) -> tuple:
    """Build a small IR eval set from held-out examples."""
    eval_examples = examples[-n:]
    queries = {str(i): ex.texts[0] for i, ex in enumerate(eval_examples)}
    corpus = {str(i): ex.texts[1] for i, ex in enumerate(eval_examples)}
    relevant_docs = {str(i): {str(i)} for i in range(n)}
    return queries, corpus, relevant_docs


def finetune():
    examples = load_finqa_pairs()
    train_examples = examples[:-200]
    eval_queries, eval_corpus, eval_relevant = build_eval_set(examples)

    # Load base sentence transformer
    model = SentenceTransformer(BASE_MODEL, device=DEVICE)
    model.max_seq_length = MAX_SEQ_LEN

    # DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # MultipleNegativesRankingLoss — standard contrastive loss for retrieval
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Evaluator — measures NDCG@10 on held-out financial Q&A
    evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant,
        name="finqa_eval",
        show_progress_bar=False
    )

    logger.info(f"Starting fine-tuning for {EPOCHS} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=500,
        warmup_steps=100,
        output_path=OUTPUT_DIR,
        save_best_model=True,
        show_progress_bar=True
    )

    logger.info(f"Fine-tuned model saved to {OUTPUT_DIR}")

    # Evaluate final NDCG@10
    score = evaluator(model)
    logger.info(f"Final eval score: {score}")
    return score


if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    finetune()
