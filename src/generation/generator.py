"""
generation/generator.py
Multi-task generation using Mistral 7B Q4 via llama-cpp-python.
Handles: summarization, classification, intelligent search (Q&A),
         and next-best-action recommendations.
Runs fully locally on Apple Silicon (M4) via Metal backend.
"""

import os
import json
import logging
from typing import List, Literal, Optional
from pathlib import Path

from llama_cpp import Llama
from dotenv import load_dotenv

from retrieval.retriever import RetrievedChunk

load_dotenv()
logger = logging.getLogger(__name__)

TASK = Literal["summarization", "classification", "qa", "next_best_action"]

RISK_CATEGORIES = [
    "credit_risk", "market_risk", "operational_risk",
    "liquidity_risk", "regulatory_risk", "cyber_risk", "macro_risk"
]


class FinancialGenerator:
    """
    Local LLM generator using Mistral 7B Instruct Q4_K_M.
    Supports four task types matching JPMorgan Applied AI JD requirements:
      - summarization: section-level filing summarizer
      - classification: risk factor categorization
      - qa: intelligent search / grounded Q&A
      - next_best_action: operational recommendation generation
    """

    def __init__(self, model_path: Optional[str] = None):
        model_path = model_path or os.getenv("MISTRAL_MODEL_PATH")
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"Mistral model not found at {model_path}.\n"
                "Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF\n"
                "Run: wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/"
                "resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -P ./models/"
            )
        logger.info(f"Loading Mistral from {model_path} with Metal backend...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,   # offload all layers to Metal GPU on M4
            n_threads=8,
            verbose=False
        )
        logger.info("Mistral loaded successfully.")

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] ({chunk.ticker} | {chunk.filing_type} | {chunk.section})\n"
                f"{chunk.content[:800]}"
            )
        return "\n\n".join(parts)

    def _generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Run inference via llama.cpp."""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            stop=["</s>", "[INST]"],
            echo=False
        )
        return response["choices"][0]["text"].strip()

    def summarize(self, chunks: List[RetrievedChunk], ticker: str) -> str:
        """Generate a structured executive summary from filing sections."""
        context = self._format_context(chunks)
        prompt = f"""<s>[INST]
You are a senior financial analyst. Using only the SEC filing excerpts below,
write a concise 3-paragraph executive summary for {ticker} covering:
1. Business overview and key revenue drivers
2. Primary risk factors disclosed
3. Notable financial trends or management commentary

Excerpts:
{context}

Executive Summary:[/INST]"""
        return self._generate(prompt, max_tokens=600)

    def classify_risk(self, chunk: RetrievedChunk) -> dict:
        """Classify a filing chunk into risk categories with confidence reasoning."""
        categories_str = ", ".join(RISK_CATEGORIES)
        prompt = f"""<s>[INST]
You are a financial risk analyst. Classify the following SEC filing excerpt into
one or more of these risk categories: {categories_str}

Respond in JSON only with keys: "primary_category", "secondary_categories", "confidence", "reasoning"

Excerpt:
{chunk.content[:600]}
[/INST]"""
        raw = self._generate(prompt, max_tokens=200)
        try:
            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return {
                "primary_category": "unknown",
                "secondary_categories": [],
                "confidence": 0.0,
                "reasoning": raw
            }

    def answer_query(self, query: str, chunks: List[RetrievedChunk]) -> dict:
        """
        Grounded Q&A — intelligent search over financial filings.
        Returns answer with source citations and confidence.
        """
        context = self._format_context(chunks)
        prompt = f"""<s>[INST]
You are a financial analyst assistant. Answer the following question using ONLY
the provided SEC filing excerpts. If the answer is not in the excerpts, say
"Not found in provided filings." Cite the source number(s) you used.

Question: {query}

Excerpts:
{context}

Answer (with citations):[/INST]"""
        answer = self._generate(prompt, max_tokens=400)
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "ticker": c.ticker,
                    "filing_type": c.filing_type,
                    "section": c.section,
                    "score": round(c.score, 4),
                    "method": c.retrieval_method
                }
                for c in chunks
            ]
        }

    def next_best_action(self, query: str, chunks: List[RetrievedChunk], context_type: str = "operations") -> str:
        """
        Generate next-best-action recommendations for operational workflows.
        Mirrors JPMorgan's operational AI use case.
        """
        context = self._format_context(chunks)
        prompt = f"""<s>[INST]
You are an AI system supporting bank operations. Based on the following financial
filing excerpts and the operational query below, recommend the next best actions
an operations team should take. Be specific and prioritize by urgency.

Operational context: {context_type}
Query: {query}

Filing excerpts:
{context}

Recommended next actions (numbered list):[/INST]"""
        return self._generate(prompt, max_tokens=400)
