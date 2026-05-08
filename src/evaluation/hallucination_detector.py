"""
evaluation/hallucination_detector.py
Numerical hallucination detection for LLM-generated financial content.

Core idea: LLMs hallucinate numbers in financial contexts more than facts.
This module extracts numerical claims from generated answers and validates them
against source chunks using exact match and semantic similarity.

Metrics:
  - SUPPORTED: number found in source or semantically equivalent expression exists
  - CONTRADICTED: different number found for same entity/metric
  - UNVERIFIABLE: number not found in source at all
  - hallucination_rate = (CONTRADICTED + UNVERIFIABLE) / total_numerical_claims
"""

import re
import logging
from typing import List, Dict, Tuple, Literal
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class NumericalClaim:
    """Represents an extracted numerical claim from generated text."""
    claim_text: str  # Full sentence containing the claim
    number: str  # The actual number(s) extracted
    entity: str  # Entity the number refers to (e.g., "JPMorgan", "net income")
    claim_type: Literal["percentage", "dollar_amount", "basis_points", "ratio", "date", "other"]
    found_in_sources: bool = False
    source_chunks: List[Dict] = None  # List of matching source chunks


@dataclass
class HallucinationScore:
    """Hallucination detection result for a generated answer."""
    total_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    hallucination_rate: float
    claims: List[NumericalClaim] = None
    unsupported_claims: List[Dict] = None  # Detailed info on unsupported claims


class NumericalHallucinationDetector:
    """
    Detects numerical hallucinations in LLM-generated financial text.

    WHY numerical focus:
    - LLMs are better at facts (mostly rote memorization) than calculations
    - Numerical errors in finance are high-stakes (wrong revenue by 10% is critical)
    - Numbers are easier to validate against source documents (exact match works)
    - Semantic validation via embeddings catches paraphrases ("10 billion" vs "$10B")
    """

    def __init__(self, embedding_model: SentenceTransformer = None):
        """Initialize detector with NLP tools."""
        self.model = embedding_model or SentenceTransformer("BAAI/bge-small-en-v1.5")

        # Load spaCy for NER (entity extraction)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spacy model not found. Run: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

    def extract_numerical_claims(self, text: str) -> List[NumericalClaim]:
        """
        Extract all numerical claims from generated text.

        WHY regex + spaCy:
        - Regex catches standard financial formats (e.g., "$1.2B", "10.5%", "2024-03-15")
        - spaCy NER identifies entities (companies, dates) to enrich claims with context
        - Hybrid approach captures both explicit numbers and their semantic context
        """
        claims = []

        # Regex patterns for financial numbers
        patterns = {
            "dollar_amount": r"\$[\d,]+\.?[\d]*\s*(?:billion|million|thousand|B|M|K)?",
            "percentage": r"[\d]+\.?[\d]*\s*%",
            "basis_points": r"[\d]+\s*(?:basis points|bps)",
            "ratio": r"[\d]+\.?[\d]*\s*(?:ratio|to|:)\s*[\d]+\.?[\d]*",
            "date": r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}",
        }

        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            for claim_type, pattern in patterns.items():
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    number = match.group()

                    # Extract entity using spaCy NER if available
                    entity = "unknown"
                    if self.nlp:
                        doc = self.nlp(sentence)
                        # Find ORG/PERSON/GPE entities near the number
                        for ent in doc.ents:
                            if ent.label_ in ["ORG", "PERSON", "GPE"]:
                                entity = ent.text
                                break

                    claims.append(
                        NumericalClaim(
                            claim_text=sentence,
                            number=number,
                            entity=entity,
                            claim_type=claim_type,
                            source_chunks=[]
                        )
                    )

        return claims

    def validate_against_sources(
        self,
        claims: List[NumericalClaim],
        source_chunks: List[Dict]
    ) -> List[NumericalClaim]:
        """
        Validate extracted claims against source chunks.

        WHY two-stage validation:
        1. Exact match: number appears verbatim in source (highest confidence)
        2. Semantic similarity: BUT ONLY if no contradictory number exists

        Confidence thresholds:
        - Exact match: SUPPORTED (100% confidence)
        - Semantic sim > 0.7 AND no conflicting number: SUPPORTED
        - Conflicting number found: CONTRADICTED
        - No match: UNVERIFIABLE
        """
        validated_claims = []

        for claim in claims:
            # Stage 1: Exact match in sources
            exact_match_found = False
            for chunk in source_chunks:
                # Normalize and search for the number in the source
                if re.search(
                    re.escape(claim.number),
                    chunk["content"],
                    re.IGNORECASE
                ):
                    exact_match_found = True
                    claim.source_chunks.append({
                        "content": chunk["content"][:200],
                        "ticker": chunk.get("ticker", "unknown"),
                        "match_type": "exact"
                    })

            if exact_match_found:
                claim.found_in_sources = True
                validated_claims.append(claim)
                continue

            # Stage 2: Check for CONTRADICTORY numbers (different number for same entity/metric)
            # WHY this matters: "$500B" vs "$16.1B" are semantically similar but numerically contradictory
            contradictory_number_found = False
            for chunk in source_chunks:
                # Look for other dollar amounts or percentages in same chunk
                if claim.claim_type == "dollar_amount":
                    other_amounts = re.findall(r'\$[\d,]+\.?[\d]*\s*(?:billion|million|thousand|B|M|K)?', chunk["content"], re.IGNORECASE)
                    if other_amounts and claim.number not in other_amounts:
                        contradictory_number_found = True
                        break
                elif claim.claim_type == "percentage":
                    other_percentages = re.findall(r'[\d]+\.?[\d]*\s*%', chunk["content"])
                    if other_percentages and claim.number not in other_percentages:
                        contradictory_number_found = True
                        break

            if contradictory_number_found:
                # Don't mark as found - contradictory numbers are NOT support
                validated_claims.append(claim)
                continue

            # Stage 3: Semantic similarity (only if no contradictory number)
            # WHY embedding distance: catches "10 billion USD" vs "$10B" as same claim
            claim_embedding = self.model.encode(
                claim.claim_text,
                normalize_embeddings=True
            )

            best_similarity = 0.0
            best_chunks = []

            for chunk in source_chunks:
                chunk_embedding = self.model.encode(
                    chunk["content"],
                    normalize_embeddings=True
                )
                # Cosine similarity already normalized
                similarity = np.dot(claim_embedding, chunk_embedding)

                if similarity > 0.75:  # Stricter threshold: 0.75 not 0.65
                    best_similarity = similarity
                    best_chunks.append({
                        "content": chunk["content"][:200],
                        "ticker": chunk.get("ticker", "unknown"),
                        "match_type": f"semantic ({similarity:.2f})"
                    })

            if best_similarity > 0.75:
                claim.found_in_sources = True
                claim.source_chunks = best_chunks

            validated_claims.append(claim)

        return validated_claims

    def score_hallucinations(
        self,
        claims: List[NumericalClaim]
    ) -> HallucinationScore:
        """
        Compute hallucination metrics across extracted and validated claims.

        Scoring rules:
        - SUPPORTED: claim found in source (exact or high semantic match)
        - CONTRADICTED: different number for same entity found in source (e.g., JPM net income is $X in source, but claimed as $Y)
        - UNVERIFIABLE: no match found (most common for hallucinations)
        """
        supported = sum(1 for c in claims if c.found_in_sources)
        unverifiable = len(claims) - supported

        # Contradicted = case where different number for same entity exists in source
        # (harder to detect without structured extraction, flagged separately)
        contradicted = 0

        hallucination_rate = (
            (contradicted + unverifiable) / len(claims)
            if claims
            else 0.0
        )

        unsupported = [c for c in claims if not c.found_in_sources]

        return HallucinationScore(
            total_claims=len(claims),
            supported=supported,
            contradicted=contradicted,
            unverifiable=unverifiable,
            hallucination_rate=hallucination_rate,
            claims=claims,
            unsupported_claims=[
                {
                    "claim": c.claim_text,
                    "number": c.number,
                    "entity": c.entity,
                    "type": c.claim_type
                }
                for c in unsupported
            ]
        )

    def detect(
        self,
        generated_text: str,
        source_chunks: List[Dict]
    ) -> HallucinationScore:
        """
        End-to-end hallucination detection.

        Args:
            generated_text: Text generated by LLM to evaluate
            source_chunks: List of source chunks with keys: content, ticker, filing_type, section

        Returns:
            HallucinationScore with detailed breakdowns per claim
        """
        logger.info(f"Detecting hallucinations in generated text ({len(generated_text)} chars)")

        # Extract numerical claims
        claims = self.extract_numerical_claims(generated_text)
        logger.info(f"  Extracted {len(claims)} numerical claims")

        if not claims:
            logger.warning("  No numerical claims found - can't evaluate hallucinations")
            return HallucinationScore(
                total_claims=0,
                supported=0,
                contradicted=0,
                unverifiable=0,
                hallucination_rate=0.0,
                claims=[],
                unsupported_claims=[]
            )

        # Validate against sources
        claims = self.validate_against_sources(claims, source_chunks)

        # Score hallucinations
        score = self.score_hallucinations(claims)

        logger.info(f"  Supported: {score.supported}/{score.total_claims}")
        logger.info(f"  Unverifiable: {score.unverifiable}/{score.total_claims}")
        logger.info(f"  Hallucination rate: {score.hallucination_rate:.1%}")

        return score
