from __future__ import annotations

from typing import TypedDict


class ArticleRecord(TypedDict, total=False):
    id: str
    article_number: str
    hierarchy: dict[str, str | None]
    ocr_quality_score: float
    quality_signals: dict[str, object]
    text: str
    filter_tier: str
    filter_reason: str
    filter_metrics: dict[str, object]


class Candidate(TypedDict):
    id: str
    article_number: str
    text: str
    tier: str
    lexical_score: float
    semantic_score: float
    vector_score: float
    exact_match_score: float
    fused_score: float
    ocr_quality_score: float


class QueryMetadata(TypedDict):
    article_reference: str | None
    wants_low_confidence: bool
    retrieval_query: str


class LegalRAGState(TypedDict, total=False):
    user_query: str
    allow_low_confidence: bool
    conversation_history: list[dict[str, str]]

    query_metadata: QueryMetadata

    tier_a_candidates: list[Candidate]
    tier_b_candidates: list[Candidate]
    tier_c_candidates: list[Candidate]
    selected_candidates: list[Candidate]
    tier_path: list[str]

    needs_clarification: bool
    warnings: list[str]

    draft_answer: str
    final_answer: str
    cited_articles: list[str]
    confidence_level: str
