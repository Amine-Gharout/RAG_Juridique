from __future__ import annotations

import math
import re
from collections import Counter

from .config import RetrievalConfig
from .models import ArticleRecord, Candidate


_ARTICLE_REF_RE = re.compile(
    r"\b(?:art(?:icle)?\.?)\s*([0-9]+(?:\s*(?:bis|ter|quater))?)\b",
    re.IGNORECASE,
)

_EN_FR_LEGAL_HINTS: dict[str, tuple[str, ...]] = {
    "theft": ("vol",),
    "steal": ("vol",),
    "penalty": ("peine",),
    "penalties": ("peines",),
    "sentence": ("peine",),
    "punishment": ("peine",),
    "crime": ("infraction",),
    "law": ("loi",),
    "judge": ("magistrat",),
    "prison": ("emprisonnement",),
    "fine": ("amende",),
    "attempt": ("tentative",),
    "state": ("etat",),
}


def normalize_article_number(value: str) -> str:
    lowered = value.strip().lower().replace(".", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def extract_article_reference(query: str) -> str | None:
    match = _ARTICLE_REF_RE.search(query)
    if not match:
        return None
    return normalize_article_number(match.group(1))


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str) -> list[str]:
    normalized = normalize_text(value)
    return normalized.split() if normalized else []


def expand_query_for_legal_search(query: str) -> str:
    expansions: list[str] = []
    for token in tokenize(query):
        expansions.extend(_EN_FR_LEGAL_HINTS.get(token, ()))
    if not expansions:
        return query
    return f"{query} {' '.join(expansions)}"


def _token_overlap_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)
    common = sum(min(query_counts[token], doc_counts[token])
                 for token in query_counts)
    denom = math.sqrt(len(query_tokens) * len(doc_tokens))
    return common / denom if denom else 0.0


def _char_trigrams(text: str) -> set[str]:
    if len(text) < 3:
        return {text} if text else set()
    return {text[i: i + 3] for i in range(len(text) - 2)}


def _semantic_score(query: str, doc: str) -> float:
    q = normalize_text(query)
    d = normalize_text(doc)
    if not q or not d:
        return 0.0
    q_trigrams = _char_trigrams(q)
    d_trigrams = _char_trigrams(d)
    if not q_trigrams or not d_trigrams:
        return 0.0
    inter = len(q_trigrams & d_trigrams)
    union = len(q_trigrams | d_trigrams)
    return inter / union if union else 0.0


def _tier_weight(tier: str) -> float:
    if tier == "A":
        return 1.0
    if tier == "B":
        return 0.82
    return 0.55


def _score_article(
    query: str,
    article: ArticleRecord,
    tier: str,
    article_reference: str | None,
) -> Candidate:
    effective_query = expand_query_for_legal_search(query)
    text = str(article.get("text", ""))
    article_number = str(article.get("article_number", ""))

    query_tokens = tokenize(effective_query)
    doc_tokens = tokenize(text)

    lexical = _token_overlap_score(query_tokens, doc_tokens)
    semantic = _semantic_score(effective_query, text)

    normalized_article_number = normalize_article_number(article_number)
    exact = 1.0 if article_reference and normalized_article_number == article_reference else 0.0

    base = (0.55 * lexical) + (0.25 * semantic) + (0.20 * exact)
    quality = float(article.get("ocr_quality_score", 0.0))
    quality_weight = 0.7 + (0.3 * max(0.0, min(1.0, quality)))
    fused = base * _tier_weight(tier) * quality_weight

    return {
        "id": str(article.get("id", "")),
        "article_number": article_number,
        "text": text,
        "tier": tier,
        "lexical_score": round(lexical, 5),
        "semantic_score": round(semantic, 5),
        "exact_match_score": round(exact, 5),
        "fused_score": round(fused, 5),
        "ocr_quality_score": round(quality, 5),
    }


def retrieve_for_tier(
    query: str,
    rows: list[ArticleRecord],
    tier: str,
    cfg: RetrievalConfig,
    article_reference: str | None,
) -> list[Candidate]:
    candidates = [
        _score_article(
            query=query,
            article=article,
            tier=tier,
            article_reference=article_reference,
        )
        for article in rows
    ]

    # For explicit article queries, keep exact hits only when available.
    if article_reference:
        exact_hits = [
            item for item in candidates if item["exact_match_score"] > 0]
        if exact_hits:
            exact_hits.sort(key=lambda item: item["fused_score"], reverse=True)
            return exact_hits[: cfg.top_k_per_tier]

    candidates = [c for c in candidates if c["fused_score"] >=
                  cfg.min_score_to_keep or c["exact_match_score"] > 0]
    candidates.sort(key=lambda item: item["fused_score"], reverse=True)
    return candidates[: cfg.top_k_per_tier]


def merge_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    best_by_id: dict[str, Candidate] = {}
    for candidate in candidates:
        previous = best_by_id.get(candidate["id"])
        if not previous or candidate["fused_score"] > previous["fused_score"]:
            best_by_id[candidate["id"]] = candidate

    merged = list(best_by_id.values())
    merged.sort(key=lambda item: item["fused_score"], reverse=True)
    return merged[:limit]
