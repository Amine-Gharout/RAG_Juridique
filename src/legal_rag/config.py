from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_per_tier: int = 8
    final_context_k: int = 6
    min_score_to_keep: float = 0.08
    min_docs_for_confident_answer: int = 2
    min_score_for_no_fallback: float = 0.2
    vector_candidates_per_tier: int = 20


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    api_key: str | None
    model: str
    output_dimension: int
    index_dir: Path
    use_vector_search: bool
    auto_build_index: bool
    metric: str
    batch_size: int


@dataclass(frozen=True)
class Settings:
    project_root: Path
    tier_a_path: Path
    tier_b_path: Path
    tier_c_path: Path
    llm_provider: str
    groq_api_key: str | None
    groq_model: str
    gemini_api_key: str | None
    gemini_model: str
    temperature: float
    retrieval: RetrievalConfig
    embedding: EmbeddingConfig


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(project_root: Path | None = None) -> Settings:
    _load_dotenv()

    root = project_root or Path(__file__).resolve().parents[2]
    output_dir = root / "out" / "cleaning"
    embedding_model = os.getenv("LEGAL_RAG_EMBEDDING_MODEL", "voyage-4-large")
    embedding_dim = int(os.getenv("LEGAL_RAG_EMBEDDING_DIM", "1024"))
    default_index_dir = root / "out" / "embeddings" / \
        f"{embedding_model}-d{embedding_dim}"

    return Settings(
        project_root=root,
        tier_a_path=_path_from_env(
            "LEGAL_RAG_TIER_A",
            output_dir / "16-FR-only.articles.trusted_A.jsonl",
        ),
        tier_b_path=_path_from_env(
            "LEGAL_RAG_TIER_B",
            output_dir / "16-FR-only.articles.review_B.jsonl",
        ),
        tier_c_path=_path_from_env(
            "LEGAL_RAG_TIER_C",
            output_dir / "16-FR-only.articles.quarantine_C.jsonl",
        ),
        llm_provider=os.getenv("LEGAL_RAG_LLM_PROVIDER", "groq").lower(),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("LEGAL_RAG_GROQ_MODEL",
                             "llama-3.3-70b-versatile"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("LEGAL_RAG_GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=float(os.getenv("LEGAL_RAG_TEMPERATURE", "0.2")),
        retrieval=RetrievalConfig(
            top_k_per_tier=int(os.getenv("LEGAL_RAG_TOP_K_PER_TIER", "15")),
            final_context_k=int(os.getenv("LEGAL_RAG_FINAL_CONTEXT_K", "12")),
            min_score_to_keep=float(
                os.getenv("LEGAL_RAG_MIN_SCORE_TO_KEEP", "0.08")),
            min_docs_for_confident_answer=int(
                os.getenv("LEGAL_RAG_MIN_DOCS_FOR_CONFIDENT_ANSWER", "2")
            ),
            min_score_for_no_fallback=float(
                os.getenv("LEGAL_RAG_MIN_SCORE_FOR_NO_FALLBACK", "0.2")
            ),
            vector_candidates_per_tier=int(
                os.getenv("LEGAL_RAG_VECTOR_CANDIDATES_PER_TIER", "20")
            ),
        ),
        embedding=EmbeddingConfig(
            provider=os.getenv("LEGAL_RAG_EMBEDDING_PROVIDER", "voyageai"),
            api_key=os.getenv("VOYAGE_API_KEY"),
            model=embedding_model,
            output_dimension=embedding_dim,
            index_dir=_path_from_env("LEGAL_RAG_INDEX_DIR", default_index_dir),
            use_vector_search=_bool_from_env(
                "LEGAL_RAG_USE_VECTOR_SEARCH", True),
            auto_build_index=_bool_from_env(
                "LEGAL_RAG_AUTO_BUILD_INDEX", False),
            metric=os.getenv("LEGAL_RAG_FAISS_METRIC", "ip"),
            batch_size=int(os.getenv("LEGAL_RAG_EMBED_BATCH_SIZE", "40")),
        ),
    )
