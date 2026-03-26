from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_per_tier: int = 8
    final_context_k: int = 6
    min_score_to_keep: float = 0.08
    min_docs_for_confident_answer: int = 2
    min_score_for_no_fallback: float = 0.2


@dataclass(frozen=True)
class Settings:
    project_root: Path
    tier_a_path: Path
    tier_b_path: Path
    tier_c_path: Path
    groq_api_key: str | None
    groq_model: str
    temperature: float
    retrieval: RetrievalConfig


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


def load_settings(project_root: Path | None = None) -> Settings:
    root = project_root or Path(__file__).resolve().parents[2]
    output_dir = root / "out" / "cleaning"

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
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("LEGAL_RAG_GROQ_MODEL",
                             "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("LEGAL_RAG_TEMPERATURE", "0.2")),
        retrieval=RetrievalConfig(),
    )
