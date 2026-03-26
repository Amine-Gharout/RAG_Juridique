from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build VoyageAI embeddings and persist FAISS indices for legal RAG tiers."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing index directory and rebuild from scratch.",
    )
    return parser


def main() -> None:
    config_module = importlib.import_module("legal_rag.config")
    corpus_module = importlib.import_module("legal_rag.corpus")
    embedding_module = importlib.import_module("legal_rag.embedding_service")
    vector_store_module = importlib.import_module("legal_rag.vector_store")

    load_settings = getattr(config_module, "load_settings")
    load_tiered_corpus = getattr(corpus_module, "load_tiered_corpus")
    VoyageEmbeddingService = getattr(
        embedding_module, "VoyageEmbeddingService")
    LegalVectorStore = getattr(vector_store_module, "LegalVectorStore")

    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings(PROJECT_ROOT)
    if not settings.embedding.api_key:
        raise SystemExit(
            "VOYAGE_API_KEY is required to build the vector index.")

    corpus_by_tier = load_tiered_corpus(settings)
    vector_store = LegalVectorStore(settings)

    if args.force and settings.embedding.index_dir.exists():
        shutil.rmtree(settings.embedding.index_dir)

    embedding_service = VoyageEmbeddingService(settings.embedding)
    output_dir = vector_store.build_and_save(
        corpus_by_tier=corpus_by_tier,
        embedding_service=embedding_service,
        force=False,
    )

    counts = {tier: len(rows) for tier, rows in corpus_by_tier.items()}
    print("Vector index build completed.")
    print(f"Output directory: {output_dir}")
    print(
        f"Model: {settings.embedding.model} | dim={settings.embedding.output_dimension}")
    print(f"Tier counts: {counts}")


if __name__ == "__main__":
    main()
