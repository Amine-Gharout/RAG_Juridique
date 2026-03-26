from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path

from .config import Settings
from .embedding_service import VoyageEmbeddingService
from .models import ArticleRecord


TIERS: tuple[str, str, str] = ("A", "B", "C")


def _numpy_module():
    try:
        return importlib.import_module("numpy")
    except Exception as exc:
        raise RuntimeError(
            "numpy is required for vector index operations. Install requirements-rag.txt."
        ) from exc


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tier_fingerprint(rows: list[ArticleRecord]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        article_id = str(row.get("id", ""))
        article_number = str(row.get("article_number", ""))
        text = str(row.get("text", ""))
        hasher.update(article_id.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(article_number.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(_text_hash(text).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


class LegalVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index_dir = settings.embedding.index_dir
        self.metric = settings.embedding.metric.lower()
        self.is_ready = False
        self.load_error: str | None = None

        self._faiss = None
        self._indices: dict[str, object] = {}
        self._meta_by_tier: dict[str, list[dict[str, object]]] = {}
        self._manifest: dict[str, object] = {}

    def _load_faiss_module(self):
        if self._faiss is not None:
            return self._faiss
        try:
            self._faiss = importlib.import_module("faiss")
        except Exception as exc:
            raise RuntimeError(
                "faiss is not installed. Install faiss-cpu from requirements-rag.txt."
            ) from exc
        return self._faiss

    def _manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def _tier_index_path(self, tier: str) -> Path:
        return self.index_dir / f"tier_{tier}.faiss"

    def _tier_meta_path(self, tier: str) -> Path:
        return self.index_dir / f"tier_{tier}.meta.json"

    def _compute_expected_fingerprints(
        self, corpus_by_tier: dict[str, list[ArticleRecord]]
    ) -> dict[str, str]:
        return {tier: _tier_fingerprint(corpus_by_tier[tier]) for tier in TIERS}

    def load(self, corpus_by_tier: dict[str, list[ArticleRecord]]) -> bool:
        self.is_ready = False
        self.load_error = None

        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            self.load_error = "Vector index manifest not found."
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.load_error = f"Could not parse vector index manifest: {exc}"
            return False

        expected_fingerprints = self._compute_expected_fingerprints(
            corpus_by_tier)

        model_ok = manifest.get("model") == self.settings.embedding.model
        dim_ok = int(manifest.get("dimension", -1)
                     ) == self.settings.embedding.output_dimension
        metric_ok = str(manifest.get("metric", "")).lower() == self.metric
        fingerprints_ok = manifest.get(
            "tier_fingerprints", {}) == expected_fingerprints

        if not (model_ok and dim_ok and metric_ok and fingerprints_ok):
            self.load_error = (
                "Vector index mismatch detected (model/dimension/metric/corpus fingerprint). "
                "Rebuild the index."
            )
            return False

        try:
            faiss = self._load_faiss_module()
            loaded_indices: dict[str, object] = {}
            loaded_meta: dict[str, list[dict[str, object]]] = {}

            for tier in TIERS:
                index = faiss.read_index(str(self._tier_index_path(tier)))
                meta = json.loads(self._tier_meta_path(
                    tier).read_text(encoding="utf-8"))
                if int(index.ntotal) != len(meta):
                    raise RuntimeError(
                        f"Tier {tier} index/meta size mismatch ({int(index.ntotal)} vs {len(meta)})."
                    )
                loaded_indices[tier] = index
                loaded_meta[tier] = meta

            self._indices = loaded_indices
            self._meta_by_tier = loaded_meta
            self._manifest = manifest
            self.is_ready = True
            return True
        except Exception as exc:
            self.load_error = f"Failed loading vector index: {exc}"
            return False

    def build_and_save(
        self,
        corpus_by_tier: dict[str, list[ArticleRecord]],
        embedding_service: VoyageEmbeddingService,
        force: bool = False,
    ) -> Path:
        if self.index_dir.exists() and not force:
            raise RuntimeError(
                f"Index directory {self.index_dir} already exists. Use force rebuild to overwrite."
            )

        faiss = self._load_faiss_module()
        np = _numpy_module()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        tier_fingerprints = self._compute_expected_fingerprints(corpus_by_tier)
        tier_counts: dict[str, int] = {}

        for tier in TIERS:
            rows = corpus_by_tier[tier]
            texts = [str(row.get("text", "")) for row in rows]
            print(f"Embedding tier '{tier}' ({len(texts)} documents)...")
            vectors = embedding_service.embed_documents(texts)

            if vectors.shape[0] != len(rows):
                raise RuntimeError(
                    f"Tier {tier}: embedding count mismatch ({vectors.shape[0]} vs {len(rows)})."
                )
            if vectors.shape[1] != self.settings.embedding.output_dimension:
                raise RuntimeError(
                    f"Tier {tier}: unexpected embedding dimension {vectors.shape[1]}."
                )

            vectors = vectors.astype(np.float32)
            if self.metric == "ip" and vectors.shape[0] > 0:
                faiss.normalize_L2(vectors)
                index = faiss.IndexFlatIP(
                    self.settings.embedding.output_dimension)
            else:
                index = faiss.IndexFlatL2(
                    self.settings.embedding.output_dimension)

            if vectors.shape[0] > 0:
                index.add(vectors)

            meta: list[dict[str, object]] = []
            for row in rows:
                text = str(row.get("text", ""))
                meta.append(
                    {
                        "id": str(row.get("id", "")),
                        "article_number": str(row.get("article_number", "")),
                        "ocr_quality_score": float(row.get("ocr_quality_score", 0.0)),
                        "text_hash": _text_hash(text),
                    }
                )

            faiss.write_index(index, str(self._tier_index_path(tier)))
            self._tier_meta_path(tier).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tier_counts[tier] = len(rows)

        manifest = {
            "provider": self.settings.embedding.provider,
            "model": self.settings.embedding.model,
            "dimension": self.settings.embedding.output_dimension,
            "metric": self.metric,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tier_counts": tier_counts,
            "tier_fingerprints": tier_fingerprints,
        }
        self._manifest_path().write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.load(corpus_by_tier)
        return self.index_dir

    def search_scores(self, tier: str, query_vector, k: int) -> dict[str, float]:
        if not self.is_ready:
            return {}
        if tier not in self._indices:
            return {}

        faiss = self._load_faiss_module()
        np = _numpy_module()
        index = self._indices[tier]
        meta = self._meta_by_tier[tier]

        vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.settings.embedding.output_dimension:
            raise RuntimeError(
                f"Query vector dimension {vector.shape[1]} does not match index dimension "
                f"{self.settings.embedding.output_dimension}."
            )

        if self.metric == "ip":
            faiss.normalize_L2(vector)

        distances, labels = index.search(vector, k)
        scores: dict[str, float] = {}

        for raw_score, row_idx in zip(distances[0], labels[0]):
            if row_idx < 0:
                continue
            row = meta[row_idx]
            article_id = str(row["id"])

            if self.metric == "ip":
                normalized_score = max(
                    0.0, min(1.0, (float(raw_score) + 1.0) / 2.0))
            else:
                normalized_score = 1.0 / (1.0 + max(0.0, float(raw_score)))

            previous = scores.get(article_id)
            if previous is None or normalized_score > previous:
                scores[article_id] = normalized_score

        return scores
