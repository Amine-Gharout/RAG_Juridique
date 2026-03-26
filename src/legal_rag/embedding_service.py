from __future__ import annotations

import importlib
import time
from typing import Literal

from .config import EmbeddingConfig


InputType = Literal["document", "query"]


def _numpy_module():
    try:
        return importlib.import_module("numpy")
    except Exception as exc:
        raise RuntimeError(
            "numpy is required for embedding operations. Install requirements-rag.txt."
        ) from exc


class VoyageEmbeddingService:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg

    def _get_client(self):
        if not self.cfg.api_key:
            raise RuntimeError(
                "Missing VOYAGE_API_KEY. Set it in your environment before embedding."
            )

        try:
            voyageai = importlib.import_module("voyageai")
            Client = getattr(voyageai, "Client")
        except Exception as exc:
            raise RuntimeError(
                "VoyageAI SDK is not installed. Install dependencies from requirements-rag.txt."
            ) from exc

        return Client(api_key=self.cfg.api_key)

    def _embed_batch(
        self,
        client,
        texts: list[str],
        input_type: InputType,
        max_retries: int = 3,
    ):
        np = _numpy_module()
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = client.embed(
                    texts=texts,
                    model=self.cfg.model,
                    input_type=input_type,
                    output_dimension=self.cfg.output_dimension,
                    truncation=True,
                )
                vectors = np.asarray(response.embeddings, dtype=np.float32)
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(0.75 * (2**attempt))

        raise RuntimeError(
            f"Voyage embedding failed after retries for input_type={input_type}."
        ) from last_error

    def embed_documents(self, texts: list[str]):
        np = _numpy_module()
        if not texts:
            return np.zeros((0, self.cfg.output_dimension), dtype=np.float32)

        client = self._get_client()
        all_vectors: list[object] = []

        batch_size = max(1, self.cfg.batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            vectors = self._embed_batch(
                client=client, texts=batch, input_type="document")
            all_vectors.append(vectors)

        return np.vstack(all_vectors)

    def embed_query(self, query: str):
        np = _numpy_module()
        if not query.strip():
            return np.zeros((self.cfg.output_dimension,), dtype=np.float32)

        client = self._get_client()
        vectors = self._embed_batch(
            client=client, texts=[query], input_type="query")
        return vectors[0]
