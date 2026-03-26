from __future__ import annotations

from pathlib import Path
import json

from .config import Settings
from .models import ArticleRecord


def load_jsonl(path: Path) -> list[ArticleRecord]:
    rows: list[ArticleRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Shrink to the first 80% of lines as requested
    limit = int(len(rows) * 0.8)
    return rows[:limit]


def load_tiered_corpus(settings: Settings) -> dict[str, list[ArticleRecord]]:
    return {
        "A": load_jsonl(settings.tier_a_path),
        "B": load_jsonl(settings.tier_b_path),
        "C": load_jsonl(settings.tier_c_path),
    }
