#!/usr/bin/env python3
"""Preprocess noisy legal corpus text before RAG ingestion.

Pipeline steps:
1) Remove pagination and obvious editorial noise.
2) Reconstruct broken paragraphs from OCR line wraps.
3) Normalize typography and whitespace.
4) Remove exact duplicate blocks.
5) Track suspicious OCR fragments with quality scores.
6) Export article-level JSONL chunks with metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PAGE_RE = re.compile(r"^\s*##\s*Page\s+\d+\s*$", re.IGNORECASE)
ORPHAN_BIS_RE = re.compile(r"^\s*\d+\s+bis\s*$", re.IGNORECASE)
ARTICLE_RE = re.compile(
    r"^\s*(?:Art(?:icle)?\.?\s*)(\d+)(?:\s*[\.-]?\s*(bis|ter|quater))?(?:\s*(\d+))?",
    re.IGNORECASE,
)

INLINE_ARTICLE_START_RE = re.compile(r"\bArt\.?\s*\d+", re.IGNORECASE)

HEADING_PREFIX_RE = re.compile(
    r"^\s*(LIVRE|TITRE|CHAPITRE|SECTION|PREMIERE PARTIE|DEUXIEME PARTIE|TROISIEME PARTIE|QUATRIEME PARTIE)\b",
    re.IGNORECASE,
)

BULLET_RE = re.compile(r"^\s*((\d+[°\)\.-])|(-\s)|([a-zA-Z]\)))")

# Editorial noise seen in the document cover pages.
EDITORIAL_LINES = {
    "CODE PENAL",
    "4eme Edition",
    "4ème Edition",
    "DEPOT LEGAL 3388 - 2004",
    "DEPOT LEGAL 3388 – 2004",
    "ISBN 9961 - 41 - 045 - 9",
    "ISBN 9961 – 41 – 045 - 9",
    "Les Editions",
    "de l'O. N. T. E",
    "de l’O. N. T. E",
    "CO",
}

ALLOWED_SYMBOLS_RE = re.compile(
    r"[^\w\s\.,;:!?()\[\]{}%/\-+'\"°ªº€$£àâäéèêëîïôöùûüçœæÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒÆ]")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
SPACED_LETTER_CLUSTER_RE = re.compile(
    r"(?:\b[A-Za-zÀ-ÖØ-öø-ÿ]\s+){3,}[A-Za-zÀ-ÖØ-öø-ÿ]\b")
NO_VOWEL_RE = re.compile(r"^[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{5,}$")


@dataclass
class QualityResult:
    score: float
    reasons: list[str]


@dataclass
class CleanStats:
    total_lines: int = 0
    removed_page_markers: int = 0
    removed_orphan_bis: int = 0
    removed_editorial_lines: int = 0
    removed_empty_after_cleanup: int = 0
    blocks_before_dedup: int = 0
    blocks_after_dedup: int = 0
    duplicate_blocks_removed: int = 0
    suspicious_blocks: int = 0


@dataclass
class FilterConfig:
    tier_a_threshold: float = 0.88
    tier_c_threshold: float = 0.70
    max_low_block_ratio: float = 0.20
    critical_repeat_threshold: int = 2


CRITICAL_OCR_REASONS = {
    "spaced_letter_clusters_detected",
    "unexpected_characters",
}


def normalize_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)
    return text.strip()


def normalize_typography(text: str) -> str:
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+-\s+", " - ", text)
    text = normalize_spaces(text)
    return text


def line_is_editorial_noise(line: str, line_no: int) -> bool:
    raw = line.strip()
    if not raw:
        return False

    normalized = normalize_typography(raw)
    if normalized in EDITORIAL_LINES:
        return True

    # Cover-page cleanup: strong editorial lines are only dropped near the beginning.
    if line_no <= 40:
        if normalized.startswith("DEPOT LEGAL") or normalized.startswith("ISBN"):
            return True
    return False


def clean_lines(raw_lines: list[str], stats: CleanStats) -> list[str]:
    cleaned: list[str] = []
    for idx, line in enumerate(raw_lines, start=1):
        stats.total_lines += 1
        normalized = normalize_typography(line)

        if PAGE_RE.match(normalized):
            stats.removed_page_markers += 1
            continue

        if ORPHAN_BIS_RE.match(normalized):
            stats.removed_orphan_bis += 1
            continue

        if line_is_editorial_noise(normalized, idx):
            stats.removed_editorial_lines += 1
            continue

        if not normalized:
            stats.removed_empty_after_cleanup += 1
            cleaned.append("")
            continue

        cleaned.append(normalized)

    return cleaned


def is_heading(line: str) -> bool:
    if HEADING_PREFIX_RE.match(line):
        return True

    compact = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", line)
    if len(compact) >= 8:
        upper = sum(1 for c in compact if c.isupper())
        if upper / max(1, len(compact)) >= 0.8 and len(line.split()) <= 8:
            return True
    return False


def starts_new_block(line: str) -> bool:
    return bool(ARTICLE_RE.match(line) or HEADING_PREFIX_RE.match(line) or BULLET_RE.match(line))


def merge_with_previous(previous: str, current: str) -> str:
    if not previous:
        return current

    if previous.endswith("-") and current and current[0].islower():
        # Typical OCR hard-wrap hyphenation: "interdic-" + "tion".
        return previous[:-1] + current

    if previous.endswith("'"):
        return previous + current

    return previous + " " + current


def reconstruct_blocks(lines: Iterable[str]) -> list[str]:
    blocks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(normalize_spaces(current))
            current = ""

    for line in lines:
        if not line:
            flush()
            continue

        if starts_new_block(line):
            flush()
            current = line
            continue

        if not current:
            current = line
            continue

        if is_heading(current):
            flush()
            current = line
            continue

        current = merge_with_previous(current, line)

    flush()
    return [b for b in blocks if b]


def split_blocks_with_multiple_articles(blocks: list[str]) -> list[str]:
    split_blocks: list[str] = []
    for block in blocks:
        raw_starts = [m.start()
                      for m in INLINE_ARTICLE_START_RE.finditer(block)]
        starts: list[int] = []

        for pos in raw_starts:
            if pos == 0:
                starts.append(pos)
                continue

            prev_non_space = None
            back = pos - 1
            while back >= 0:
                ch = block[back]
                if not ch.isspace():
                    prev_non_space = ch
                    break
                back -= 1

            # Split only when the new Art. marker starts a new sentence-like segment.
            if prev_non_space in {".", ";", ":", "?", "!", "»", '"'}:
                starts.append(pos)

        if len(starts) <= 1:
            split_blocks.append(block)
            continue

        # Keep any prefix before the first article marker as its own block.
        prefix = block[: starts[0]].strip()
        if prefix:
            split_blocks.append(prefix)

        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(block)
            part = block[start:end].strip()
            if part:
                split_blocks.append(part)

    return split_blocks


def canonical_hash(text: str) -> str:
    canonical = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deduplicate_blocks(blocks: list[str], stats: CleanStats) -> tuple[list[str], list[dict[str, object]]]:
    seen: dict[str, int] = {}
    deduped: list[str] = []
    duplicates: list[dict[str, object]] = []

    stats.blocks_before_dedup = len(blocks)

    for index, block in enumerate(blocks):
        h = canonical_hash(block)
        if h in seen:
            stats.duplicate_blocks_removed += 1
            duplicates.append(
                {
                    "duplicate_block_index": index,
                    "first_seen_block_index": seen[h],
                    "hash": h,
                    "preview": block[:220],
                }
            )
            continue

        seen[h] = index
        deduped.append(block)

    stats.blocks_after_dedup = len(deduped)
    return deduped, duplicates


def score_ocr_quality(text: str) -> QualityResult:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return QualityResult(score=1.0, reasons=[])

    if is_heading(text):
        return QualityResult(score=1.0, reasons=[])

    def is_roman_numeral(token: str) -> bool:
        return bool(re.fullmatch(r"[IVXLCDM]+", token.upper()))

    if len(tokens) <= 5 and not SPACED_LETTER_CLUSTER_RE.search(text):
        # Very short legal fragments (amounts, list markers) should not be over-penalized.
        return QualityResult(score=0.95, reasons=[])

    meaningful_single_char = [t for t in tokens if len(
        t) == 1 and not is_roman_numeral(t)]
    single_char_ratio = len(meaningful_single_char) / len(tokens)
    no_vowel_words = [t for t in tokens if len(
        t) >= 5 and NO_VOWEL_RE.match(t)]
    no_vowel_ratio = len(no_vowel_words) / max(1, len(tokens))

    spaced_clusters = SPACED_LETTER_CLUSTER_RE.findall(text)
    cluster_density = len(spaced_clusters) / max(1.0, len(tokens) / 35)

    bad_chars = ALLOWED_SYMBOLS_RE.findall(text)
    bad_char_ratio = len(bad_chars) / max(1, len(text))

    score = 1.0 - min(
        1.0,
        (0.7 * single_char_ratio)
        + (1.3 * cluster_density)
        + (1.0 * no_vowel_ratio)
        + (1.4 * bad_char_ratio),
    )
    score = max(0.0, min(1.0, score))

    reasons: list[str] = []
    if single_char_ratio > 0.22:
        reasons.append("high_single_char_token_ratio")
    if cluster_density > 0.18:
        reasons.append("spaced_letter_clusters_detected")
    if no_vowel_ratio > 0.08:
        reasons.append("many_words_without_vowels")
    if bad_char_ratio > 0.015:
        reasons.append("unexpected_characters")

    return QualityResult(score=score, reasons=reasons)


def extract_article_number(block: str) -> str | None:
    match = ARTICLE_RE.match(block)
    if not match:
        return None

    base, suffix, sub = match.groups()
    parts = [base]
    if suffix:
        parts.append(suffix.lower())
    if sub:
        parts.append(sub)
    return " ".join(parts)


def slugify_article(article_number: str) -> str:
    return re.sub(r"\s+", "-", article_number.strip().lower())


def extract_hierarchy_label(block: str, prefix: str) -> str | None:
    if not block.lower().startswith(prefix.lower()):
        return None

    # Avoid false section/chapter captures from long prose lines.
    if len(block) > 90:
        return None

    if prefix.lower() in {"section", "chapitre"} and " " in block.strip() and not re.match(
        rf"^\s*{prefix}\b", block, re.IGNORECASE
    ):
        return None

    return block


def build_articles(blocks: list[str], quality: list[QualityResult]) -> list[dict[str, object]]:
    if len(blocks) != len(quality):
        raise ValueError("blocks and quality lengths must match")

    context = {
        "livre": None,
        "titre": None,
        "chapitre": None,
        "section": None,
    }

    articles: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    def close_current() -> None:
        nonlocal current
        if not current:
            return
        scores: list[float] = current.pop(
            "_scores")  # type: ignore[assignment]
        quality_items: list[dict[str, object]] = current.pop(
            "_quality_items")  # type: ignore[assignment]

        block_count = len(scores)
        low_score_block_count = sum(1 for s in scores if s < 0.70)
        critical_reason_hits = sum(
            1
            for item in quality_items
            if any(
                reason in CRITICAL_OCR_REASONS
                # type: ignore[union-attr]
                for reason in item.get("reasons", [])
            )
        )
        all_reasons = sorted(
            {
                reason
                for item in quality_items
                # type: ignore[union-attr]
                for reason in item.get("reasons", [])
            }
        )

        current["ocr_quality_score"] = round(
            float(statistics.mean(scores)), 4) if scores else 1.0
        current["quality_signals"] = {
            "block_count": block_count,
            "low_score_block_count": low_score_block_count,
            "critical_reason_hits": critical_reason_hits,
            "reasons": all_reasons,
        }
        current["text"] = "\n\n".join(
            current["paragraphs"])  # type: ignore[index]
        del current["paragraphs"]
        articles.append(current)
        current = None

    for block, q in zip(blocks, quality):
        if extract_hierarchy_label(block, "LIVRE"):
            context["livre"] = block
            continue
        if extract_hierarchy_label(block, "TITRE"):
            context["titre"] = block
            continue
        if extract_hierarchy_label(block, "Chapitre"):
            context["chapitre"] = block
            continue
        if extract_hierarchy_label(block, "Section"):
            context["section"] = block
            continue

        article_number = extract_article_number(block)
        if article_number:
            close_current()
            current = {
                "id": f"art-{slugify_article(article_number)}",
                "article_number": article_number,
                "hierarchy": dict(context),
                "paragraphs": [block],
                "_scores": [q.score],
                "_quality_items": [{"score": q.score, "reasons": q.reasons}],
            }
            continue

        if current:
            current["paragraphs"].append(block)  # type: ignore[index]
            current["_scores"].append(q.score)  # type: ignore[index]
            current["_quality_items"].append(  # type: ignore[index]
                {"score": q.score, "reasons": q.reasons}
            )

    close_current()
    return articles


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_article_filter_tier(article: dict[str, object], cfg: FilterConfig) -> tuple[str, str, dict[str, object]]:
    score = float(article.get("ocr_quality_score", 1.0))
    quality_signals = article.get("quality_signals", {})
    if not isinstance(quality_signals, dict):
        quality_signals = {}

    block_count = int(quality_signals.get("block_count", 1) or 1)
    low_score_block_count = int(
        quality_signals.get("low_score_block_count", 0) or 0)
    critical_reason_hits = int(
        quality_signals.get("critical_reason_hits", 0) or 0)

    low_block_ratio = low_score_block_count / max(1, block_count)

    metrics = {
        "ocr_quality_score": round(score, 4),
        "block_count": block_count,
        "low_score_block_count": low_score_block_count,
        "low_block_ratio": round(low_block_ratio, 4),
        "critical_reason_hits": critical_reason_hits,
    }

    if (
        score < cfg.tier_c_threshold
        or critical_reason_hits >= cfg.critical_repeat_threshold
        or low_block_ratio > cfg.max_low_block_ratio
    ):
        return "C", "quarantine_low_confidence", metrics

    if (
        score >= cfg.tier_a_threshold
        and critical_reason_hits == 0
        and low_score_block_count == 0
    ):
        return "A", "trusted_high_confidence", metrics

    return "B", "review_medium_confidence", metrics


def apply_article_filtering(
    articles: list[dict[str, object]], cfg: FilterConfig
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    tier_a: list[dict[str, object]] = []
    tier_b: list[dict[str, object]] = []
    tier_c: list[dict[str, object]] = []

    for article in articles:
        tier, reason, metrics = classify_article_filter_tier(article, cfg)
        article["filter_tier"] = tier
        article["filter_reason"] = reason
        article["filter_metrics"] = metrics

        if tier == "A":
            tier_a.append(article)
        elif tier == "B":
            tier_b.append(article)
        else:
            tier_c.append(article)

    return tier_a, tier_b, tier_c


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    suspicious_threshold: float,
    filter_config: FilterConfig,
) -> dict[str, object]:
    stats = CleanStats()
    raw_text = input_path.read_text(encoding="utf-8")
    cleaned_lines = clean_lines(raw_text.splitlines(), stats)

    blocks = reconstruct_blocks(cleaned_lines)
    blocks = split_blocks_with_multiple_articles(blocks)
    deduped_blocks, duplicates = deduplicate_blocks(blocks, stats)

    quality = [score_ocr_quality(block) for block in deduped_blocks]

    suspicious: list[dict[str, object]] = []
    for idx, (block, q) in enumerate(zip(deduped_blocks, quality)):
        if q.score < suspicious_threshold:
            stats.suspicious_blocks += 1
            suspicious.append(
                {
                    "block_index": idx,
                    "ocr_quality_score": round(q.score, 4),
                    "reasons": q.reasons,
                    "excerpt": block[:320],
                    "text": block,
                }
            )

    articles = build_articles(deduped_blocks, quality)
    tier_a, tier_b, tier_c = apply_article_filtering(articles, filter_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    cleaned_text_path = output_dir / f"{stem}.cleaned.md"
    cleaned_text_path.write_text("\n\n".join(
        deduped_blocks) + "\n", encoding="utf-8")

    suspicious_path = output_dir / f"{stem}.suspicious.jsonl"
    write_jsonl(suspicious_path, suspicious)

    duplicates_path = output_dir / f"{stem}.duplicates.jsonl"
    write_jsonl(duplicates_path, duplicates)

    articles_path = output_dir / f"{stem}.articles.jsonl"
    write_jsonl(articles_path, articles)

    articles_tier_a_path = output_dir / f"{stem}.articles.trusted_A.jsonl"
    write_jsonl(articles_tier_a_path, tier_a)

    articles_tier_b_path = output_dir / f"{stem}.articles.review_B.jsonl"
    write_jsonl(articles_tier_b_path, tier_b)

    articles_tier_c_path = output_dir / f"{stem}.articles.quarantine_C.jsonl"
    write_jsonl(articles_tier_c_path, tier_c)

    report = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "stats": {
            "total_lines": stats.total_lines,
            "removed_page_markers": stats.removed_page_markers,
            "removed_orphan_bis": stats.removed_orphan_bis,
            "removed_editorial_lines": stats.removed_editorial_lines,
            "removed_empty_after_cleanup": stats.removed_empty_after_cleanup,
            "blocks_before_dedup": stats.blocks_before_dedup,
            "blocks_after_dedup": stats.blocks_after_dedup,
            "duplicate_blocks_removed": stats.duplicate_blocks_removed,
            "suspicious_blocks": stats.suspicious_blocks,
            "articles_extracted": len(articles),
            "articles_tier_a": len(tier_a),
            "articles_tier_b": len(tier_b),
            "articles_tier_c": len(tier_c),
        },
        "quality": {
            "suspicious_threshold": suspicious_threshold,
            "avg_block_score": round(float(statistics.mean([q.score for q in quality])) if quality else 1.0, 4),
            "min_block_score": round(min((q.score for q in quality), default=1.0), 4),
            "max_block_score": round(max((q.score for q in quality), default=1.0), 4),
        },
        "filtering": {
            "tier_a_threshold": filter_config.tier_a_threshold,
            "tier_c_threshold": filter_config.tier_c_threshold,
            "max_low_block_ratio": filter_config.max_low_block_ratio,
            "critical_repeat_threshold": filter_config.critical_repeat_threshold,
            "routing_policy": "use A first, fallback to B, use C only when explicitly requested",
        },
        "files": {
            "cleaned_markdown": str(cleaned_text_path),
            "articles_jsonl": str(articles_path),
            "articles_tier_a_jsonl": str(articles_tier_a_path),
            "articles_tier_b_jsonl": str(articles_tier_b_path),
            "articles_tier_c_jsonl": str(articles_tier_c_path),
            "duplicates_jsonl": str(duplicates_path),
            "suspicious_jsonl": str(suspicious_path),
        },
        "notes": [
            "This pipeline removes obvious layout noise but preserves legal tokens (Art, bis/ter/quater, references).",
            "Suspicious OCR fragments are not deleted automatically; they are exported for manual review.",
            "Use tier A index first for retrieval, tier B as fallback, and tier C only with explicit low-confidence mode.",
        ],
    }

    report_path = output_dir / f"{stem}.report.json"
    report_path.write_text(json.dumps(
        report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare noisy legal text before RAG indexing.")
    parser.add_argument("--input", required=True,
                        help="Input text/markdown file")
    parser.add_argument(
        "--output-dir",
        default="out/cleaning",
        help="Output directory for cleaned corpus and audit files",
    )
    parser.add_argument(
        "--suspicious-threshold",
        type=float,
        default=0.78,
        help="OCR score threshold below which fragments are flagged for manual review",
    )
    parser.add_argument(
        "--tier-a-threshold",
        type=float,
        default=0.88,
        help="Minimum article score for trusted tier A",
    )
    parser.add_argument(
        "--tier-c-threshold",
        type=float,
        default=0.70,
        help="Maximum article score before quarantine tier C",
    )
    parser.add_argument(
        "--max-low-block-ratio",
        type=float,
        default=0.20,
        help="Maximum ratio of low-score blocks allowed outside quarantine",
    )
    parser.add_argument(
        "--critical-repeat-threshold",
        type=int,
        default=2,
        help="Number of critical OCR hits that triggers quarantine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    filter_config = FilterConfig(
        tier_a_threshold=args.tier_a_threshold,
        tier_c_threshold=args.tier_c_threshold,
        max_low_block_ratio=args.max_low_block_ratio,
        critical_repeat_threshold=args.critical_repeat_threshold,
    )

    report = run_pipeline(
        input_path,
        output_dir,
        args.suspicious_threshold,
        filter_config,
    )
    print(json.dumps(report["stats"], ensure_ascii=False))
    print(f"report={output_dir / (input_path.stem + '.report.json')}")


if __name__ == "__main__":
    main()
