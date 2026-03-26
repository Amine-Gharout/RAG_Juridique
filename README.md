## Legal Corpus Cleaning (Pre-RAG)

This repository now includes a preprocessing pipeline for noisy legal text.

### Script

- `scripts/prepare_legal_corpus.py`

### What it does

1. Removes pagination markers and obvious editorial noise.
2. Rebuilds broken OCR paragraphs.
3. Normalizes typography and spacing.
4. Removes exact duplicate blocks.
5. Flags suspicious OCR fragments with quality scores.
6. Exports article-level JSONL chunks for RAG indexing.

### Usage

```bash
python3 scripts/prepare_legal_corpus.py \
	--input 16-FR-only.md \
	--output-dir out/cleaning \
	--suspicious-threshold 0.78 \
	--tier-a-threshold 0.88 \
	--tier-c-threshold 0.70 \
	--max-low-block-ratio 0.20 \
	--critical-repeat-threshold 2
```

### Outputs

- `out/cleaning/16-FR-only.cleaned.md`: cleaned deduplicated corpus.
- `out/cleaning/16-FR-only.articles.jsonl`: article-level chunks + hierarchy metadata + OCR quality.
- `out/cleaning/16-FR-only.articles.trusted_A.jsonl`: high-confidence articles for primary index.
- `out/cleaning/16-FR-only.articles.review_B.jsonl`: medium-confidence articles for fallback index.
- `out/cleaning/16-FR-only.articles.quarantine_C.jsonl`: quarantined low-confidence articles.
- `out/cleaning/16-FR-only.duplicates.jsonl`: exact duplicate audit log.
- `out/cleaning/16-FR-only.suspicious.jsonl`: low-confidence OCR fragments for manual review.
- `out/cleaning/16-FR-only.report.json`: summary stats and file pointers.

### Notes

- Legal markers such as `Art.`, `Article`, and `bis/ter/quater` are preserved.
- Suspicious fragments are tracked, not silently discarded.
- Retrieval policy recommendation: query tier A first, tier B on fallback, and tier C only in explicit low-confidence mode.
