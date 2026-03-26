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

## MVP Legal RAG (LangGraph + Groq)

This repository now includes a first MVP of a confidence-aware legal chat RAG pipeline.

### Implemented MVP Components

- `src/legal_rag/config.py`: runtime config and file path settings.
- `src/legal_rag/corpus.py`: JSONL loading for tier A/B/C corpora.
- `src/legal_rag/retrieval.py`: hybrid retrieval (exact article match + lexical + semantic proxy scoring).
- `src/legal_rag/llm.py`: Groq chat completion integration.
- `src/legal_rag/graph.py`: LangGraph workflow with confidence routing and citation verification.
- `src/legal_rag/cli.py`: interactive or single-query chat CLI.
- `scripts/run_legal_rag.py`: launcher script from project root.

### Graph Behavior

1. Parse user query and detect explicit article references.
2. Retrieve from tier A first.
3. Fallback to tier B when tier A confidence is insufficient.
4. Use tier C only if low-confidence mode is enabled.
5. Generate answer via Groq.
6. Verify citations against retrieved context before returning response.

### Install RAG Dependencies

```bash
python3 -m pip install -r requirements-rag.txt
```

### Configure Environment

1. Copy `.env.example` to `.env`.
2. Set `GROQ_API_KEY`.
3. Optionally adjust `LEGAL_RAG_GROQ_MODEL` and temperature.

### Run (Single Query)

```bash
python3 scripts/run_legal_rag.py \
	--query "What does Art. 1 say?" \
	--show-debug
```

### Run (Interactive Chat)

```bash
python3 scripts/run_legal_rag.py --show-debug
```

### Optional Low-Confidence Mode

This mode allows tier C retrieval when needed.

```bash
python3 scripts/run_legal_rag.py \
	--query "Explain Art. 77" \
	--allow-low-confidence \
	--show-debug
```

### Current Scope and Next Expansion

- MVP focus: reliable retrieval pathing, grounded answers, verified citations.
- Next step: add evaluation suite (golden queries + citation accuracy metrics) and API serving layer.
