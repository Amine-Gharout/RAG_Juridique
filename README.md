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

## MVP Legal RAG (LangGraph + Groq/Gemini)

This repository now includes a first MVP of a confidence-aware legal chat RAG pipeline, capable of using either Groq (Llama) or Google Gemini as the core reasoning engine.

### Implemented MVP Components

- `src/legal_rag/config.py`: runtime config and file path settings.
- `src/legal_rag/corpus.py`: JSONL loading for tier A/B/C corpora.
- `src/legal_rag/retrieval.py`: hybrid retrieval (exact article match + lexical + semantic proxy scoring).
- `src/legal_rag/llm.py`: Groq chat completion integration.
- `src/legal_rag/graph.py`: LangGraph workflow with confidence routing and citation verification.
- `src/legal_rag/cli.py`: interactive or single-query chat CLI.
- `src/legal_rag/embedding_service.py`: VoyageAI embedding client wrapper (query/document modes).
- `src/legal_rag/vector_store.py`: FAISS index load/build/search + manifest validation.
- `scripts/run_legal_rag.py`: launcher script from project root.
- `scripts/build_voyage_faiss_index.py`: offline index build script for embeddings.

### Graph Behavior

1. Parse user query and detect explicit article references.
2. Retrieve from tier A first using hybrid scoring (exact + lexical + vector, when available).
3. Fallback to tier B when tier A confidence is insufficient.
4. Use tier C only if low-confidence mode is enabled.
5. Generate answer via Groq.
6. Verify citations against retrieved context before returning response.

### Vector Retrieval (VoyageAI + FAISS)

This project supports persisted vector retrieval with `voyage-4-large` embeddings and FAISS.

- Documents are embedded with `input_type=document`.
- Queries are embedded with `input_type=query`.
- Indices are stored on disk and loaded on next startup (no re-embedding each run).
- If vector indices are missing or invalid, the system falls back to lexical retrieval.

### Install RAG Dependencies

```bash
python3 -m pip install -r requirements-rag.txt
```

### Configure Environment

1. Copy `.env.example` to `.env`.
2. Set your preferred LLM provider via `LEGAL_RAG_LLM_PROVIDER`: `groq` or `gemini`.
3. Set your chosen valid LLM API key (`GROQ_API_KEY` or `GEMINI_API_KEY`).
4. Set `VOYAGE_API_KEY` for embeddings.
5. Optionally override default retrieval capacities (`LEGAL_RAG_TOP_K_PER_TIER` to fetch more documents, etc).

### Build Vector Index (One-Time or On Corpus Update)

```bash
python3 scripts/build_voyage_faiss_index.py
```

Force rebuild:

```bash
python3 scripts/build_voyage_faiss_index.py --force
```

Index output default:

- `out/embeddings/voyage-4-large-d1024/manifest.json`
- `out/embeddings/voyage-4-large-d1024/tier_A.faiss`
- `out/embeddings/voyage-4-large-d1024/tier_B.faiss`
- `out/embeddings/voyage-4-large-d1024/tier_C.faiss`
- `out/embeddings/voyage-4-large-d1024/tier_A.meta.json` (and B/C)

### Run (Single Query)

```bash
# You can optionally specify --provider to switch LLMs on the fly
python3 scripts/run_legal_rag.py \
        --query "What does Art. 1 say?" \
        --show-debug \
        --provider gemini

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

- MVP focus: reliable retrieval pathing, grounded answers, verified citations, and persistent vector search.
- Next step: add evaluation suite (golden queries + citation accuracy metrics) and API serving layer.
