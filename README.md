# Medical RAG

Retrieval-augmented question answering over medical reference texts (BNF 85,
Oxford Handbook of Clinical Medicine, and other PDFs in `RAW_DATA/`).

```
PDF parser
    ↓
Semantic chunker          breakpoint percentile 85 · overlap 20%
    ↓
PubMedBERT embeddings     NeuML/pubmedbert-base-embeddings (768-d)
    ↓
Qdrant HNSW               m=32 · ef_construct=200 · dense + BM25 sparse
    ↓
Query expansion           layman terms → medical terms        ← new component
    ↓
Hybrid search             BM25 + dense, RRF fusion (k=60)
    ↓
Cross-encoder reranker    MedCPT · mandatory, not optional
    ↓
Strict guardrails         confidence ≥ 0.8 · mandatory disclaimer
    ↓
Groq LLM                  llama-3.3-70b-versatile
```

See [docs/architecture.md](docs/architecture.md) for why each stage is tuned the
way it is.

---

## Setup

Works the same on a CPU laptop and a CUDA desktop — only `.env` differs.

```bash
git clone https://github.com/artifix09/medical_rag.git
cd medical_rag
python setup_env.py
```

`setup_env.py` creates `.venv`, detects whether an NVIDIA GPU is present,
installs the matching torch build, installs everything else, and creates `.env`
from the template. Then activate it:

```powershell
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
```

### Configuration

Edit `.env`. Only two values are mandatory:

| Variable | What it is |
|---|---|
| `GROQ_API_KEY` (alias: `GROK`) | Groq API key, `gsk_...` |
| `QDRANT_URL` | `http://localhost:6333` for Docker, or your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` (alias: `QDRANT`) | only needed for Qdrant Cloud |

Everything else has a working default that matches the architecture.

### Qdrant

Local, via Docker:

```bash
docker compose up -d           # http://localhost:6333/dashboard
```

Or point `QDRANT_URL` at a Qdrant Cloud cluster and set `QDRANT_API_KEY`.
Nothing else in the code changes.

### Verify

```bash
python scripts/check_setup.py --models
```

Checks dependencies, device, keys, `RAW_DATA/`, Docker, Qdrant connectivity, a
live Groq call, and (with `--models`) downloads and loads the embedding and
reranker weights. Every check runs independently, so one failure does not hide
the rest.

---

## Ingestion

Put PDFs in `RAW_DATA/` (git-ignored). Then, **on the GPU machine**:

```bash
python scripts/ingest.py --recreate     # full rebuild
python scripts/ingest.py                # resume, or add new PDFs
python scripts/ingest.py --status       # what is done so far
```

Ingestion runs in two independently resumable stages, tracked in
`data/processed/manifest.json`:

| Stage | Command | What it does |
|---|---|---|
| chunk | `--stage chunk` | parse → sentence-embed → semantic split → `data/processed/chunks/<doc_id>.jsonl` |
| index | `--stage index` | exact BM25 statistics → embed → upsert to Qdrant |

A crash or Ctrl-C costs one document, not the whole run. The BNF alone is ~1800
pages, so this matters.

To sanity-check parsing and chunking without touching Qdrant:

```bash
python scripts/ingest.py --dry-run --source "RAW_DATA/B09474-eng.pdf"
```

---

## Querying

```bash
python scripts/query.py "what are the contraindications for metformin?"
python scripts/query.py                          # REPL
python scripts/query.py --search "heart attack"  # retrieval only, no generation
python scripts/query.py -v "sepsis management"   # show retrieved passages
```

API:

```bash
uvicorn medical_rag.api.main:app --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | device, models, Qdrant state |
| `POST /ask` | full pipeline: answer + citations + confidence + guardrails |
| `POST /search` | retrieval only, with per-channel scores — for tuning |
| `POST /expand` | query expansion in isolation — for tuning the lexicon |

Interactive docs at `http://localhost:8000/docs`.

---

## Tests

```bash
pytest
```

All tests run offline: no models, no network, no API keys. The chunker is tested
against a stub encoder, retrieval against a stub Qdrant.

---

## Two-machine workflow

The laptop is for code; the desktop has the GPU.

```bash
# laptop — write code, run tests, commit
pytest && git add -A && git commit -m "..." && git push

# desktop — pull and run the heavy work
git pull
python scripts/ingest.py --recreate
```

`.env`, `RAW_DATA/`, `data/processed/` and `.venv/` are all git-ignored, so each
machine keeps its own keys, PDFs, derived artefacts and environment. The Qdrant
collection is the shared state — index once, query from anywhere.

---

## Safety

This system answers from indexed medical literature and is **not** a clinical
decision tool. It refuses rather than guesses:

- **Emergencies** short-circuit before retrieval and return emergency-services guidance.
- **Confidence below 0.8** returns a refusal, not a low-quality answer.
- **Answers with unsupported doses** — any dose or interval not present verbatim in the retrieved sources — are blocked.
- **Every answer carries a disclaimer**, and the pipeline will not answer at all if the reranker is unavailable.
