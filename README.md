# Medical RAG

[![CI](https://github.com/artifix09/medical_rag/actions/workflows/ci.yml/badge.svg)](https://github.com/artifix09/medical_rag/actions/workflows/ci.yml)

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
Groq LLM                  openai/gpt-oss-120b
```

See [docs/architecture.md](docs/architecture.md) for why each stage is tuned the
way it is.

---

## Quickstart

Four commands from a bare machine. No Docker, no database server, no cloud
account.

```bash
git clone https://github.com/artifix09/medical_rag.git
cd medical_rag
python setup_env.py                    # venv + deps + .env; asks for your Groq key
```

```powershell
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux / macOS
```

**Then copy your PDFs into `RAW_DATA/`.** They are git-ignored (large and
usually copyrighted), so this is the one step cloning cannot do for you.

```bash
python scripts/check_setup.py --models   # must print "0 failure(s)"
python scripts/ingest.py --recreate      # the long one-time job
python scripts/query.py "what are the contraindications for metformin?"
```

`setup_env.py` picks the CUDA torch build if it finds an NVIDIA GPU and the CPU
build otherwise. Both work — see [Hardware](#hardware) for what CPU costs you.

### Requirements

- Python 3.10+ (tested through 3.12; newer usually works)
- A **Groq API key** from [console.groq.com](https://console.groq.com) — the only
  mandatory credential. `setup_env.py` will prompt for it.
- ~2 GB of disk for model weights, plus the index.

### Configuration

Edit `.env`. Only **one** value is mandatory:

| Variable | What it is |
|---|---|
| `GROQ_API_KEY` (alias: `GROK`) | Groq API key, `gsk_...` — required |
| `QDRANT_URL` | leave **empty** for embedded mode, or set a cluster URL |
| `QDRANT_API_KEY` (alias: `QDRANT`) | only when `QDRANT_URL` is a remote cluster |

Everything else has a working default that matches the architecture.

> **Save `.env` as UTF-8 without a BOM.** A BOM prefixes the first variable
> invisibly, so it silently goes missing. The loader strips one if present, but
> older checkouts do not — keeping a `# comment` as line 1 makes it harmless.

### Hardware

Nothing in this project is trained. The models are used for inference only, so a
GPU changes how long the **one-time** ingest takes, not whether it works.

| | Ingest (~2,900 pages) | Per query |
|---|---|---|
| NVIDIA GPU | ~30 min | ~2 s |
| CPU only | a few hours, resumable | ~5–30 s, reranker-dominated |

AMD and Intel GPUs cannot accelerate PyTorch here (ROCm is Linux-only and drops
older cards; DirectML needs Python ≤3.12 and pinned torch), so those machines run
CPU-only. If queries feel slow, see [Tuning](#tuning).

### Qdrant — no server to run

Two modes, selected by whether `QDRANT_URL` is set. There is no container and
no daemon in either case.

| `QDRANT_URL` | Mode | Use when |
|---|---|---|
| empty | embedded on-disk store at `QDRANT_PATH` (`data/qdrant`) | single machine, zero setup |
| set | remote Qdrant / Qdrant Cloud | sharing one index between machines |

The embedded store is a local directory, so it is git-ignored and each machine
builds its own. Point `QDRANT_URL` at a cluster when you want the laptop to
query an index the GPU box built — that is the only way the two machines share
retrieval state.

### Verify

```bash
python scripts/check_setup.py --models
```

Checks dependencies, device, keys, `RAW_DATA/`, Qdrant connectivity, a live Groq
call, and (with `--models`) downloads and loads the embedding and reranker
weights. Every check runs independently, so one failure does not hide the rest.

---

## Ingestion

Put PDFs in `RAW_DATA/` (git-ignored), then:

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

To inspect PDF extraction with **no model and no Qdrant** — this runs on any
machine, including a laptop with no torch installed:

```bash
python scripts/ingest.py --parse-only --source "RAW_DATA/B09474-eng.pdf"
```

To also chunk (needs the embedding model, since chunking is embedding-based):

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

## Evaluation

After ingesting, use the golden question set to tune the knobs that cannot be
chosen from first principles — the confidence threshold, `top_k`, and whether
query expansion is earning its latency.

```bash
python scripts/evaluate.py --retrieval-only    # fast, no Groq calls
python scripts/evaluate.py                     # full pipeline + guardrail suite
python scripts/evaluate.py --no-expansion      # A/B the expansion stage
python scripts/evaluate.py --json runs/eval.json
```

It reports `hit@k`, MRR, mean confidence and refusal rate over
[eval/golden_questions.json](eval/golden_questions.json), then runs the
guardrail cases (emergencies must refuse; out-of-corpus questions must fail the
confidence gate). It exits non-zero if any guardrail case fails, so it can gate
a deploy.

Add your own questions to that file — it's a starting point, not a benchmark.
The most useful loop is `--retrieval-only` while tuning retrieval, since it
makes no LLM calls.

## Tests

```bash
pytest
```

132 tests, all offline: no models, no network, no API keys. The chunker is tested
against a stub encoder, and the vector store against a real embedded Qdrant, so
CI runs without downloading torch. `requirements-test.txt` is the minimal set.

---

## Tuning

Two knobs dominate query latency, and both are one-line `.env` changes. Measured
on a 4-thread CPU, reranking is ~95% of a query:

| Change | Effect |
|---|---|
| `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` | ~4× faster, loses medical-specific precision |
| `RERANK_CANDIDATES=20` (from 40) | ~2× faster, slightly lower recall |
| `CONFIDENCE_THRESHOLD` | lower → fewer refusals, more risk |

Change one, re-run `python scripts/evaluate.py --retrieval-only`, and compare
`hit_rate` and `mrr`. Decide from the numbers rather than from feel — on medical
content the precision you give up is not obvious by eye.

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

`.env`, `RAW_DATA/`, `data/` and `.venv/` are all git-ignored, so each machine
keeps its own keys, PDFs, derived artefacts and environment. Only code moves
through git.

In embedded mode each machine builds its own index, so the laptop would have to
repeat the ingestion. If you want to ingest once on the GPU box and query from
the laptop, set `QDRANT_URL` to a shared cluster on both — that is the only
piece of state the two machines can share.

---

## Safety

This system answers from indexed medical literature and is **not** a clinical
decision tool. It refuses rather than guesses:

- **Emergencies** short-circuit before retrieval and return emergency-services guidance.
- **Confidence below 0.8** returns a refusal, not a low-quality answer.
- **Answers with unsupported doses** — any dose or interval not present verbatim in the retrieved sources — are blocked.
- **Every answer carries a disclaimer**, and the pipeline will not answer at all if the reranker is unavailable.
