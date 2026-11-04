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
| `GROQ_MAX_TOKENS` | Maximum generated tokens per response (default: 32,768; model limits still apply) |
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

> **This machine specifically:** `DEVICE` defaults to `cpu` even though a
> DirectML path exists for the RX590 (`src/medical_rag/device.py`,
> `requirements-dml.txt`). This machine has been losing power unexpectedly
> 1–7×/day for at least a month, including days with no GPU load — a general
> hardware/power-delivery issue, not something DirectML load specifically
> causes. CPU-only stays the default as a precaution against the RX590's
> ~225W draw while that fault is undiagnosed. Treat `DEVICE=dml` as a
> deliberate, supervised one-off (e.g. `scripts/bench_device.py`, or a
> full-corpus rebuild where the speed is worth it) — resumable ingestion
> (see `scripts/ingest.py`) makes a mid-run reboot cost one document, not the
> run.

### Qdrant — no server to run

Two modes, selected by whether `QDRANT_URL` is set. There is no container and
no daemon in either case.

| `QDRANT_URL` | Mode | Use when |
|---|---|---|
| empty | embedded on-disk store at `QDRANT_PATH` (`data/qdrant`) | single machine, zero setup |
| set | remote Qdrant / Qdrant Cloud | sharing one index between machines |

The embedded store is a local directory, so it is git-ignored and each machine
builds its own. Point `QDRANT_URL` at a cluster when you want one machine to
query an index another machine built — that is the only way two machines share
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

To rebuild and re-index one document without disturbing other indexed sources:

```bash
python scripts/ingest.py --stage chunk --source "RAW_DATA/BNF.pdf" --force
python scripts/ingest.py --stage index --source "RAW_DATA/BNF.pdf" --force
```

Forced indexing replaces that document's existing Qdrant points before writing
the updated chunks. Do not combine source-filtered indexing with `--recreate`,
which rebuilds the entire collection.

To inspect PDF extraction without the embedding model or Qdrant, run:

```bash
python scripts/ingest.py --parse-only --source "RAW_DATA/B09474-eng.pdf"
```

`pymupdf-layout` is available as an opt-in parser path. Set
`PDF_LAYOUT_ANALYSIS=true` in `.env` to activate its ONNX model locally on CPU.
It is disabled by default: a distributed 100-page BNF comparison found more
table blocks with layout enabled, but also lower text similarity and several
garbled table extractions. Validate the report against the source before
enabling it for ingestion. No embedding model is needed for parse-only.

To compare layout analysis against PyMuPDF's line-based table detection across
an evenly distributed 100-page sample, save a JSON report, and inspect
per-page extraction differences:

```bash
python scripts/compare_pdf_layout.py "RAW_DATA/BNF.pdf"
python scripts/compare_pdf_layout.py "RAW_DATA/BNF.pdf" --all-pages
```

The report includes elapsed time, extracted text/table volume, per-page text
similarity, and table differences. These are comparison signals, not accuracy
scores; inspect differing pages against the source before changing parser
defaults. Use `--sample-pages 250` to choose another evenly distributed sample
size.

### BNF-specific extraction and chunking

A full or sampled parse of the British National Formulary uses the BNF profile. It keeps line geometry and reads each page right-column top-to-bottom, then left-column top-to-bottom, tracks chapter and subsection headings, monograph names, BNF field labels, and both printed and PDF page numbers. Gaps in a sampled page list reset carried context so unrelated pages cannot inherit a drug or field. The pale-blue `#E0E7F5` fill marks a dose panel when it contains `INDICATIONS AND DOSE`, or when an unlabelled panel contains both a route and a population dose. Same-colour safety callouts without dose evidence are excluded. Dose panels remain structured dose content, not tables. Appendix list tables with a blue `TABLE n` title band and an adjoining pale-blue body are captured as titled table blocks, including their independent list columns. Drug-interaction appendix pages carry an `INTERACTIONS` field.

The BNF chunker uses those structural boundaries and a token budget without sentence embeddings. Other documents retain the semantic-embedding chunker. BNF chunks are split at changes in drug/entity, field, dose indication, route, and side-effect frequency. Oversized tables are split by rows; a continuation dose row is not promoted to a repeated header, and page/entity metadata stays on each split chunk. Chunks carry source path, printed page, PDF page range, hierarchy, entity, and field metadata. The `/search` endpoint accepts exact `entity` and `field` filters.

This profile is specific to this BNF layout. Review parse-only output from representative pages before rebuilding chunks, then force chunking and indexing for the BNF so the stored chunks and Qdrant points use the new metadata. Other documents use the generic parser and chunker until a matching document-specific profile is added.
To preview generic parsing and chunking (non-BNF documents use the embedding model):

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
python scripts/evaluate.py --source OHCM --retrieval-only --debug-source --show-misses
```

It reports `hit@k`, MRR, mean confidence and refusal rate over
[eval/golden_questions.json](eval/golden_questions.json), then runs the
guardrail cases (emergencies must refuse; out-of-corpus questions must fail the
confidence gate). It exits non-zero if any guardrail case fails, so it can gate
a deploy.

Add your own questions to that file — it's a starting point, not a benchmark.
The most useful loop is `--retrieval-only` while tuning retrieval, since it
makes no LLM calls. `--source` selects cases tagged only with that source and filters
retrieval to matching indexed documents in the manifest. `--debug-source`
shows the matching documents and their indexing status; `--show-misses` prints
up to three retrieved evidence snippets per missed retrieval case. Results
also include metrics by test category and a `no_source_chunks` count.

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

Only code moves through git. `.env`, `RAW_DATA/`, `data/` and `.venv/` are all
git-ignored, so each machine keeps its own keys, PDFs, index and environment.

```bash
# wherever you write code
pytest && git add -A && git commit -m "..." && git push

# wherever you run it
git pull
python scripts/ingest.py --recreate
```

Because the embedded store is a local directory, **each machine builds its own
index** — the ingest does not transfer. To ingest once and query from several
machines, set the same `QDRANT_URL` on all of them; a shared cluster is the only
piece of state they can actually share.

---

## Safety

This system answers from indexed medical literature and is **not** a clinical
decision tool. It refuses rather than guesses:

- **Emergencies** short-circuit before retrieval and return emergency-services guidance.
- **Confidence below 0.8** returns a refusal, not a low-quality answer.
- **Answers with unsupported doses** — any dose or interval not present verbatim in the retrieved sources — are blocked.
- **Every answer carries a disclaimer**, and the pipeline will not answer at all if the reranker is unavailable.
