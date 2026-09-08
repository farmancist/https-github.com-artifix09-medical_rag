# Architecture

Why each stage is tuned the way it is, and what changed from the general-purpose
book RAG this was derived from.

| Stage | Book RAG | Medical RAG | Status |
|---|---|---|---|
| PDF parser | PyMuPDF | PyMuPDF + table extraction, page anchors | unchanged |
| Semantic chunker | threshold 95 · overlap 15% | **threshold 85 · overlap 20%** | modified |
| Embedding model | all-MiniLM-L6-v2 | **PubMedBERT (768-d)** | modified |
| Vector store | Qdrant HNSW m=16 · ef=100 | **Qdrant HNSW m=32 · ef=200** | modified |
| Query expansion | — | **layman → medical terms** | **new** |
| Hybrid search | BM25 + dense · RRF | BM25 + dense · RRF | unchanged |
| Reranker | optional | **mandatory cross-encoder** | modified |
| Guardrails | prompt-level · conf 0.5 | **conf ≥ 0.8 · mandatory disclaimer** | modified |
| Generation | Groq | Groq | unchanged |

---

## 1. PDF parser

`src/medical_rag/ingest/pdf_parser.py`

Medical references are dense two-column layouts with running headers, footers
and a lot of tables. Three things the parser does beyond extracting text:

- **Reading order.** Blocks are sorted into coarse vertical bands (30pt) and then
  left-to-right, so a two-column page is read column-wise rather than zig-zagging
  across the gutter.
- **Boilerplate removal.** Short lines that repeat on ≥50% of pages are running
  headers/footers and are dropped. Without this, "British National Formulary 85"
  appears in a thousand chunks and pollutes BM25.
- **Tables stay whole.** `find_tables()` output is converted to Markdown and kept
  as an atomic block. A dose table split across chunks is worse than useless —
  it separates a drug from its dose.

Every block carries its page number, because citations are mandatory downstream:
an answer that cannot name a page cannot pass the output guardrail.

## 2. Semantic chunker — 85 / 20%

`src/medical_rag/ingest/semantic_chunker.py`

Sentences are embedded, cosine distance between consecutive sentences is
computed, and a distance above the **85th percentile** becomes a chunk boundary.

**Why 85 and not 95.** Clinical text changes topic far more often per page than
prose does: indications → dosage → contraindications → interactions, sometimes
within a single paragraph. A 95th-percentile threshold cuts too rarely and glues
distinct clinical facts into one chunk. That is precisely the failure mode behind
confidently wrong dosage answers — the retriever returns a chunk that contains
the right drug and *a* dose, but the dose belongs to a different population.

**Why 20% overlap and not 15%.** The qualifier and the value are frequently in
adjacent sentences ("In renal impairment, reduce the dose." / "The usual adult
dose is 500 mg."). Higher overlap keeps a dose attached to the population that
qualifies it.

Two hard boundaries override the distance test: a section heading change, and a
table (kept atomic). Chunks are then held to 64–512 tokens.

## 3. Embeddings — PubMedBERT

`src/medical_rag/embeddings/encoder.py`

`NeuML/pubmedbert-base-embeddings`, 768-d, trained on PubMed. A general-purpose
model like MiniLM does not know that "MI", "myocardial infarction" and "heart
attack" are the same thing, or that "-emia" and "-uria" suffixes are
compositional. Vectors are L2-normalised at encode time, so cosine similarity is
a dot product and matches the collection's distance metric.

`DEVICE=auto` resolves to `cuda` on the desktop and `cpu` on the laptop; batch
size is scaled ×4 on GPU. Nothing else in the codebase imports torch.

## 4. Qdrant — m=32, ef_construct=200

`src/medical_rag/vectorstore/qdrant_store.py`

Two named vectors per point:

- `dense` — PubMedBERT, cosine, HNSW `m=32` / `ef_construct=200`
- `bm25` — sparse, with `Modifier.IDF` so Qdrant applies the IDF factor server-side

Higher `m` and `ef` than the book RAG buy recall at the cost of index size and
build time. That is the right trade here: a missed passage in a book RAG is an
annoyance, a missed contraindication is a hazard. `ef` at search time is
configurable separately (`QDRANT_HNSW_EF_SEARCH`).

**Why BM25 lives in Qdrant.** A local `rank_bm25` index would be a pickle that
has to be rebuilt on the GPU box and shipped to the laptop, and the two would
silently drift. Storing BM25 term weights as sparse vectors means one index and
one source of truth. The formula splits cleanly:

```
score = IDF(t) · [ tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl)) ]
        \_____/   \_________________________________________/
        Qdrant              embeddings/sparse.py
```

`avgdl` is computed exactly over the finished chunk set in stage 2 of ingestion,
rather than drifting as documents stream in.

## 5. Query expansion — the new component

`src/medical_rag/retrieval/query_expansion.py`

The corpus is written in clinical vocabulary; users type lay vocabulary. Dense
retrieval bridges part of that gap. **BM25 bridges none of it** — the clinical
token is simply absent from the query, so the lexical channel contributes
nothing without expansion. That is why this component is new here and absent
from the book RAG, where query and corpus share a vocabulary.

Two tiers:

1. **Lexicon** — a deterministic, auditable dictionary of ~200 layman → medical
   mappings (`resources/medical_lexicon.json`). Zero latency, no API call,
   longest-match-first so "high blood pressure" wins over "pressure".
2. **LLM** — a small fast Groq model (`llama-3.1-8b-instant`) rewrites phrasings
   the lexicon misses, constrained to return a JSON array and forbidden from
   adding conditions the user did not mention.

`hybrid` mode runs both. If Groq is unreachable the LLM tier degrades silently
to the lexicon result — expansion is an optimisation and must never fail a
request.

## 6. Hybrid search — RRF

`src/medical_rag/retrieval/hybrid.py`

```
RRF(chunk) = Σ over ranked lists L   weight(L) / (k + rank_L(chunk)),   k = 60
```

Fusion is done client-side rather than with Qdrant's server-side fusion, because
expansion produces N query variants and we therefore have up to **2N** ranked
lists (dense and sparse per variant) to fuse at once, not two. Variant lists are
weighted 0.7 against the original query's 1.0 — an expanded query is a hypothesis
about what the user meant, not the user's actual words. Searches run
concurrently; they are network-bound.

## 7. Reranker — mandatory

`src/medical_rag/retrieval/reranker.py`

`ncbi/MedCPT-Cross-Encoder`, domain-trained on PubMed query–article pairs.
Mandatory here for two reasons:

1. Bi-encoder retrieval on medical text returns topically correct but clinically
   wrong passages with high similarity — the paediatric and adult dose sections
   of a drug monograph look nearly identical to a bi-encoder. Only a
   cross-encoder, reading query and passage jointly, separates them.
2. The confidence gate needs a **calibrated** relevance signal. RRF scores are
   rank-derived and have no absolute meaning; a cross-encoder logit does.

Reranking scores against the *original* query, never an expanded variant:
expansion widens recall, but precision must be judged against what was actually
asked. Logits are squashed through a sigmoid so the gate has a stable [0,1] scale.

If the reranker fails, the pipeline **refuses to answer** rather than falling
back to un-reranked results.

## 8. Guardrails — confidence ≥ 0.8

`src/medical_rag/generation/guardrails.py`

Three gates.

### Input (before retrieval)

Emergencies short-circuit the entire pipeline — when the correct response is
"call for help now", retrieval latency is not acceptable. Also detected:
prompt-injection phrasing (warns, rules are enforced regardless) and requests for
a personal clinical decision (warns; the answer describes the literature rather
than directing care).

### Evidence (after reranking)

A weighted score over four components, because any one alone is gameable:

| Component | Weight | What it catches |
|---|---|---|
| `top_relevance` | 0.45 | is there a genuinely on-point passage at all? |
| `support` | 0.25 | one good hit can be a fluke; three agreeing hits rarely are |
| `margin` | 0.15 | a flat distribution means the retriever could not discriminate — the corpus probably does not cover the question |
| `agreement` | 0.15 | a strong #1 followed by noise |

Below `CONFIDENCE_THRESHOLD` (0.8) the pipeline **refuses and does not call the
LLM at all**. The book RAG's 0.5 accepts a coin-flip; an unsupported answer about
a book is a nuisance, an unsupported answer about a dose is a hazard.

### Output (after generation)

- Every `[n]` marker must resolve to a real retrieved source.
- An answer with no valid citation is blocked (unless the model correctly
  declined with "The provided sources do not cover this").
- **Numeric verification**: every dose, interval and threshold in the answer must
  appear verbatim in the retrieved context, whitespace-normalised. Numeric
  hallucination is the highest-severity failure mode here and one of the few
  catchable deterministically.
- The disclaimer is appended unconditionally, exactly once.

## 9. Generation — Groq

`src/medical_rag/generation/prompts.py`, `llm.py`

`llama-3.3-70b-versatile` at temperature 0.1. The system prompt is itself the
first guardrail layer: the model is an extractor and summariser of retrieved
context, not a clinician. It is instructed to reproduce doses exactly, preserve
population qualifiers, surface disagreement between sources, and say "The
provided sources do not cover this" rather than fill a gap from memory.

---

## Machine split

| | Laptop (CPU) | Desktop (CUDA) |
|---|---|---|
| Code, tests | ✅ | ✅ |
| Dry-run parse/chunk | ✅ | ✅ |
| Full ingestion | slow | ✅ |
| Serving / querying | ✅ | ✅ |

The same code runs on both; only `.env` differs, and `DEVICE=auto` resolves
itself. The Qdrant collection is the shared state: index once on the desktop,
query from either machine.
