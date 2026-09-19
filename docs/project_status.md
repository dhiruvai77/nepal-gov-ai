# NepalGov AI — Current Project Status

Last updated: 2026-09-19

## Project

**NepalGov AI — Evidence-Grounded Search and Question Answering over Nepal Government Documents**

Formal subtitle:

**Nepal Government Knowledge Intelligence — Multilingual RAG for Public-Sector Documents**

Objective:

> Design, implement, and evaluate a production-oriented multilingual RAG system capable of retrieving information from heterogeneous Government of Nepal documents and generating transparent, evidence-grounded answers with verifiable source attribution.

V1 languages:

- English
- Nepali

V1 domains:

- Constitution & Law
- Finance & Economy
- Health & Population
- Education

---

## Current Milestone

**Reranking implementation and evaluation**

First-stage retrieval has been implemented, evaluated, and its production architecture has been selected.

The current task is to integrate and evaluate:

`BAAI/bge-reranker-v2-m3`

using a hosted Hugging Face Text Embeddings Inference (TEI) reranking endpoint.

---

## Last Validated Repository State

Latest reranking infrastructure checkpoint:

`6876b97 — Add hosted multilingual reranking infrastructure`

Test suite:

`126 passed`

Development environment:

- Windows
- Python 3.12 virtual environment
- Windows CMD
- Qdrant via Docker
- Hosted Hugging Face inference
- Smart App Control remains enabled

Important:

- Do not disable Smart App Control.
- Avoid WSL unless explicitly requested.
- Do not expose `HF_TOKEN`.
- Do not commit hosted endpoint URLs or secrets.

---

## Corpus

Current development corpus:

1. `constitution_nepal_current_en`
2. `public_health_service_act_2075_en`
3. `compulsory_free_education_act_2075_en`
4. `economic_survey_2023_24_en`
5. `economic_survey_2081_82_ne`
6. `budget_speech_2025_26_en`

Total indexed chunks:

`2,276`

The Nepali Economic Survey requires forced OCR.

---

## Embeddings

Primary model:

`intfloat/multilingual-e5-large-instruct`

Embedding dimension:

`1024`

Query instruction:

`Retrieve relevant official Nepal government passages that answer the user's question.`

Query representation:

`Instruct: <instruction>\nQuery: <query>`

Passage representations stored in Qdrant:

- `dense`
  - original raw passage embedding
- `dense_contextual`
  - passage embedding enriched with document metadata
- `bm25`
  - sparse lexical representation

Raw passage text remains unchanged in the payload and is used for citations and downstream RAG stages.

---

## Contextual Embedding Representation

The contextual passage representation contains available metadata such as:

- document title
- organization
- document type
- section
- subsection
- article number
- article title
- original passage text

All `2,276` existing points were successfully backfilled with `dense_contextual`.

Raw dense vectors and BM25 vectors were preserved.

---

## Production First-Stage Retrieval Architecture

### Same-language retrieval

English query -> English evidence:

`dense_contextual + BM25 -> Reciprocal Rank Fusion`

Nepali query -> Nepali evidence:

`dense_contextual + BM25 -> Reciprocal Rank Fusion`

### Cross-lingual retrieval

English query -> Nepali evidence:

`dense_contextual only`

Nepali query -> English evidence:

`dense_contextual only`

BM25 is intentionally skipped for cross-lingual routing because lexical overlap is unreliable across English and Nepali.

### Raw dense representation

The original `dense` vector remains stored for:

- controlled baselines
- diagnostics
- experiments

It is not the selected production semantic representation.

---

## Retrieval Evaluation Dataset

Manually curated benchmark:

`30 questions`

Slices:

- 12 English -> English
- 6 Nepali -> Nepali
- 6 English -> Nepali
- 6 Nepali -> English

Metrics:

- Hit Rate
- Mean Reciprocal Rank
- Recall

Gold evidence was manually verified rather than inferred from retrieval output.

---

## Production-Style top_k=5 Results

Current contextual production retrieval:

| Metric | Raw baseline | Contextual | Improvement |
|---|---:|---:|---:|
| Hit@5 | 0.567 | 0.767 | +0.200 |
| MRR@5 | 0.434 | 0.519 | +0.084 |
| Recall@5 | 0.489 | 0.694 | +0.206 |

Language-pair Hit@5:

- English -> English: `0.667`
- Nepali -> Nepali: `0.833`
- English -> Nepali: `0.833`
- Nepali -> English: `0.833`

The largest gain came from Nepali -> English retrieval.

---

## Fixed-Depth Production Evaluation

Retrieval depth:

`20`

Metrics are calculated from the same fixed ranked result list.

### @5

- Hit Rate: `0.767`
- MRR: `0.522`
- Recall: `0.686`

### @10

- Hit Rate: `0.900`
- MRR: `0.541`
- Recall: `0.856`

### @20

- Hit Rate: `0.900`
- MRR: `0.541`
- Recall: `0.886`

Persistent primary-evidence misses at @20:

- `en_en_011`
- `en_ne_002`
- `ne_en_005`

These are useful targets for reranker evaluation.

---

## Retrieval Experiments Already Completed

Completed experiments include:

- raw dense retrieval
- BM25 retrieval
- dense + BM25 RRF
- contextualized gold-chunk diagnostic
- document-constrained dense retrieval
- cross-lingual English-reformulation diagnostic
- corpus-wide raw vs contextual dense A/B
- production-routing raw vs contextual A/B
- raw + contextual dual-dense fusion

Important result:

Equal-weight fusion of raw dense + contextual dense degraded overall retrieval quality, especially Nepali -> English.

Therefore:

**Do not add raw dense back into production RRF unless new evaluation evidence justifies it.**

---

## Reranker Architecture

Provider-independent abstraction:

`src/reranking/base.py`

Hosted TEI provider:

`src/reranking/hf_bge_reranker.py`

Tests:

- `tests/test_reranker_base.py`
- `tests/test_hf_bge_reranker.py`

Selected model:

`BAAI/bge-reranker-v2-m3`

Reranking input:

- user query
- original `chunk_text`

The contextual embedding representation is not passed to the reranker.

Reranking output preserves:

- original retrieval result
- reranker score
- original first-stage rank

---

## Hugging Face Reranker Endpoint

Inference engine:

**Text Embeddings Inference (TEI)**

Model:

`BAAI/bge-reranker-v2-m3`

Authentication:

Private

Autoscaling:

- minimum replicas: `0`
- maximum replicas: `1`
- scale-to-zero: enabled
- idle timeout: 15 minutes

Deployment history:

- 4 GB CPU instance -> failed with memory limit exceeded
- 8 GB CPU instance -> failed with memory limit exceeded
- 16 GB instance -> **running successfully**

The endpoint URL must be supplied locally through:

`HF_RERANKER_ENDPOINT_URL`

The Hugging Face token remains:

`HF_TOKEN`

Neither value should be committed to Git.

---

## Immediate Next Step

Do **not** run the full 30-question reranker benchmark yet.

First run a minimal live smoke test against the running TEI endpoint:

- one query
- two passages
- verify `/rerank` connectivity
- verify authentication
- verify returned `index`
- verify returned `score`
- verify ranking order
- verify provider parsing

Only after the smoke test succeeds:

1. retrieve top-20 first-stage candidates
2. rerank those candidates with `BAAI/bge-reranker-v2-m3`
3. evaluate retrieval vs reranked rankings
4. calculate @5, @10, and @20
5. inspect improvements and regressions by language pair
6. decide whether reranking becomes part of production retrieval

---

## Planned Pipeline

Current:

`Query -> contextual dense / BM25 -> RRF -> candidates`

Next:

`Query -> contextual dense / BM25 -> RRF -> BGE reranker -> context selection`

Later:

`Query`
`-> retrieval`
`-> reranking`
`-> context selection`
`-> GenerationService`
`-> GeminiProvider`
`-> grounded answer`
`-> citations / evidence`

---

## Remaining Major Work

After reranking:

1. Context selection
2. Generation service abstraction
3. Gemini generation provider
4. Grounded prompt construction
5. Citation/evidence generation
6. Insufficient-evidence handling
7. End-to-end RAG evaluation
8. FastAPI application
9. Streamlit interface
10. Structured logging
11. MLflow experiment tracking
12. Docker/Compose production integration
13. CI refinement
14. README and architecture documentation
15. Portfolio screenshots/demo

---

## Development Rules

Before significant work:

`git status --short`

After code changes:

`python -m pytest -q`

Before committing:

`git diff --cached --check`

Push explicitly:

`git push origin main`

Additional rules:

- Keep experiments separate from production behavior until evaluated.
- Do not tune retrieval architecture without measured evidence.
- Do not rerun expensive OCR unnecessarily.
- Do not rerun the full contextual backfill unless required.
- Preserve raw `dense` for baseline comparison.
- Preserve original passage text for citations.
- All substantive Python code should include useful comments explaining assumptions and non-obvious design choices.
- If repository state and project-status documentation disagree, stop and investigate before modifying code.