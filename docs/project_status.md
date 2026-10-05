# NepalGov AI — Current Project Status

Last updated: 2026-10-05

## Purpose of This Document

This file is the handoff/checkpoint document for NepalGov AI.

It exists so development can resume accurately across conversations without reconstructing architectural decisions, benchmark results, deployment state, and remaining work from scratch.

Technical source of truth remains:

- current repository code
- automated tests
- persisted evaluation outputs
- Git history
- deployed infrastructure state

This status document is updated **only when explicitly requested by the user**. It is not updated on a fixed milestone cadence.

---

# Project

**NepalGov AI — Evidence-Grounded Search and Question Answering over Nepal Government Documents**

Formal subtitle:

**Nepal Government Knowledge Intelligence — Multilingual RAG for Public-Sector Documents**

Objective:

> Design, implement, evaluate, and deploy a production-oriented multilingual RAG system capable of retrieving information from heterogeneous Government of Nepal documents and generating transparent, evidence-grounded answers with verifiable source attribution.

V1 languages:

- English
- Nepali

V1 domains:

- Constitution & Law
- Finance & Economy
- Health & Population
- Education

---

# Current Project State

The full V1 RAG path is implemented and has progressed through evaluation, API, UI, containerization, and verified public cloud deployment.

Current major layers:

1. document ingestion and chunking
2. multilingual dense embeddings
3. contextual dense embeddings
4. sparse BM25 representations
5. multilingual retrieval
6. multilingual reranking
7. context selection
8. grounded generation
9. citation/evidence processing
10. structural evidence guard
11. deterministic and human evaluation
12. FastAPI application layer
13. readiness and operational checks
14. static web UI
15. Docker deployment baseline
16. Qdrant Cloud migration
17. Railway deployment and successful external HTTPS validation

Current production retrieval/generation behavior remains unchanged from the validated production baseline.

Latest confirmed pushed commit on `main` before this checkpoint update:

```text
d869d65 — Update deployment progress checkpoint
```

The user confirmed a clean working tree and `HEAD -> main, origin/main` at this commit on 2026-10-05. The commit containing this updated checkpoint has not yet been created or pushed.

Latest validated full local test suite:

```text
727 passed, 1 warning
```

Current phase:

**Milestone 8B — Cloud deployment and external production validation**

Milestone 8B external deployment validation **passed on 2026-10-05**. Final repository review, commit, and push of this checkpoint remain pending before administrative closure.

Public application: https://nepal-gov-ai-production.up.railway.app

The last full-suite result above is retained from Milestone 8A; the suite was not rerun during the configuration-only deployment fixes.

---

# Corpus

The indexed corpus contains six Government of Nepal documents and 2,276 chunks.

Documents:

```text
constitution_nepal_current_en
public_health_service_act_2075_en
compulsory_free_education_act_2075_en
economic_survey_2023_24_en
economic_survey_2081_82_ne
budget_speech_2025_26_en
```

The Nepali Economic Survey requires forced OCR.

Do not rerun forced OCR unless a corpus change or explicit reprocessing requirement makes it necessary.

Original extracted passage text remains canonical evidence.

---

# Embeddings and Qdrant Index

Embedding model:

```text
intfloat/multilingual-e5-large-instruct
```

Embedding dimension:

```text
1024
```

Collection:

```text
nepal_gov_documents
```

Named vectors:

```text
dense
dense_contextual
bm25
```

Production semantic representation:

```text
dense_contextual
```

Production collection count:

```text
points_count = 2276
indexed_vectors_count = 2276
```

Payload indexes:

```text
language
category
organization
document_id
document_type
```

Preservation rules:

- keep raw dense vectors
- keep contextual dense vectors
- keep BM25 sparse vectors
- keep original passage text
- do not re-ingest/backfill merely because a new experiment looks promising

---

# Production Retrieval Architecture

Same-language retrieval:

```text
dense_contextual + BM25 -> Reciprocal Rank Fusion
```

Cross-language retrieval:

```text
dense_contextual only
```

Cross-language BM25 is intentionally skipped because lexical overlap cannot be assumed across languages.

Candidate depth:

```text
20
```

Production reranker:

```text
BAAI/bge-reranker-v2-m3
```

Final selected context:

```text
top 5
```

Reranker receives the first-stage candidate pool. A passage outside that pool cannot be recovered by reranking.

---

# Generation Architecture

Package:

```text
google-genai==2.24.0
```

Model:

```text
gemini-3.8-flash
```

Transport:

```text
Gemini Interactions API
```

Production call:

```python
interaction = client.interactions.create(
    model=self.model_name,
    input=prompt,
)

answer_text = interaction.output_text
```

Runtime configuration:

```text
GEMINI_API_KEY
HF_TOKEN
HF_RERANKER_ENDPOINT_URL
```

Never commit or expose real credential values.

---

# Evidence Guard

The production guard is structural.

It rejects:

- missing selected context
- missing required citations
- invalid evidence IDs
- references to evidence that was not selected

It deliberately does not use an arbitrary retrieval-score threshold.

Structural guard state and semantic answer behavior are evaluated separately.

---

# Official Production RAG Benchmark

Artifact:

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

Questions:

```text
30
```

Language allocation:

```text
EN -> EN: 12
NE -> NE:  6
EN -> NE:  6
NE -> EN:  6
```

Notation means:

```text
query language -> evidence/document language
```

Answer language remains the query language.

## Structural Results

| Slice | N | Accept | SelHit | SelRec | CitHit | CitPrec | CitRec | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EN -> EN | 12 | 1.000 | 0.833 | 0.792 | 0.833 | 0.533 | 0.792 | 1.000 |
| NE -> NE | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.458 | 0.917 | 1.000 |
| EN -> NE | 6 | 1.000 | 0.833 | 0.861 | 0.833 | 0.347 | 0.778 | 1.000 |
| NE -> EN | 6 | 1.000 | 0.667 | 0.722 | 0.667 | 0.539 | 0.722 | 1.000 |
| **Overall** | **30** | **1.000** | **0.833** | **0.833** | **0.833** | **0.482** | **0.800** | **1.000** |

These are structural evidence-identity metrics, not claim-level factual-faithfulness metrics.

---

# Human Semantic Citation Evaluation

Human-reviewed claims:

```text
48/48
```

Headline results:

```text
Fully supported                    = 0.950
At least partially supported       = 1.000
Unsupported                        = 0.000
Required citation coverage         = 1.000
```

Individual citation evidence:

```text
Supported                          = 44
Partially supported                = 3
Unsupported                        = 1
Needs review                       = 0
```

Rates:

```text
Individual full support            = 0.917
Individual at least partial        = 0.979
Individual unsupported             = 0.021
```

---

# Automated Semantic Judge

Original human-reference agreement:

```text
Semantic exact agreement           = 0.979
Citation-requirement agreement     = 1.000
Individual-evidence agreement      = 0.958
```

Hard challenge set:

```text
Semantic exact                     = 0.667
Citation-requirement exact         = 0.667
Individual-evidence exact          = 0.667
```

The judge handles clear unsupported cases well but is over-decisive on deliberately ambiguous cases.

Human labels remain authoritative.

---

# Insufficient-Evidence Evaluation

Original 16-question structural benchmark:

```text
Decision accuracy                 = 0.938
Answer success                    = 1.000
Withholding success               = 0.875
False answer                      = 0.125
False withhold                    = 0.000
```

Original human response-level behavior:

```text
Behavior accuracy                 = 1.000
Answer delivery                   = 1.000
Abstention success                = 1.000
Unsafe substantive answer rate    = 0.000
Guard agreement                   = 0.938
```

Partial/mixed extension:

```text
Structural decision accuracy      = 1.000
Partial acceptance                = 1.000
Human behavior accuracy           = 1.000
Partial response success          = 1.000
Unsafe response rate              = 0.000
```

---

# Human Answer Quality

Artifact:

```text
data/evaluation/rag_runs/production_rag_v2_answer_quality_review_v1.jsonl
```

Reviewed:

```text
30/30
```

Results:

```text
Fully complete                    = 21/30 = 0.700
At least mostly complete          = 24/30 = 0.800
Incomplete                        =  6/30 = 0.200

Fully faithful                    = 28/30 = 0.933
No major factual error            = 30/30 = 1.000
Major factual issue               =  0/30 = 0.000

Strong answer                     = 20/30 = 0.667
Acceptable answer                 = 24/30 = 0.800
```

Main measured quality gap:

```text
answer completeness > major factual hallucination
```

A known localized numerical mismatch:

```text
generated = 33.8%
source    = 33.6%
```

was classified as a minor issue.

---

# Production Quality Checkpoints

V1:

```text
src/evaluation/production_quality_checkpoint.py
data/evaluation/production_quality_checkpoint_v1.json
```

V2:

```text
src/evaluation/production_quality_checkpoint_v2.py
data/evaluation/production_quality_checkpoint_v2.json
```

V3:

```text
production quality checkpoint V3 completed
commit: f56b5e5 — Add production quality checkpoint V3
```

No experimental retrieval or prompt change was promoted to production during the M1-M4 improvement cycle.

---

# Evaluation Improvement Cycle Completed

## Milestone 1 — Automatic NE -> EN Translation Benchmark

Commit:

```text
50d3de7 — Benchmark automatic NE to EN translation
```

Result:

- automatic translation/reformulation was benchmarked
- no production retrieval change was promoted

## Milestone 2 — Table-Aware Passage Representation

Commit:

```text
2baead9 — Benchmark table-aware passage representation
```

Result:

- table-aware representations were evaluated
- promising behavior remained experimental
- no production vector backfill was performed

## Milestone 3 — Answer Completeness Prompt

Commit:

```text
b078b92 — Evaluate answer completeness prompt
```

Result:

- completeness-oriented prompt intervention was evaluated
- no production prompt change was promoted

## Milestone 4 — Production Quality Checkpoint V3

Commit:

```text
f56b5e5 — Add production quality checkpoint V3
```

Validated full suite at this point:

```text
691 passed
```

Production system remained unchanged.

---

# Application Milestones

## Milestone 5 — FastAPI Application Layer

Commit:

```text
b6615d2 — Add FastAPI application layer
```

Added:

```text
GET  /health
POST /v1/answer
```

Features:

- injectable RAG pipeline factory
- Pydantic request validation
- stable RAG response mapping
- 422 validation handling
- 503 infrastructure handling
- startup/shutdown lifecycle

Validation:

```text
700 passed
```

Real local smoke test succeeded through the complete production RAG path.

---

## Milestone 6 — API Hardening

Commit:

```text
ec26f30 — Harden FastAPI application layer
```

Added:

```text
GET /health/ready
```

Operational features:

- structured source objects
- `X-Request-ID`
- `X-Process-Time-Ms`
- startup failure leaves HTTP service alive but not ready
- API-to-real-RAG integration coverage

Readiness checks:

```text
pipeline_initialized
gemini_configured
hf_token_configured
reranker_endpoint_configured
qdrant_reachable
qdrant_collection_ready
```

Validation:

```text
710 passed
```

Observed real answer latency:

```text
~55 seconds
```

Treat this as a measured optimization item; do not change retrieval blindly.

---

## Milestone 7 — Web UI

Commit:

```text
6c3d30b — Add NepalGov AI web interface
```

Implementation:

- static frontend served by FastAPI
- no Streamlit dependency
- English/Nepali question entry
- answer-language selector
- document filter
- evidence-language filter
- known-document scopes
- example questions
- readiness indicator
- loading timer/state
- accepted/withheld state
- structured source cards
- request/model/context/time diagnostics
- safe plain-text rendering
- CSP/basic security headers

Validation:

```text
717 passed
```

Visual browser smoke test succeeded.

---

# Milestone 8A — Container Deployment Baseline

Commit:

```text
4a92204 — Add container deployment baseline
```

Added/changed:

```text
Dockerfile
.dockerignore
.env.example
docker-compose.yml
requirements-runtime.txt
requirements.txt
.gitignore
src/indexing/qdrant_setup.py
src/retrieval/run_hybrid_retrieval.py
tests/test_deployment_readiness.py
```

Deployment configuration supports:

```text
QDRANT_URL
QDRANT_API_KEY
```

Local default remains:

```text
http://localhost:6333
```

The runtime image intentionally excludes local development-heavy packages such as:

```text
torch
sentence-transformers
transformers
pytest
```

The runtime explicitly includes:

```text
qdrant-client[fastembed]==1.19.0
```

because production same-language retrieval uses the local BM25 inference path.

Docker validation:

```text
docker compose build app        succeeded
docker compose up -d            succeeded
app container                   healthy
Qdrant container                preserved
```

Container readiness:

```text
HTTP 200
all readiness checks = true
```

Full containerized RAG smoke test:

```text
HTTP 200
request_id = docker-smoke-001
accepted = true
withheld = false
provider = gemini
model = gemini-3.8-flash
selected_context_count = 5
```

Observed processing time:

```text
53092.461 ms
```

Qdrant after container validation:

```text
status = green
points_count = 2276
indexed_vectors_count = 2276
```

Validated full suite:

```text
727 passed, 1 warning
```

Milestone 8A is complete and pushed.

---

# Local/Container Configuration

Local development uses persistent Windows user environment variables rather than a project `.env` file.

Required local variables:

```text
GEMINI_API_KEY
HF_TOKEN
HF_RERANKER_ENDPOINT_URL
```

Cloud migration helper variables were intentionally stored locally as:

```text
QDRANT_CLOUD_URL
QDRANT_CLOUD_API_KEY
```

The application itself expects:

```text
QDRANT_URL
QDRANT_API_KEY
```

Do not confuse the local migration-helper names with production app configuration names.

`.env.example` must contain no real credentials.

`.env` is ignored by Git even though the current workflow does not require one.

---

# Credential Incident and Current Secret Handling

During Docker validation, a `docker compose config` command expanded real credential values in pasted output.

Those credentials were treated as exposed.

Required response:

- rotate Gemini API key
- rotate Hugging Face token
- do not paste replacement values into chat
- prefer `docker compose config --quiet`
- store production credentials in platform secret/environment configuration

Current preferred secret handling:

```text
Windows persistent environment variables for local development
Railway service variables for deployment
```

Recommended Railway handling:

```text
GEMINI_API_KEY           -> sealed
HF_TOKEN                 -> sealed
HF_RERANKER_ENDPOINT_URL -> sealed
QDRANT_API_KEY           -> sealed
QDRANT_URL               -> normal variable
```

Never commit or expose real values.

---

# Milestone 8B — Cloud Deployment

Status:

```text
EXTERNAL VALIDATION PASSED — CHECKPOINT COMMIT/PUSH PENDING
```

## Qdrant Snapshot

A collection snapshot was created from local Qdrant v1.19.1.

Snapshot:

```text
nepal_gov_documents-6849673994721032-2026-09-28-09-17-03.snapshot
```

Size:

```text
61,651,456 bytes
```

SHA-256 checksum:

```text
fa933314d458486d11c533ed0ae8acb8b5a36abb04e7e11169d46fee492be981
```

Local backup directory:

```text
C:\Users\saman\nepal-gov-ai-backups
```

The backup is intentionally outside the Git repository.

## Qdrant Cloud Migration

Qdrant Cloud version:

```text
1.19.1
```

This exactly matches the local snapshot source version.

Snapshot restore response:

```text
result = true
status = ok
```

Cloud collection validation:

```text
status = green
points_count = 2276
indexed_vectors_count = 2276
segments_count = 1
```

Named vectors preserved:

```text
dense
dense_contextual
bm25
```

Payload indexes preserved:

```text
language
organization
document_type
document_id
category
```

No OCR, re-ingestion, or vector backfill was required.

## Railway

Railway project created:

```text
NepalGov AI
```

Railway service:

```text
nepal-gov-ai
```

Source:

```text
dhiruvai77/nepal-gov-ai
branch: main
```

Service configuration:

```text
Dockerfile: Dockerfile
health check: /health/ready
health-check timeout: 180 seconds
restart policy: ON_FAILURE
max retries: 3
```

Production variables added:

```text
GEMINI_API_KEY
HF_TOKEN
HF_RERANKER_ENDPOINT_URL
QDRANT_URL
QDRANT_API_KEY
```

The first Railway deployment failed because it ran before the required variables were applied:

```text
ValueError: GEMINI_API_KEY must be set for Gemini generation.
```

This was an expected configuration failure, not a Docker build failure.

The Railway service was observed Online. Supplied runtime logs showed successful application startup and Uvicorn listening on port 8080.

## Public HTTPS Validation — 2026-10-05

Public application:

```text
https://nepal-gov-ai-production.up.railway.app
```

Observed checks:

- `GET /health`: HTTP 200, `pipeline_ready = true`.
- `GET /health/ready`: HTTP 200, `status = ready`; all six checks true.
- `GET /openapi.json`: HTTP 200; request schema inspected before the smoke test.
- Supplied Railway access logs showed HTTP 200 for `/`, `/static/styles.css`, and `/static/app.js`. This is not a new visual UI review.
- User confirmed Qdrant Cloud collection `nepal_gov_documents` still reported 2,276 points after the successful answer test. The vector counts and indexes above were verified at migration time, not re-inspected on this date.

Successful production smoke request:

```json
{
  "query": "What rights relating to education are guaranteed by Article 31 of the Constitution of Nepal?",
  "answer_language": "en",
  "filters": {"document_id": "constitution_nepal_current_en"}
}
```

Observed response and headers:

```text
POST /v1/answer = HTTP 200
request_id = railway-public-smoke-20261005-003
X-Request-ID = railway-public-smoke-20261005-003
accepted = true
withheld = false
reason = null
provider = gemini
model = gemini-3.8-flash
selected_context_count = 5
sources_count = 2
X-Process-Time-Ms = 58824.674
client_observed_elapsed_seconds = 65.793
```

The answer covered access to basic education, compulsory/free basic and free secondary education, free higher education for eligible citizens, accessible education, and mother-tongue education. It returned inline E1/E2 citations and structured source metadata:

| Evidence | Document | Chunk | Pages |
|---|---|---|---|
| E1 | constitution_nepal_current_en | constitution_nepal_current_en_chunk_00023 | 15–16 |
| E2 | constitution_nepal_current_en | constitution_nepal_current_en_chunk_00024 | 16 |

Both source objects identified the Constitution of Nepal, Nepal Law Commission, English evidence, and the canonical source URL. This confirms successful execution of one deployed RAG request; it is not a rerun of the 30-question quality benchmark or a new claim-level faithfulness evaluation.

## Deployment Failures Resolved

1. Request `railway-public-smoke-20261005-001` returned HTTP 503. Runtime logs showed HTTP 401 from the hosted E5 embedding service (`Invalid username or password`). The user updated Railway's `HF_TOKEN` and redeployed. Subsequent requests progressed beyond embeddings and retrieval.
2. Request `railway-public-smoke-20261005-002` returned HTTP 503. Runtime logs showed HTTP 400 from the hosted BGE `/rerank` endpoint. A direct diagnostic using the current Railway token returned: `The endpoint is paused, ask a maintainer to restart it`. The user resumed the endpoint, after which request `003` succeeded.
3. The existing local Windows terminal's `HF_TOKEN` separately returned HTTP 401. A hidden-input diagnostic with the current Railway token reached the paused endpoint. Persistent local-token synchronization has not yet been confirmed; check it before the next local hosted-service run.

No application code, request payload, retrieval configuration, vectors, or generation prompt was changed to resolve these failures. The uploaded reranker implementation already used TEI's request format with `truncate = true`.

## Operational Findings

- Railway runs the application; the deployed setup uses Qdrant Cloud, hosted Hugging Face services, and Gemini. Local Docker/Qdrant need not run for public requests.
- Readiness passed even when the HF token was rejected or the reranker was paused. Its configuration checks do not prove hosted inference calls will succeed. A controlled end-to-end smoke test remains necessary for deployment validation.
- The current reranker exception preserves HTTP status and the cause chain but does not include the provider's response body. Safe, bounded provider-error diagnostics are a follow-up hardening candidate.
- A resumed reranker must remain available for public answer requests. Endpoint lifecycle and cost/availability policy should be documented during deployment hardening.
- Server processing was 58.8 seconds for the successful request; client-observed elapsed time was 65.8 seconds. This single measurement does not locate the performance bottleneck.

## Remaining Milestone 8B Closure Work

1. Install this updated checkpoint at `docs/project_status.md` in the local repository.
2. Review the diff and run `git diff --check`; this update is documentation-only. Run relevant repository-required documentation checks if present. If code changes are introduced, run the required full suite.
3. Stage only `docs/project_status.md`, inspect the staged diff, commit, and push `main`.
4. Confirm the final commit is pushed and the working tree is clean. The checkpoint commit hash is not yet known.

External deployment validation is complete. Milestone closure still requires the intended repository update to be committed and pushed. Railway resource metrics were not reviewed in this session.

---

# Current Measured Weaknesses

## Answer Completeness

```text
fully complete = 0.700
at least mostly complete = 0.800
```

## NE -> EN Retrieval

Production:

```text
selected primary hit     = 0.667
selected relevant recall = 0.722
```

## Table-Heavy Evidence

Current contextual representation can rank substantially worse than raw dense for some Economic Survey tables.

## Automated Judge Ambiguity

The semantic judge remains too decisive on intentional ambiguity.

## Numerical Fidelity

At least one localized numerical mismatch was observed.

## OCR Quality

Forced-OCR Nepali Economic Survey text contains occasional corruption.

## Citation Efficiency

Some individual citations are related but do not independently establish the associated claim.

## Latency

Observed production RAG latency:

```text
Earlier local/container processing: approximately 53–55 seconds
Public Railway server processing (2026-10-05): 58.825 seconds
Public client-observed elapsed time: 65.793 seconds
```

This is a deployment/performance issue to measure later, not a reason to modify the production retrieval stack without profiling.

---

# Architecture Principles

- retrieval remains independently callable
- reranking remains independently callable
- context selection remains independently callable
- generation never reruns retrieval
- Gemini transport does not own prompt policy
- prompt construction does not call Gemini
- citation processing does not trust model-generated source metadata
- invalid evidence references are never silently accepted
- no-selected-evidence cases skip generation
- original `chunk_text` remains canonical evidence
- raw dense vectors remain preserved
- contextual vectors remain separately preserved
- BM25 vectors remain preserved
- selected evidence order remains stable
- source provenance remains available end to end
- semantic evaluation remains separate from production generation
- human semantic labels remain the judge reference standard
- structural guard state is not equivalent to semantic answer behavior
- completeness is measured separately from factual fidelity
- experiments are not promoted without evaluation
- expensive preprocessing is not rerun without need
- persisted evaluation artifacts are preferred over unnecessary hosted reruns
- production changes are driven by measured failures
- production baselines remain immutable until a measured intervention is deliberately promoted

---

# Important Persisted Evaluation Artifacts

Official production benchmark:

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

Retrieval benchmark:

```text
data/evaluation/retrieval_questions.jsonl
```

Production semantic claims:

```text
data/evaluation/semantic/production_rag_v2_claims.jsonl
```

Human semantic reference:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_labeled.jsonl
```

Automated semantic judge:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_judge.jsonl
```

Hard semantic judge set:

```text
data/evaluation/semantic/semantic_judge_hard_cases_v1.jsonl
data/evaluation/semantic/semantic_judge_hard_cases_v1_judge.jsonl
```

Insufficient-evidence artifacts:

```text
data/evaluation/insufficient_evidence_questions.jsonl
data/evaluation/rag_runs/insufficient_evidence_v1.jsonl
data/evaluation/rag_runs/insufficient_evidence_v1_response_review.jsonl
```

Partial/mixed evidence:

```text
data/evaluation/insufficient_evidence_partial_mixed_v1.jsonl
data/evaluation/rag_runs/insufficient_evidence_partial_mixed_v1.jsonl
data/evaluation/rag_runs/insufficient_evidence_partial_mixed_v1_response_review.jsonl
```

Targeted NE -> EN retrieval:

```text
data/evaluation/retrieval_runs/ne_en_targeted_retrieval_v1.json
```

Answer-quality review:

```text
data/evaluation/rag_runs/production_rag_v2_answer_quality_review_v1.jsonl
```

Production checkpoints:

```text
data/evaluation/production_quality_checkpoint_v1.json
data/evaluation/production_quality_checkpoint_v2.json
```

---

# Expensive Operations

Do not rerun without a clear reason:

- forced OCR
- corpus ingestion
- dense embedding ingestion
- contextual-vector backfill
- full hosted retrieval benchmark
- full hosted reranker benchmark
- completed 30-question Gemini production benchmark
- completed semantic-judge calls

The current corpus and vectors remain valid.

---

# Development Environment

Primary development environment:

```text
Windows
Python 3.12 virtual environment
Windows CMD
Docker
Qdrant
Hugging Face hosted E5
Hugging Face hosted BGE reranker
Gemini Developer API
```

Local project path:

```text
C:\Users\saman\OneDrive\Desktop\Projects\nepal-gov-ai
```

Virtual environment:

```text
.venv
```

Avoid WSL unless explicitly requested.

Do not disable Smart App Control.

---

# Git and OneDrive Note

The working clone is under OneDrive.

Git automatic housekeeping previously caused deletion/retry problems under:

```text
.git/objects
```

Repository integrity was checked:

```text
git fsck --full
```

No corruption was found.

Local configuration:

```text
git config --local gc.auto 0
```

Keep `gc.auto 0`.

Do not routinely run:

```text
git gc
```

Do not manually delete:

```text
.git/objects/*
```

If stale locks appear:

1. verify no Git process is running
2. remove only the specific stale lock if necessary

A future fresh clone outside OneDrive may be safer.

---

# Git Workflow

Before significant work:

```text
git status --short
```

After code changes:

```text
python -m pytest -q
```

Before staging:

```text
git diff --check
```

Stage only intended files.

Avoid broad staging when unrelated files may exist.

Before commit:

```text
git diff --cached --check
git diff --cached --stat
git status --short
```

Push explicitly:

```text
git push origin main
```

Verify:

```text
git status --short
git log -1 --oneline
```

A milestone is complete only after its intended repository changes are committed and pushed.

---

# Git Milestone History

Evaluation/application/deployment milestones:

```text
50d3de7 — Benchmark automatic NE to EN translation
2baead9 — Benchmark table-aware passage representation
b078b92 — Evaluate answer completeness prompt
f56b5e5 — Add production quality checkpoint V3
b6615d2 — Add FastAPI application layer
ec26f30 — Harden FastAPI application layer
6c3d30b — Add NepalGov AI web interface
4a92204 — Add container deployment baseline
d869d65 — Update deployment progress checkpoint
```

Previous quality/evaluation milestones:

```text
f44c4be — Expand insufficient evidence benchmark
1d091e0 — Add semantic judge hard cases
d6fa4e8 — Benchmark NE to EN retrieval improvements
1e7619d — Add answer quality evaluation checkpoint
7a62a77 — Add automated semantic judge evaluation
c3d60b7 — Add insufficient evidence benchmark
d622873 — Add response abstention evaluation
ab91220 — Correct response review language notes
99e10df — Add production quality checkpoint
c2c6f08 — Update production quality checkpoint
2de6b86 — Update semantic evaluation checkpoint
```

Earlier architecture/benchmark milestones include:

```text
1bc404b — Record production RAG benchmark
e9ab5ec — Migrate Gemini generation to Interactions API
9d14f05 — Add resumable RAG evaluation runner
80058cf — Add deterministic RAG evaluation metrics
4691192 — Add end-to-end RAG orchestration
b67d42b — Add insufficient evidence handling
83af4f5 — Add evidence citation processing
bb387e6 — Add grounded generation prompt
99cb986 — Add Gemini generation provider
e0332a2 — Add generation service abstraction
e9642ee — Add evaluated context selection pipeline
4c38b3b — Promote title-aware multilingual reranking
```

---

# Current Production Decisions

Retrieval:

```text
same-language:
dense_contextual + BM25 -> RRF

cross-language:
dense_contextual only
```

Candidate depth:

```text
20
```

Reranker:

```text
BAAI/bge-reranker-v2-m3
```

Final context:

```text
top 5
```

Generation:

```text
Gemini
gemini-3.8-flash
Interactions API
```

Evidence IDs:

```text
E1
E2
E3
...
```

Guard:

```text
structural evidence/citation validation
```

Official production benchmark remains:

```text
production_rag_v2_interactions.jsonl
```

No experimental retrieval/representation/prompt intervention has replaced this baseline.

---

# Next Action When Work Resumes

Finish the documentation commit/push for Milestone 8B, then proceed to deployment hardening/documentation and portfolio polish/performance profiling.

Already verified on 2026-10-05:

- Public health/readiness endpoints returned HTTP 200.
- Public production answer request `railway-public-smoke-20261005-003` returned HTTP 200, an accepted answer, five selected passages, and two structured cited sources.
- User confirmed 2,276 points in Qdrant Cloud.
- The Railway HF token was updated and the paused reranker was resumed.

Immediate next steps:

1. Review and commit/push `docs/project_status.md`; verify clean status and the resulting pushed commit.
2. Before local hosted-service work, verify that Windows uses the current HF token without displaying its value.
3. Plan measured hardening: hosted-service availability diagnostics, safe provider-error reporting, endpoint lifecycle documentation, and stage-by-stage latency profiling.

Preserve the production retrieval and generation baseline. Do not repeat the snapshot migration, OCR, ingestion, or completed hosted benchmarks without a concrete need. Do not alter retrieval solely because the successful public request took about 59 seconds of server processing; profile first.
