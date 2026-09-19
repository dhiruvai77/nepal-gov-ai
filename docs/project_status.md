# NepalGov AI — Current Project Status

Last updated: 2026-09-19

## Project

**NepalGov AI — Evidence-Grounded Search and Question Answering over Nepal Government Documents**

Formal subtitle:

**Nepal Government Knowledge Intelligence — Multilingual RAG for Public-Sector Documents**

Objective:

> Design, implement, and evaluate a production-oriented multilingual RAG system capable of retrieving information from heterogeneous Government of Nepal documents and generating transparent, evidence-grounded answers with verifiable source attribution.

V1 languages:

* English
* Nepali

V1 domains:

* Constitution & Law
* Finance & Economy
* Health & Population
* Education

---

## Current Milestone

**Reranking completed; context selection is the next milestone**

First-stage multilingual retrieval has been implemented and evaluated.

Hosted multilingual reranking with:

`BAAI/bge-reranker-v2-m3`

has now been:

* implemented
* unit tested
* deployed through Hugging Face TEI
* smoke tested against the live endpoint
* evaluated on the 30-question multilingual retrieval benchmark
* compared using plain passage input and title-aware passage input
* integrated behind a production retrieval-and-reranking entry point

The selected production reranking representation is:

```text
Document: <document title>

<original chunk_text>
```

The document title is supplied only to the reranking model.

The original `chunk_text` stored in `RetrievalResult` remains unchanged for:

* citations
* evidence display
* context selection
* generation
* evaluation

The next milestone is:

**Context Selection**

---

## Last Validated Repository State

Previous committed reranking-infrastructure checkpoint:

`6876b97 — Add hosted multilingual reranking infrastructure`

Previous documentation checkpoint:

`593315d — Document current NepalGov AI project status`

Current reranking milestone includes new local changes that must be committed and pushed after final validation.

Last confirmed test suite before the final production-wrapper integration:

`132 passed`

A final full-suite rerun is required before committing the completed reranking milestone.

Development environment:

* Windows
* Python 3.12 virtual environment
* Windows CMD
* Qdrant via Docker
* Hosted Hugging Face inference
* Smart App Control remains enabled

Important:

* Do not disable Smart App Control.
* Avoid WSL unless explicitly requested.
* Do not expose `HF_TOKEN`.
* Do not commit hosted endpoint URLs or secrets.
* Qdrant development storage is persisted through Docker Compose.
* Hugging Face endpoint configuration remains local.

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

Do not rerun OCR or ingestion unless corpus changes require it.

---

## Embeddings

Primary model:

`intfloat/multilingual-e5-large-instruct`

Embedding dimension:

`1024`

Query instruction:

`Retrieve relevant official Nepal government passages that answer the user's question.`

Query representation:

```text
Instruct: <instruction>
Query: <query>
```

Passage representations stored in Qdrant:

* `dense`

  * original raw passage embedding
* `dense_contextual`

  * metadata-enriched passage embedding
* `bm25`

  * sparse lexical representation

Raw passage text remains unchanged in the Qdrant payload.

---

## Contextual Embedding Representation

The contextual dense passage representation contains available metadata such as:

* document title
* organization
* document type
* section
* subsection
* article number
* article title
* original passage text

All `2,276` indexed points were successfully backfilled with:

`dense_contextual`

Raw dense vectors and BM25 vectors were preserved.

The contextual representation is used for first-stage semantic retrieval.

It is **not** passed in full to the reranker.

---

## Production First-Stage Retrieval Architecture

### Same-language retrieval

English query -> English evidence:

```text
dense_contextual + BM25 -> Reciprocal Rank Fusion
```

Nepali query -> Nepali evidence:

```text
dense_contextual + BM25 -> Reciprocal Rank Fusion
```

### Cross-lingual retrieval

English query -> Nepali evidence:

```text
dense_contextual only
```

Nepali query -> English evidence:

```text
dense_contextual only
```

BM25 is intentionally skipped for cross-lingual routing because lexical overlap is unreliable across English and Nepali.

### Raw dense representation

The original:

`dense`

vector remains stored for:

* controlled baselines
* diagnostics
* experiments

It is not part of selected production retrieval.

### First-stage candidate depth

The selected reranking pipeline retrieves:

`20`

first-stage candidates before cross-encoder reranking.

This matches the candidate depth used during controlled reranker evaluation.

---

## Retrieval Evaluation Dataset

Manually curated benchmark:

`30 questions`

Slices:

* 12 English -> English
* 6 Nepali -> Nepali
* 6 English -> Nepali
* 6 Nepali -> English

Metrics:

* Hit Rate
* Mean Reciprocal Rank
* Recall

Gold evidence was manually verified rather than inferred from retrieval output.

Primary evidence is deliberately conservative.

`primary_relevant_chunk_ids` represents the strongest single-pass evidence.

`relevant_chunk_ids` contains the broader set of manually verified useful evidence.

This distinction is important when interpreting reranker regressions because a primary chunk may move downward while another manually verified relevant passage moves upward.

---

## Current Fixed-Depth First-Stage Baseline

Retrieval depth:

`20`

Each query is retrieved once.

Metrics at @5, @10, and @20 are calculated from the same ranked candidate list.

The current baseline was reproduced independently by:

* `src.evaluation.run_retrieval_evaluation`
* the reranker-comparison benchmark

### @5

* Hit Rate: `0.767`
* MRR: `0.505`
* Recall: `0.686`

Language slices:

| Slice    | Hit@5 | MRR@5 | Recall@5 |
| -------- | ----: | ----: | -------: |
| EN -> EN | 0.583 | 0.458 |    0.507 |
| NE -> NE | 1.000 | 0.672 |    0.889 |
| EN -> NE | 0.833 | 0.403 |    0.861 |
| NE -> EN | 0.833 | 0.533 |    0.667 |

### @10

* Hit Rate: `0.900`
* MRR: `0.524`
* Recall: `0.856`

### @20

* Hit Rate: `0.900`
* MRR: `0.524`
* Recall: `0.886`

Persistent primary-evidence misses at @20:

* `en_en_011`
* `en_ne_002`
* `ne_en_005`

These remain **first-stage retrieval misses**.

A reranker cannot recover evidence that does not enter its top-20 candidate pool.

### Historical metric correction

An earlier project-status snapshot recorded fixed-depth MRR values of:

* `0.522` @5
* `0.541` @10
* `0.541` @20

A fresh official evaluation and the independent reranker benchmark both reproduced:

* `0.505` @5
* `0.524` @10
* `0.524` @20

Hit Rate, Recall, and persistent @20 misses remained unchanged.

Therefore the freshly reproduced values are now the validated baseline.

The cause of the earlier small MRR difference was not established and should not be inferred without evidence.

---

## Retrieval Experiments Already Completed

Completed experiments include:

* raw dense retrieval
* BM25 retrieval
* dense + BM25 RRF
* contextualized gold-chunk diagnostic
* document-constrained dense retrieval
* cross-lingual English-reformulation diagnostic
* corpus-wide raw vs contextual dense A/B
* production-routing raw vs contextual A/B
* raw + contextual dual-dense fusion
* plain BGE reranking
* document-title-aware BGE reranking

Important first-stage result:

Equal-weight fusion of raw dense + contextual dense degraded retrieval quality, especially Nepali -> English.

Therefore:

**Do not add raw dense back into production RRF unless new evaluation evidence justifies it.**

---

## Reranker Architecture

Provider-independent abstraction:

`src/reranking/base.py`

Hosted TEI provider:

`src/reranking/hf_bge_reranker.py`

Production integration:

`src/retrieval/run_reranked_retrieval.py`

Evaluation:

* `src/evaluation/compare_reranked_retrieval.py`
* `src/evaluation/compare_reranker_inputs.py`

Tests include:

* `tests/test_reranker_base.py`
* `tests/test_hf_bge_reranker.py`
* `tests/test_compare_reranked_retrieval.py`
* `tests/test_compare_reranker_inputs.py`
* `tests/test_run_reranked_retrieval.py`

Selected model:

`BAAI/bge-reranker-v2-m3`

The reranker output preserves:

* original `RetrievalResult`
* original `chunk_text`
* citation metadata
* reranker score
* original first-stage rank

---

## Hugging Face Reranker Endpoint

Inference engine:

**Text Embeddings Inference (TEI)**

Model:

`BAAI/bge-reranker-v2-m3`

Authentication:

Private

Autoscaling:

* minimum replicas: `0`
* maximum replicas: `1`
* scale-to-zero: enabled
* idle timeout: 15 minutes

Deployment history:

* 4 GB CPU instance -> failed with memory limit exceeded
* 8 GB CPU instance -> failed with memory limit exceeded
* 16 GB instance -> running successfully

Local configuration:

`HF_RERANKER_ENDPOINT_URL`

Authentication token:

`HF_TOKEN`

Neither value is committed to Git.

---

## Live Reranker Smoke Test

A minimal two-passage live TEI smoke test was completed successfully.

The test verified:

* `/rerank` connectivity
* private endpoint authentication
* TEI response parsing
* returned `index`
* returned `score`
* candidate mapping
* ranking order
* provider integration

Observed example result:

```text
Relevant passage:
score ≈ 0.953

Irrelevant passage:
score ≈ 0.000016
```

The relevant passage was correctly ranked first.

Result:

**PASSED**

This cleared the live-inference gate before full benchmark execution.

---

## Plain BGE Reranking Benchmark

Initial reranker input:

```text
<original chunk_text>
```

The exact same first-stage top-20 candidates were reranked.

### @5

First-stage:

* Hit: `0.767`
* MRR: `0.505`
* Recall: `0.686`

Plain BGE:

* Hit: `0.833`
* MRR: `0.629`
* Recall: `0.783`

Changes:

* Hit: `+0.067`
* MRR: `+0.124`
* Recall: `+0.097`

Recovered primary-evidence questions @5:

* `en_en_002`
* `en_en_003`
* `en_en_009`

Plain-BGE primary regression @5:

* `ne_ne_002`

### @10

Plain BGE:

* Hit: `0.867`
* MRR: `0.634`
* Recall: `0.853`

A genuine source-sensitive regression was observed:

`en_en_007`

The question explicitly requested evidence from the Constitution of Nepal, but plain BGE promoted semantically related Public Health Service Act passages above the constitutional evidence.

The primary Constitution passage moved:

```text
baseline rank 7
-> plain BGE rank 12
```

This motivated a controlled source-aware reranker-input experiment.

---

## Title-Aware Reranker Experiment

Hypothesis:

> Adding only the document title may help BGE distinguish semantically similar passages from different government documents and better respect explicit source intent.

Experimental representation:

```text
Document: <document title>

<original chunk_text>
```

The full contextual embedding representation was intentionally **not** used.

The same first-stage top-20 candidate pools were used for:

1. baseline first-stage ranking
2. plain BGE
3. title-aware BGE

This kept the experiment controlled.

---

## Selected Title-Aware BGE Results

### @5

| Metric | First-stage | Plain BGE | Title-aware BGE |
| ------ | ----------: | --------: | --------------: |
| Hit    |       0.767 |     0.833 |       **0.833** |
| MRR    |       0.505 |     0.629 |       **0.697** |
| Recall |       0.686 |     0.783 |       **0.833** |

Language slices for title-aware BGE:

| Slice    | Hit@5 | MRR@5 | Recall@5 |
| -------- | ----: | ----: | -------: |
| EN -> EN | 0.833 | 0.639 |    0.792 |
| NE -> NE | 1.000 | 0.833 |    1.000 |
| EN -> NE | 0.833 | 0.708 |    0.861 |
| NE -> EN | 0.667 | 0.667 |    0.722 |

### @10

| Metric | First-stage | Plain BGE | Title-aware BGE |
| ------ | ----------: | --------: | --------------: |
| Hit    |       0.900 |     0.867 |       **0.900** |
| MRR    |       0.524 |     0.634 |       **0.706** |
| Recall |       0.856 |     0.853 |       **0.869** |

### @20

| Metric | First-stage | Plain BGE | Title-aware BGE |
| ------ | ----------: | --------: | --------------: |
| Hit    |       0.900 |     0.900 |       **0.900** |
| MRR    |       0.524 |     0.636 |       **0.706** |
| Recall |       0.886 |     0.886 |       **0.886** |

Because reranking receives the same fixed top-20 candidate pool:

* Hit@20 is expected to remain unchanged.
* Recall@20 is expected to remain unchanged.
* MRR measures whether relevant evidence is moved closer to the top.

Title-aware BGE improved overall MRR@20 from:

`0.524 -> 0.706`

without changing candidate membership.

---

## Source-Aware Reranking Findings

The most important source-sensitive case:

`en_en_007`

Question:

`What does the Constitution of Nepal guarantee regarding the right to health?`

Primary evidence rank:

```text
First-stage: 7
Plain BGE: 12
Title-aware BGE: 2
```

Adding the document title corrected the source-confusion failure.

Another diagnostic case:

`ne_ne_002`

Primary rank:

```text
First-stage: 2
Plain BGE: 7
Title-aware BGE: 2
```

The title-aware representation restored the primary passage while also ranking broader relevant evidence strongly.

---

## Interpretation of Remaining @5 Regressions

Title-aware BGE showed two primary-evidence Hit@5 regressions relative to plain BGE:

* `en_en_002`
* `ne_en_003`

Both were manually inspected.

### `en_en_002`

Primary Constitution evidence moved outside top 5.

However, another manually annotated relevant passage from:

**Compulsory and Free Education Act 2075**

remained at rank 1.

The query asks broadly:

`What does the law say about free and compulsory education?`

Therefore the top-ranked evidence remains directly useful even though the conservative primary-evidence metric records a miss.

### `ne_en_003`

The designated primary emergency-health passage moved to rank 6.

However, two other manually annotated relevant Public Health Service Act passages were ranked:

* rank 1
* rank 2

Therefore useful manually verified evidence remained at the top of the result list.

Conclusion:

These cases primarily expose the conservativeness of the single-primary-passage metric rather than demonstrating a clear user-facing retrieval failure.

The evaluation dataset should **not** be modified retrospectively merely to improve the reranker score.

---

## Production Reranking Decision

Selected model:

`BAAI/bge-reranker-v2-m3`

Selected passage representation:

```text
Document: <title>

<chunk_text>
```

Reasons:

1. Strong improvement over first-stage retrieval.
2. Higher MRR than plain BGE at every evaluated cutoff.
3. Higher Recall@5 than plain BGE.
4. Removes the plain-BGE Hit@10 regression.
5. Corrects documented source-confusion behavior.
6. Works across English and Nepali retrieval slices.
7. Preserves original passage text and citation metadata.
8. Requires only one additional metadata field rather than the complete contextual dense representation.

Plain passage-only BGE is **not** the selected production configuration.

---

## Production Reranking Pipeline

Production first-stage retrieval remains independently callable through:

`src/retrieval/run_hybrid_retrieval.py`

This is intentional.

Reranking is added as a separate production stage through:

`src/retrieval/run_reranked_retrieval.py`

Selected pipeline:

```text
Query
  ->
Language-aware first-stage retrieval
  ->
same-language:
dense_contextual + BM25 -> RRF
  OR
cross-lingual:
dense_contextual only
  ->
top-20 candidates
  ->
BAAI/bge-reranker-v2-m3
  ->
title-aware passage representation
  ->
reranked candidate pool
```

The reranking stage defaults to returning the full reranked candidate pool.

It does **not** decide how many passages should be sent to the LLM.

That responsibility belongs to the next milestone:

**Context Selection**

---

## Current Production Retrieval Stack

```text
User Query
   |
   v
Language Detection
   |
   +-------------------------------+
   |                               |
Same-language                 Cross-lingual
   |                               |
   v                               v
Contextual E5                  Contextual E5
   +                               |
BM25                               |
   |                               |
   v                               |
RRF                               |
   +---------------+---------------+
                   |
                   v
              Top-20
                   |
                   v
      Title-aware BGE Reranker
                   |
                   v
        Reranked Candidate Pool
                   |
                   v
          Context Selection
              [NEXT]
```

---

## Immediate Next Milestone

**Context Selection**

Goal:

Determine which reranked passages should actually be supplied to the generation model.

The context selector must address:

* final passage count
* token budget
* redundant/overlapping chunks
* repeated evidence from adjacent pages
* document/source diversity
* reranker scores
* preservation of strong evidence
* citation traceability
* deterministic behavior
* insufficient evidence

The context selector should consume:

`RerankedResult`

rather than re-running retrieval.

Questions to evaluate include:

* Should the system use a fixed top-k context?
* Should context be token-budget based?
* Should near-duplicate chunks be removed?
* Should multiple passages from one document be capped?
* Should low reranker-score candidates be excluded?
* Should evidence diversity be rewarded?
* How should adjacent chunks from the same source be handled?

Do not implement generation before context-selection behavior is defined and evaluated.

---

## Planned RAG Pipeline

```text
Query
  ->
First-stage retrieval
  ->
BGE reranking
  ->
Context selection
  ->
GenerationService
  ->
GeminiProvider
  ->
Grounded prompt
  ->
Grounded answer
  ->
Citations / evidence
```

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

Before committing:

```text
git diff --cached --check
```

After every completed milestone:

```text
git commit
git push origin main
```

A milestone is not considered complete until its validated changes have been committed and pushed.

Additional rules:

* Keep experiments separate from production behavior until evaluated.
* Do not tune retrieval architecture without measured evidence.
* Do not rerun expensive OCR unnecessarily.
* Do not rerun the contextual-vector backfill unless required.
* Preserve raw `dense` for baseline comparison.
* Preserve original `chunk_text` for citations.
* Do not mutate evidence text merely to construct model input.
* Keep first-stage retrieval independently callable.
* Context selection should consume reranked results rather than repeat retrieval.
* All substantive Python code should contain useful comments for assumptions and non-obvious behavior.
* Never commit `HF_TOKEN`.
* Never commit `HF_RERANKER_ENDPOINT_URL`.
* If repository state and project-status documentation disagree, stop and investigate before modifying production behavior.
