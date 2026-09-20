# NepalGov AI — Current Project Status

Last updated: 2026-09-20

## Purpose of This Document

This file is a project handoff and checkpoint document.

It exists primarily so development can continue accurately across ChatGPT conversations without reconstructing major architectural decisions, experiments, benchmark results, and remaining work from scratch.

Technical source of truth remains:

* current repository code
* automated tests
* evaluation outputs
* Git history

This status document should be updated at meaningful checkpoints and completed milestones rather than after every small code change.

---

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

**Context selection completed; generation service abstraction is the next milestone**

The production multilingual retrieval pipeline now includes:

* language-aware first-stage retrieval
* contextual multilingual E5 dense retrieval
* BM25 for same-language retrieval
* Reciprocal Rank Fusion
* hosted multilingual BGE reranking
* title-aware reranker input
* evaluated context selection

The selected production context-selection strategy is:

**Fixed top-5 reranked passages**

Context selection was evaluated using:

* fixed top-3
* fixed top-5
* fixed top-8
* token budget 1400
* token budget 1800
* token budget 2200
* adjacent-chunk-aware top-5

The fixed top-5 strategy was selected because it provides the strongest measured balance of evidence coverage, ranking quality, simplicity, and context cost.

Production context selection is implemented separately from retrieval and reranking through:

`src/context_selection/run_context_selection.py`

The next milestone is:

**Generation Service Abstraction**

---

## Last Validated Repository State

Last committed production reranking milestone:

`4c38b3b — Promote title-aware multilingual reranking`

Current Context Selection milestone is fully implemented and locally validated.

Latest full test suite:

`199 passed`

Latest whitespace/error validation:

`git diff --check` — clean

Context Selection changes are ready for final staging, commit, and push.

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
* Do not expose `HF_RERANKER_ENDPOINT_URL`.
* Do not commit hosted endpoint URLs or secrets.
* Qdrant development storage is persisted through Docker Compose.
* Hugging Face endpoint configuration remains local.
* Do not rerun expensive OCR or vector backfills unless a corpus or representation change actually requires it.

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

## Chunking

Current chunking implementation:

`src/chunking/chunk_documents.py`

Tokenizer:

`intfloat/multilingual-e5-large-instruct`

Baseline chunk parameters:

* target chunk size: approximately `400` tokens
* overlap: approximately `60` tokens

Chunks preserve provenance including:

* `chunk_id`
* `document_id`
* document title
* organization
* category
* document type
* language
* publication date
* source URL
* download URL
* retrieval timestamp
* page start
* page end
* section
* subsection
* article number
* article title
* `chunk_index`
* original `chunk_text`
* `token_count`
* extraction method

The current baseline is not yet a structure-aware parent/child chunking system.

Adjacent chunks may intentionally share content because of the configured overlap.

This became important during context-selection evaluation.

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

## Qdrant Payload Metadata

Qdrant stores the complete chunk payload.

Important fields already present in indexed payloads include:

* `chunk_index`
* `token_count`

These fields were originally not exposed through `RetrievalResult`.

During the Context Selection milestone, `RetrievalResult` was extended with optional:

```python
chunk_index: int | None = None
token_count: int | None = None
```

and `normalize_search_result()` now preserves those payload values.

No reingestion was required.

No OCR rerun was required.

No vector backfill was required.

These fields are now available downstream for:

* context selection
* token-cost measurement
* adjacency diagnostics
* future context-management strategies

They remain optional for backward compatibility.

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

This distinction is important when interpreting reranker and context-selection regressions because a primary chunk may move downward while another manually verified relevant passage remains highly ranked.

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

A reranker or context selector cannot recover evidence that does not enter the top-20 first-stage candidate pool.

---

## Historical Retrieval Metric Correction

An earlier project-status snapshot recorded fixed-depth MRR values of:

* `0.522` @5
* `0.541` @10
* `0.541` @20

A fresh official evaluation and the independent reranker benchmark both reproduced:

* `0.505` @5
* `0.524` @10
* `0.524` @20

Hit Rate, Recall, and persistent @20 misses remained unchanged.

Therefore the freshly reproduced values are the validated baseline.

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
* fixed-count context selection
* token-budget context selection
* adjacent-chunk-aware context selection

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

## Interpretation of Remaining Reranker @5 Regressions

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

That responsibility now belongs to the independent Context Selection layer.

---

## Context Selection Architecture

Provider-independent abstraction:

`src/context_selection/base.py`

Production baseline selector:

`src/context_selection/fixed_top_k.py`

Evaluated experimental selectors:

* `src/context_selection/token_budget.py`
* `src/context_selection/adjacent_chunk.py`

Production integration:

`src/context_selection/run_context_selection.py`

Evaluation:

`src/evaluation/compare_context_selection.py`

Tests include:

* `tests/test_context_selector_base.py`
* `tests/test_token_budget_context_selector.py`
* `tests/test_adjacent_chunk_context_selector.py`
* `tests/test_compare_context_selection.py`
* `tests/test_run_context_selection.py`

The selector consumes:

`RerankedResult`

It does not repeat retrieval or reranking.

The production reranker continues returning its full top-20 candidate pool.

Context selection is applied only afterward.

This keeps the pipeline stages independently testable:

```text
retrieval
->
reranking
->
context selection
->
generation
```

---

## Context Selection Evaluation Methodology

The same manually verified 30-question multilingual benchmark was used.

Every question was:

1. retrieved once to a fixed depth of 20
2. reranked once using title-aware BGE
3. cached as a full reranked candidate pool
4. passed through each context-selection strategy

Therefore selectors were compared against the exact same retrieval and reranking results.

The hosted BGE endpoint was **not** invoked independently for each selector.

Metrics included:

* Hit Rate
* MRR
* Recall
* average selected passages
* average source-passage tokens
* average adjacent same-document chunk pairs

This allows context quality and context cost to be evaluated together.

---

## Fixed Top-k Context Selection Results

### Fixed top-3

Overall:

* Hit: `0.800`
* MRR: `0.689`
* Recall: `0.761`
* average passages: `3.00`
* average tokens: `1011.1`
* average adjacent pairs: `0.40`

### Fixed top-5

Overall:

* Hit: `0.833`
* MRR: `0.697`
* Recall: `0.833`
* average passages: `5.00`
* average tokens: `1703.9`
* average adjacent pairs: `1.07`

Language slices:

| Slice    |   Hit |   MRR | Recall | Passages | Tokens | AdjPairs |
| -------- | ----: | ----: | -----: | -------: | -----: | -------: |
| EN -> EN | 0.833 | 0.639 |  0.792 |     5.00 | 1722.5 |     1.58 |
| NE -> NE | 1.000 | 0.833 |  1.000 |     5.00 | 1710.7 |     0.67 |
| EN -> NE | 0.833 | 0.708 |  0.861 |     5.00 | 1652.0 |     0.67 |
| NE -> EN | 0.667 | 0.667 |  0.722 |     5.00 | 1711.7 |     0.83 |

### Fixed top-8

Overall:

* Hit: `0.867`
* MRR: `0.703`
* Recall: `0.853`
* average passages: `8.00`
* average tokens: `2669.1`
* average adjacent pairs: `2.23`

Summary:

| Strategy |       Hit |       MRR |    Recall | Avg passages | Avg tokens | AdjPairs |
| -------- | --------: | --------: | --------: | -----------: | ---------: | -------: |
| Top-3    |     0.800 |     0.689 |     0.761 |         3.00 |     1011.1 |     0.40 |
| Top-5    | **0.833** | **0.697** | **0.833** |         5.00 |     1703.9 |     1.07 |
| Top-8    |     0.867 |     0.703 |     0.853 |         8.00 |     2669.1 |     2.23 |

Top-8 recovers additional evidence but requires approximately 57% more source-passage tokens than top-5.

The marginal gain from top-5 to top-8 is:

* Hit: `+0.034`
* MRR: `+0.006`
* Recall: `+0.020`

for approximately:

`+965`

average source-passage tokens.

Top-5 therefore provides the preferred cost/quality balance for the initial generation baseline.

---

## Token-Budget Context Selection Experiment

A strict-prefix token-budget selector was implemented.

The selector:

1. preserves reranker order
2. adds passages sequentially
3. stops when the next ranked passage would exceed the configured budget
4. does not allow lower-ranked shorter passages to leapfrog higher-ranked evidence

Evaluated budgets:

* `1400`
* `1800`
* `2200`

Results:

| Strategy    |   Hit |   MRR | Recall | Avg passages | Avg tokens | AdjPairs |
| ----------- | ----: | ----: | -----: | -----------: | ---------: | -------: |
| Budget 1400 | 0.800 | 0.689 |  0.761 |         3.63 |     1214.9 |     0.63 |
| Budget 1800 | 0.833 | 0.697 |  0.822 |         4.87 |     1634.4 |     1.07 |
| Budget 2200 | 0.833 | 0.697 |  0.842 |         5.97 |     2026.2 |     1.47 |

### Interpretation

Budget 1400:

* same Hit as top-3
* same MRR as top-3
* same Recall as top-3
* uses more average tokens than top-3

Therefore budget 1400 is dominated by fixed top-3.

Budget 1800:

* same Hit as fixed top-5
* same MRR as fixed top-5
* Recall decreases from `0.833` to `0.822`
* saves only about `70` average tokens

Therefore budget 1800 does not improve the quality/cost tradeoff.

Budget 2200:

* same Hit as fixed top-5
* same MRR as fixed top-5
* Recall improves slightly to `0.842`
* average token use rises to `2026.2`

The small recall improvement does not justify replacing the simpler top-5 production baseline.

Conclusion:

**The tested strict-prefix token-budget strategy is not selected for production.**

It remains in the repository as an evaluated experimental selector.

---

## Adjacent-Chunk Redundancy Experiment

Because baseline chunking uses approximately 60 tokens of overlap, adjacent chunks from the same document were investigated as a possible source of wasted generation context.

An adjacent-aware selector was implemented that:

* preserves reranker order
* rejects chunks directly adjacent to an already selected chunk from the same document
* continues further down the ranked list to fill the requested context count
* allows non-adjacent passages from the same document
* avoids assuming redundancy when `chunk_index` is missing

### Fixed top-5 baseline

* Hit: `0.833`
* MRR: `0.697`
* Recall: `0.833`
* average passages: `5.00`
* average tokens: `1703.9`
* average adjacent pairs: `1.07`

### Adjacent-aware top-5

* Hit: `0.767`
* MRR: `0.675`
* Recall: `0.756`
* average passages: `5.00`
* average tokens: `1702.2`
* average adjacent pairs: `0.00`

Language slices:

| Slice    |   Hit |   MRR | Recall | Passages | Tokens | AdjPairs |
| -------- | ----: | ----: | -----: | -------: | -----: | -------: |
| EN -> EN | 0.667 | 0.583 |  0.639 |     5.00 | 1737.8 |     0.00 |
| NE -> NE | 1.000 | 0.833 |  1.000 |     5.00 | 1729.8 |     0.00 |
| EN -> NE | 0.833 | 0.708 |  0.861 |     5.00 | 1611.3 |     0.00 |
| NE -> EN | 0.667 | 0.667 |  0.639 |     5.00 | 1694.2 |     0.00 |

### Interpretation

The selector successfully removed all measured adjacent chunk pairs:

```text
1.07
->
0.00
```

However, evidence quality fell materially:

```text
Hit:
0.833 -> 0.767

MRR:
0.697 -> 0.675

Recall:
0.833 -> 0.756
```

The largest degradation occurred in English -> English retrieval.

This indicates that adjacent chunks in the current corpus frequently contain complementary useful evidence rather than merely duplicated overlap.

Therefore:

**Do not suppress adjacent chunks in production context selection without new evaluation evidence.**

Simple adjacency is not a reliable proxy for useless redundancy in the current corpus.

---

## Production Context Selection Decision

Selected strategy:

**Fixed top-5**

Selected pipeline behavior:

```text
Reranked top-20 candidate pool
  ->
preserve reranker order
  ->
select first 5 passages
  ->
generation context
```

Reasons:

1. Strong evidence coverage.
2. Strong reranker MRR is preserved.
3. Substantially lower context cost than top-8.
4. Simpler and more deterministic than token budgeting.
5. Token-budget experiments did not improve the measured quality/cost tradeoff.
6. Aggressive adjacent-chunk suppression degraded evidence quality.
7. Original passage text and citation metadata remain unchanged.
8. Retrieval, reranking, and context selection remain independently testable.
9. Five passages provide a practical initial generation context without introducing an arbitrary large prompt.
10. The strategy is straightforward to reevaluate later when generation-level metrics become available.

Experimental context selectors remain in the repository for future controlled comparisons.

---

## Production Context Selection Integration

Production integration:

`src/context_selection/run_context_selection.py`

Default context count:

`5`

Production orchestration:

```text
run_context_selection()
  ->
run_reranked_retrieval(
    candidate_count=20,
    top_k=None
  )
  ->
FixedTopKContextSelector(top_k=5)
  ->
list[RerankedResult]
```

Important architectural boundary:

The reranker receives the complete first-stage top-20 pool.

The context selector receives the complete reranked pool.

The selector alone decides which passages continue to generation.

No earlier stage is truncated merely to implement generation context size.

Dependencies remain injectable for:

* tests
* FastAPI
* Streamlit
* future generation orchestration
* future experimental selectors

---

## Current Production Retrieval and Context Stack

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
           Fixed Top-5
         Context Selection
                   |
                   v
         Generation Service
              [NEXT]
```

---

## Planned RAG Pipeline

```text
Query
  ->
First-stage retrieval
  ->
BGE reranking
  ->
Fixed top-5 context selection
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

## Immediate Next Milestone

**Generation Service Abstraction**

Goal:

Create a provider-independent generation interface that accepts selected evidence context and produces a structured generation result without coupling the RAG pipeline directly to Gemini.

The generation layer should initially address:

* provider-independent abstraction
* clean request/result contracts
* selected evidence input
* answer text output
* language handling
* dependency injection for tests
* deterministic orchestration boundaries
* future support for grounded citations
* future insufficient-evidence behavior

The generic generation abstraction should not hard-code:

* Gemini-specific API behavior
* citation rendering
* final refusal policy
* retrieval logic
* reranking logic
* context-selection logic

Those concerns should remain separate pipeline layers.

---

## Generation Design Constraints

The next implementation should preserve the architecture already established.

Generation should consume the selected context rather than:

* rerunning retrieval
* rerunning reranking
* selecting new chunks independently
* mutating original passage text

Selected context currently consists of:

`list[RerankedResult]`

Each selected item retains:

* original passage text
* title
* organization
* document ID
* page provenance
* source URL
* language
* reranker score
* first-stage rank
* chunk ID
* chunk index
* token count

This information will later support:

* grounded prompt construction
* source labels
* inline citations
* evidence panels
* answer traceability
* groundedness evaluation

---

## Insufficient Evidence

Insufficient-evidence behavior has **not yet been implemented**.

It should not be conflated with context selection.

Current context selection simply chooses the strongest available reranked evidence.

Future work must determine when the system should:

* answer normally
* answer cautiously
* state that available evidence is insufficient
* avoid unsupported factual claims

Potential future signals may include:

* reranker scores
* evidence agreement
* source quality
* retrieval coverage
* generation groundedness
* citation support

No score threshold should be added to production without evaluation.

---

## Remaining Major Work

After context selection:

1. Generation service abstraction
2. Gemini generation provider
3. Grounded prompt construction
4. Citation/evidence generation
5. Insufficient-evidence handling
6. End-to-end RAG evaluation
7. FastAPI application
8. Streamlit interface
9. Structured logging
10. MLflow experiment tracking
11. Docker/Compose production integration
12. CI refinement
13. README and architecture documentation
14. Portfolio screenshots/demo

---

## End-to-End Evaluation Still Required

Retrieval and context selection are evaluated independently.

Full RAG evaluation remains future work.

Generation-level evaluation should eventually measure:

* answer relevance
* groundedness
* faithfulness
* factual consistency
* evidence utilization
* citation correctness
* citation completeness
* source attribution
* unsupported-claim rate
* insufficient-evidence behavior
* multilingual answer quality
* English -> English
* Nepali -> Nepali
* English -> Nepali
* Nepali -> English
* failure cases

Retrieval metrics alone must not be treated as proof of answer quality.

---

## Important Current Findings

### Retrieval

Selected first-stage architecture:

```text
same-language:
dense_contextual + BM25 -> RRF

cross-lingual:
dense_contextual only
```

Raw dense remains stored for diagnostics but is not in production retrieval.

### Reranking

Selected model:

`BAAI/bge-reranker-v2-m3`

Selected input:

```text
Document: <title>

<original chunk_text>
```

Title-aware reranking substantially improved MRR and corrected a documented source-confusion case.

### Context selection

Selected strategy:

**Fixed top-5**

Rejected as production defaults:

* strict-prefix token budgeting
* adjacent-chunk suppression

Reason:

Neither produced a better measured quality/cost tradeoff than fixed top-5.

---

## Current Test State

Latest completed full test run:

```text
199 passed in 1.96s
```

Latest validation:

```text
git diff --check
```

Result:

clean

The Context Selection milestone is ready for final staged validation, commit, and push.

---

## Git Workflow for Completing Current Milestone

Before commit:

```text
python -m pytest -q
git diff --check
git status --short
```

Stage the Context Selection milestone:

```text
git add docs/project_status.md
git add src/retrieval/dense_retriever.py
git add src/context_selection
git add src/evaluation/compare_context_selection.py
git add tests/test_dense_retriever.py
git add tests/test_adjacent_chunk_context_selector.py
git add tests/test_compare_context_selection.py
git add tests/test_context_selector_base.py
git add tests/test_run_context_selection.py
git add tests/test_token_budget_context_selector.py
```

Validate staged changes:

```text
git diff --cached --check
git diff --cached --stat
```

Commit:

```text
git commit -m "Add evaluated context selection pipeline"
```

Push:

```text
git push origin main
```

Then verify:

```text
git status --short
git log -1 --oneline
```

The milestone is complete only after the push succeeds and the working tree is clean.

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
* Do not rerun contextual-vector backfill unless required.
* Preserve raw `dense` for baseline comparison.
* Preserve `dense_contextual` as the selected semantic retrieval representation.
* Preserve original `chunk_text` for generation and citations.
* Do not mutate evidence text merely to construct model input.
* Keep first-stage retrieval independently callable.
* Keep reranking independently callable.
* Context selection must consume reranked results rather than repeat retrieval.
* Generation must consume selected evidence rather than repeat retrieval, reranking, or selection.
* Preserve source provenance throughout the pipeline.
* All substantive Python code should contain useful comments for assumptions and non-obvious behavior.
* Never commit `HF_TOKEN`.
* Never commit `HF_RERANKER_ENDPOINT_URL`.
* Do not introduce production thresholds, diversity rules, or deduplication heuristics without evaluation evidence.
* Treat repository code, tests, benchmark output, and Git history as the technical source of truth.
* Treat this file as a handoff/checkpoint document and update it at meaningful milestones rather than every small change.
