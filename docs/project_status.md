# NepalGov AI — Current Project Status

Last updated: 2026-09-20

## Purpose of This Document

This file is the project handoff and checkpoint document for NepalGov AI.

It exists so development can continue accurately across ChatGPT conversations without reconstructing architectural decisions, benchmark results, implementation history, and remaining work from scratch.

Technical source of truth remains:

* current repository code
* automated tests
* evaluation outputs
* Git history

Project-status updates are performed after every **four completed milestones**, rather than after every individual milestone.

Between checkpoints:

* Git history is the milestone record
* tests are the validation record
* repository code is the architecture source of truth
* persisted benchmark outputs are the evaluation record

---

# Project

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

# Current Project State

The complete V1 RAG execution path is implemented and has now been evaluated end to end on the existing 30-question multilingual benchmark.

Completed major layers:

1. document ingestion and chunking
2. multilingual dense embeddings
3. sparse BM25 representations
4. multilingual hybrid retrieval
5. multilingual reranking
6. context selection
7. provider-independent generation abstraction
8. Gemini generation provider
9. grounded evidence prompt
10. citation/evidence processing
11. insufficient-evidence handling
12. application-level RAG orchestration
13. deterministic RAG evaluation metrics
14. resumable production RAG benchmark runner
15. Gemini Interactions API transport
16. complete 30-question production RAG benchmark

Current validated test suite:

`325 passed`

Latest committed milestone on `main`:

`e9ab5ec — Migrate Gemini generation to Interactions API`

Current completed but not yet committed checkpoint milestone:

**Production 30-question RAG Benchmark**

Official benchmark output:

`data/evaluation/rag_runs/production_rag_v2_interactions.jsonl`

Rows:

`30`

Next major development direction:

**Claim-level citation correctness and semantic faithfulness evaluation**

---

# Latest Four-Milestone Checkpoint

This checkpoint captures the following four completed milestones:

1. Deterministic RAG Evaluation Metrics
2. Production 30-question RAG Evaluation Runner
3. Gemini Interactions API Migration
4. Complete Production RAG Benchmark

---

# Milestone 1 — Deterministic RAG Evaluation Metrics

Committed as:

`80058cf — Add deterministic RAG evaluation metrics`

Implementation:

`src/evaluation/rag_evaluator.py`

Supporting orchestration change:

`src/rag/pipeline.py`

Tests:

`tests/test_rag_evaluator.py`

`tests/test_rag_pipeline.py`

The evaluator reuses manually verified retrieval gold annotations and compares them against:

* selected RAG context
* cited evidence
* citation reference validity
* acceptance/withholding behavior

Per-query metrics:

```python
RAGQueryMetrics(
    question_id,
    accepted,
    withheld,
    selected_primary_hit,
    selected_relevant_recall,
    cited_primary_hit,
    cited_relevant_precision,
    cited_relevant_recall,
    valid_reference_ratio,
    selected_context_count,
    valid_citation_count,
    invalid_citation_count,
    rendered_source_count,
)
```

Aggregate metrics include:

* acceptance rate
* withholding rate
* selected primary hit rate
* selected relevant recall
* cited primary hit rate
* cited relevant precision
* cited relevant recall
* valid-reference ratio
* average selected context count
* average valid citation count
* average invalid citation count
* average rendered source count

The existing retrieval benchmark uses Qdrant `point_id` values as the gold passage identifiers.

The RAG evaluator therefore intentionally evaluates:

```python
item.result.point_id
```

rather than introducing a different passage identity.

## Important limitation

These deterministic metrics measure:

* evidence coverage
* citation selection
* citation-reference validity
* structural grounding behavior

They do **not** establish that every generated claim is semantically entailed by its cited passage.

For example:

```text
claim [E1]
```

can contain a structurally valid citation even when the relationship between the claim and passage still requires semantic evaluation.

Claim-level citation correctness remains a separate evaluation problem.

---

# Milestone 2 — Production RAG Evaluation Runner

Committed as:

`9d14f05 — Add resumable RAG evaluation runner`

Implementation:

`src/evaluation/run_rag_evaluation.py`

Tests:

`tests/test_run_rag_evaluation.py`

The runner executes the existing 30-question multilingual retrieval benchmark through the complete production RAG pipeline.

Execution path:

```text
Evaluation Question
       |
       v
Production Retrieval
       |
       v
Production Reranking
       |
       v
Fixed Top-5 Context
       |
       v
Grounded Gemini Generation
       |
       v
Citation Processing
       |
       v
Evidence Guard
       |
       v
Deterministic RAG Metrics
       |
       v
Persistent JSONL Result
```

The runner supports:

* full production execution
* immediate persistence after each completed question
* safe resume after interruption
* validation of persisted rows before reuse
* duplicate-question protection
* benchmark/run configuration validation
* dataset/gold annotation validation
* `--limit` for controlled hosted execution
* `--reset` for intentional clean reruns
* language-pair metric summaries
* overall metric summaries

Production language behavior:

```text
query_language -> answer language
target_language -> retrieval corpus filter
```

Therefore:

English -> Nepali evidence:

```text
query: English
retrieval corpus: Nepali
answer: English
```

Nepali -> English evidence:

```text
query: Nepali
retrieval corpus: English
answer: Nepali
```

This distinction is intentional.

The runner persists each successfully completed row immediately.

This behavior was validated in practice when external Gemini failures interrupted evaluation: previously completed rows remained reusable and were not regenerated.

---

# Milestone 3 — Gemini Interactions API Migration

Committed as:

`e9ab5ec — Migrate Gemini generation to Interactions API`

Implementation:

`src/generation/gemini_service.py`

Tests:

`tests/test_gemini_generation_service.py`

Production model remains:

`gemini-3.8-flash`

The public generation abstraction did not change.

Existing interface:

```python
GenerationService.generate(
    request: GenerationRequest,
) -> GenerationResult
```

remains intact.

Only the Gemini SDK transport changed.

Previous path:

```python
client.models.generate_content(
    model=...,
    contents=...,
)
```

Current production path:

```python
client.interactions.create(
    model=...,
    input=...,
)
```

Response text is read from:

```python
interaction.output_text
```

## Reason for migration

During production benchmark validation, repeated:

```text
503 UNAVAILABLE
```

errors occurred through `Models.generate_content()` across multiple Gemini 3 Flash models.

Diagnostics established that:

* API authentication worked
* model discovery worked
* `count_tokens` worked
* the configured Gemini models were visible
* Gemini Interactions API generation worked
* `Models.generate_content()` remained unavailable

A direct Interactions diagnostic using:

`gemini-3.8-flash`

returned successfully.

Therefore the provider transport was migrated without changing:

* retrieval
* reranking
* context selection
* prompt construction
* citation processing
* evidence guarding
* application orchestration
* production model identity

The earlier automatic-function-calling warning associated with direct `Models.generate_content()` also disappeared after migration.

---

# Milestone 4 — Complete Production RAG Benchmark

Official benchmark:

`data/evaluation/rag_runs/production_rag_v2_interactions.jsonl`

Run configuration:

`production-rag-v2-interactions`

Questions:

`30/30`

Language slices:

* 12 English -> English
* 6 Nepali -> Nepali
* 6 English -> Nepali
* 6 Nepali -> English

Production stack evaluated:

```text
contextual multilingual E5
        +
BM25 for same-language retrieval
        |
        v
RRF where applicable
        |
        v
Top-20 candidates
        |
        v
BAAI/bge-reranker-v2-m3
        |
        v
Fixed Top-5 context
        |
        v
GroundedPromptBuilder
        |
        v
gemini-3.8-flash
Interactions API
        |
        v
Citation Processing
        |
        v
Evidence Guard
```

## Final benchmark results

| Slice       |      N |    Accept |    SelHit |    SelRec |    CitHit |   CitPrec |    CitRec |     Valid |
| ----------- | -----: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| EN -> EN    |     12 |     1.000 |     0.833 |     0.792 |     0.833 |     0.533 |     0.792 |     1.000 |
| NE -> NE    |      6 |     1.000 |     1.000 |     1.000 |     1.000 |     0.458 |     0.917 |     1.000 |
| EN -> NE    |      6 |     1.000 |     0.833 |     0.861 |     0.833 |     0.347 |     0.778 |     1.000 |
| NE -> EN    |      6 |     1.000 |     0.667 |     0.722 |     0.667 |     0.539 |     0.722 |     1.000 |
| **Overall** | **30** | **1.000** | **0.833** | **0.833** | **0.833** | **0.482** | **0.800** | **1.000** |

Metric meanings:

```text
Accept
    accepted-answer rate

SelHit
    selected context contains at least one primary gold passage

SelRec
    recall of gold-relevant passages in selected context

CitHit
    cited passages contain at least one primary gold passage

CitPrec
    fraction of cited passage identities that appear in the
    benchmark's broader relevant gold set

CitRec
    recall of benchmark-relevant passages among cited evidence

Valid
    fraction of generated evidence references that resolve to
    real selected evidence IDs
```

---

# Benchmark Interpretation

## Structural citation behavior

Overall:

```text
Acceptance rate = 1.000
Valid reference ratio = 1.000
```

All 30 generated answers:

* passed the current evidence guard
* contained valid selected-context evidence references
* contained no unresolved evidence IDs

This validates the current structural citation contract.

It does **not** prove semantic claim-level faithfulness.

---

## Selected evidence coverage

Overall:

```text
Selected primary hit = 0.833
Selected relevant recall = 0.833
```

The fixed top-5 context therefore retained strong evidence coverage across the full benchmark.

The result is consistent with the previous context-selection benchmark, where fixed top-5 also achieved:

```text
Hit = 0.833
Recall = 0.833
```

The generation benchmark therefore did not expose an unexpected deterioration in selected-context coverage.

---

## Citation recall

Overall:

```text
Cited relevant recall = 0.800
```

Compared with:

```text
Selected relevant recall = 0.833
```

the model cited most, but not all, gold-relevant evidence available in its selected context.

The gap is relatively small at aggregate level but should be investigated per question.

---

## Citation precision

Overall deterministic citation precision:

```text
0.482
```

This is the clearest next evaluation signal.

It means that, under the existing manually annotated gold passage set, approximately 48.2% of distinct cited passage identities are classified as benchmark-relevant on average.

This metric must be interpreted cautiously.

It does **not** establish that every citation outside the current gold set is incorrect.

Possible explanations include:

* the model cites additional supporting passages
* a cited passage is useful but absent from the manually annotated gold set
* overlapping chunks contain valid support not separately annotated
* the model over-cites context passages
* some citations are only partially relevant
* some citations may genuinely fail to support the associated claim

Therefore deterministic citation precision should be treated as a diagnostic trigger for semantic evaluation rather than a final groundedness score.

---

## English -> English

Results:

```text
SelHit  = 0.833
SelRec  = 0.792
CitHit  = 0.833
CitPrec = 0.533
CitRec  = 0.792
```

The generation stage retained the selected-context evidence recall in its citation behavior.

---

## Nepali -> Nepali

Results:

```text
SelHit  = 1.000
SelRec  = 1.000
CitHit  = 1.000
CitPrec = 0.458
CitRec  = 0.917
```

This is the strongest selected-context slice.

All primary evidence was selected.

The model cited nearly all broader gold evidence but also cited passages outside the current gold annotations.

This slice is a useful candidate for examining whether low deterministic citation precision represents:

* legitimate supplementary evidence
* incomplete gold annotations
* unnecessary over-citation

---

## English -> Nepali

Results:

```text
SelHit  = 0.833
SelRec  = 0.861
CitHit  = 0.833
CitPrec = 0.347
CitRec  = 0.778
```

This slice has the lowest deterministic citation precision.

It should receive special attention in claim-level semantic evaluation.

The result may reflect:

* cross-lingual evidence interpretation
* broader citation behavior
* incomplete relevant-passage annotations
* overlapping Nepali evidence chunks
* genuine citation-quality problems

No conclusion should be made without semantic inspection.

---

## Nepali -> English

Results:

```text
SelHit  = 0.667
SelRec  = 0.722
CitHit  = 0.667
CitPrec = 0.539
CitRec  = 0.722
```

This is the weakest selected-evidence slice.

The main limitation appears upstream of generation because selected-context metrics are already lower than the other language pairs.

This makes Nepali-query -> English-document retrieval an important future retrieval-improvement target.

---

# Evaluation Caveats

The current 30-question benchmark was originally designed primarily for retrieval evaluation.

It contains manually verified relevant evidence, but it does not yet contain:

* reference answers
* per-claim support labels
* unsupported-claim labels
* citation entailment labels
* answer completeness labels
* deliberately unanswerable questions
* expected withholding labels

Therefore current results should not be described as a complete factual-accuracy benchmark.

In particular:

```text
Acceptance = 1.000
```

does not validate insufficient-evidence handling because the benchmark questions are designed around retrievable government evidence.

A separate unanswerable/insufficient-evidence benchmark is required.

---

# Corpus

Current development corpus contains six official Nepal government documents:

1. `constitution_nepal_current_en`
2. `public_health_service_act_2075_en`
3. `compulsory_free_education_act_2075_en`
4. `economic_survey_2023_24_en`
5. `economic_survey_2081_82_ne`
6. `budget_speech_2025_26_en`

Total indexed chunks:

`2,276`

The Nepali Economic Survey requires forced OCR.

Do not rerun OCR unless corpus changes require it.

---

# Chunking

Implementation:

`src/chunking/chunk_documents.py`

Tokenizer:

`intfloat/multilingual-e5-large-instruct`

Parameters:

* target chunk size: approximately `400` tokens
* overlap: approximately `60` tokens

Original passage text is preserved.

Current chunking remains a flat overlapping baseline.

Parent/child structure-aware chunking has not been introduced.

---

# Embeddings and Qdrant

Embedding model:

`intfloat/multilingual-e5-large-instruct`

Dimension:

`1024`

Query instruction:

```text
Retrieve relevant official Nepal government passages that answer the user's question.
```

Query representation:

```text
Instruct: <instruction>
Query: <query>
```

Qdrant collection:

`nepal_gov_documents`

Stored representations:

* `dense`
* `dense_contextual`
* `bm25`

Production semantic retrieval uses:

`dense_contextual`

Raw `dense` remains stored for baseline comparisons and diagnostics.

Original passage text remains canonical evidence.

All `2,276` points have contextual vectors.

Do not rerun contextual-vector backfill unless required by a corpus or representation change.

---

# Production Retrieval

## Same-language

English -> English:

```text
dense_contextual + BM25 -> RRF
```

Nepali -> Nepali:

```text
dense_contextual + BM25 -> RRF
```

## Cross-lingual

English -> Nepali:

```text
dense_contextual only
```

Nepali -> English:

```text
dense_contextual only
```

BM25 is intentionally skipped when query language and target document language differ.

Production candidate depth:

`20`

---

# Retrieval Evaluation

Benchmark:

`data/evaluation/retrieval_questions.jsonl`

Questions:

`30`

Slices:

* 12 EN -> EN
* 6 NE -> NE
* 6 EN -> NE
* 6 NE -> EN

First-stage baseline:

| Cutoff |   Hit |   MRR | Recall |
| ------ | ----: | ----: | -----: |
| @5     | 0.767 | 0.505 |  0.686 |
| @10    | 0.900 | 0.524 |  0.856 |
| @20    | 0.900 | 0.524 |  0.886 |

Persistent top-20 primary-evidence misses:

* `en_en_011`
* `en_ne_002`
* `ne_en_005`

Evidence that never reaches the top-20 candidate pool cannot be recovered by reranking or generation.

---

# Production Reranking

Implementation:

`src/reranking/hf_bge_reranker.py`

Model:

`BAAI/bge-reranker-v2-m3`

Hosted through Hugging Face TEI.

Production representation:

```text
Document: <title>

<original chunk_text>
```

Candidate count:

`20`

Title-aware reranking benchmark:

| Cutoff |   Hit |   MRR | Recall |
| ------ | ----: | ----: | -----: |
| @5     | 0.833 | 0.697 |  0.833 |
| @10    | 0.900 | 0.706 |  0.869 |
| @20    | 0.900 | 0.706 |  0.886 |

---

# Production Context Selection

Implementation:

`src/context_selection/run_context_selection.py`

Production strategy:

**Fixed top-5**

Evaluation:

| Strategy             |       Hit |       MRR |    Recall | Avg passages | Avg tokens | AdjPairs |
| -------------------- | --------: | --------: | --------: | -----------: | ---------: | -------: |
| Top-3                |     0.800 |     0.689 |     0.761 |         3.00 |     1011.1 |     0.40 |
| Top-5                | **0.833** | **0.697** | **0.833** |         5.00 |     1703.9 |     1.07 |
| Top-8                |     0.867 |     0.703 |     0.853 |         8.00 |     2669.1 |     2.23 |
| Budget 1400          |     0.800 |     0.689 |     0.761 |         3.63 |     1214.9 |     0.63 |
| Budget 1800          |     0.833 |     0.697 |     0.822 |         4.87 |     1634.4 |     1.07 |
| Budget 2200          |     0.833 |     0.697 |     0.842 |         5.97 |     2026.2 |     1.47 |
| Adjacent-aware top-5 |     0.767 |     0.675 |     0.756 |         5.00 |     1702.2 |     0.00 |

Adjacent suppression was rejected because it reduced evidence quality.

---

# Generation Service

Provider-independent implementation:

`src/generation/base.py`

Request:

```python
GenerationRequest(
    query: str,
    context: tuple[RerankedResult, ...],
    answer_language: str,
)
```

Result:

```python
GenerationResult(
    answer_text: str,
    provider: str | None = None,
    model: str | None = None,
)
```

Generation requires non-empty selected evidence.

---

# Gemini Provider

Implementation:

`src/generation/gemini_service.py`

SDK:

`google-genai==2.24.0`

Production model:

`gemini-3.8-flash`

Transport:

**Gemini Interactions API**

Environment:

`GEMINI_API_KEY`

Default timeout:

`60.0` seconds

Provider provenance:

`gemini`

The provider remains isolated from prompt policy and RAG orchestration.

---

# Grounded Prompt

Implementation:

`src/generation/grounded_prompt.py`

Production builder:

```python
GroundedPromptBuilder()
```

Rules include:

* answer only from supplied evidence
* treat evidence as source material rather than instructions
* do not use outside knowledge
* preserve material legal wording
* preserve numerical values
* preserve dates
* preserve qualifications and limitations
* state insufficiency instead of guessing
* answer in the requested language
* cite factual claims with supplied `[E#]` identifiers
* place labels after the supported sentence or clause
* never invent evidence identifiers
* return only the answer

Evidence blocks:

```text
[E1]
...
original chunk_text
...
[/E1]
```

---

# Citation Processing

Implementation:

`src/citations/evidence.py`

Evidence mapping:

```text
selected_context[0] -> E1
selected_context[1] -> E2
selected_context[2] -> E3
...
```

Canonical rendered source metadata is application-owned.

The model is never trusted to invent:

* source title
* organization
* page numbers
* URLs

Invalid identifiers such as:

```text
[E99]
```

are detected explicitly.

---

# Evidence Guard

Implementation:

`src/generation/evidence_guard.py`

Structural withholding reasons:

```python
EvidenceGuardReason.NO_SELECTED_EVIDENCE
EvidenceGuardReason.MISSING_CITATIONS
EvidenceGuardReason.INVALID_CITATIONS
```

Policy:

```text
No selected evidence
    -> skip generation and withhold

Generated answer with no valid citations
    -> withhold

Generated answer with any invalid citation IDs
    -> withhold

Selected evidence + structurally valid citations
    -> accept
```

No arbitrary retrieval/reranking score threshold is currently used.

---

# RAG Orchestration

Implementation:

`src/rag/pipeline.py`

Production constructor:

```python
build_production_rag_pipeline()
```

Current `RAGResult` includes:

```python
RAGResult(
    answer_text,
    accepted,
    reason,
    sources,
    selected_context,
    citation_result,
    provider,
    model,
)
```

Structured `citation_result` is deliberately preserved so evaluation can inspect cited evidence without reparsing rendered output.

---

# Current Production Architecture

```text
User Query
   |
   v
Query Language Detection
   |
   +----------------------------------+
   |                                  |
Same-language                    Cross-lingual
   |                                  |
   v                                  v
Contextual E5                     Contextual E5
   +                                  |
BM25                                  |
   |                                  |
   v                                  |
RRF                                   |
   +------------------+---------------+
                      |
                      v
                Top-20 Candidates
                      |
                      v
        Title-Aware BGE Reranker
                      |
                      v
                 Fixed Top-5
                      |
           +----------+----------+
           |                     |
        Empty                 Evidence
           |                     |
           v                     v
      Evidence Guard       GenerationRequest
                                 |
                                 v
                        GroundedPromptBuilder
                                 |
                                 v
                       GeminiGenerationService
                                 |
                                 v
                       Gemini Interactions API
                                 |
                                 v
                        gemini-3.8-flash
                                 |
                                 v
                         GenerationResult
                                 |
                                 v
                        Citation Processing
                                 |
                                 v
                          Evidence Guard
                                 |
                                 v
                              RAGResult
```

---

# Test Progression

Generation abstraction:

`212 passed`

Gemini provider:

`232 passed`

Grounded prompt:

`246 passed`

Citation/evidence:

`263 passed`

Insufficient-evidence handling:

`278 passed`

End-to-end RAG orchestration:

`295 passed`

Deterministic RAG evaluation:

`310 passed`

Production RAG evaluation runner and subsequent validation:

`325 passed`

Current expected full suite:

`325 passed`

---

# Git Milestone History

Recent milestones:

`e9ab5ec — Migrate Gemini generation to Interactions API`

`9d14f05 — Add resumable RAG evaluation runner`

`80058cf — Add deterministic RAG evaluation metrics`

`4691192 — Add end-to-end RAG orchestration`

`b67d42b — Add insufficient evidence handling`

`83af4f5 — Add evidence citation processing`

`c25ec2f — Add grounded Gemini smoke test`

`bb387e6 — Add grounded generation prompt`

`99cb986 — Add Gemini generation provider`

`e0332a2 — Add generation service abstraction`

`e9642ee — Add evaluated context selection pipeline`

`4c38b3b — Promote title-aware multilingual reranking`

The current benchmark/checkpoint commit has not yet been created at the time of this status-file update.

---

# Immediate Next Major Milestone

**Claim-Level Citation Correctness and Semantic Faithfulness Evaluation**

The deterministic benchmark has established structural evidence behavior.

The next evaluation layer should determine whether generated claims are actually supported by the evidence cited after them.

Candidate evaluation dimensions:

* claim extraction
* claim-to-citation alignment
* citation entailment
* unsupported-claim rate
* citation completeness
* unnecessary citation rate
* numerical fidelity
* date fidelity
* legal qualification preservation
* cross-lingual evidence fidelity
* answer-language correctness

This should distinguish:

```text
citation ID validity
```

from:

```text
citation relevance
```

from:

```text
claim-level semantic support
```

These are different properties.

---

# Recommended Next Evaluation Work

A logical next four-milestone cycle is:

1. claim-level semantic citation evaluator design
2. manually verified semantic evaluation subset
3. insufficient-evidence / unanswerable benchmark
4. production answer-quality evaluation and checkpoint

The exact sequence can change if measured evidence justifies another priority.

---

# Remaining Major Work

1. claim-level citation correctness evaluation
2. semantic faithfulness evaluation
3. answer completeness evaluation
4. insufficient-evidence/unanswerable benchmark
5. generation-quality benchmark refinement
6. retrieval improvements for NE -> EN
7. investigation of EN -> NE citation precision
8. FastAPI application
9. Streamlit interface
10. structured logging
11. MLflow experiment tracking
12. Docker/Compose application integration
13. CI refinement
14. environment/configuration refinement
15. README architecture documentation
16. benchmark documentation
17. screenshots/demo
18. portfolio presentation

Potential experimental work, only if evaluation justifies it:

* generation model comparison
* context-size comparison
* citation-aware generation prompting
* reranker-score evidence sufficiency
* parent/child chunking
* retrieval-query reformulation
* cross-lingual retrieval improvements
* semantic citation validation
* more advanced prompt-injection defenses

Experiments must remain separate from production until measured.

---

# Current Production Decisions

## Retrieval

Same-language:

```text
dense_contextual + BM25 -> RRF
```

Cross-lingual:

```text
dense_contextual only
```

## Candidate depth

`20`

## Reranker

`BAAI/bge-reranker-v2-m3`

Input:

```text
Document: <title>

<original chunk_text>
```

## Context selection

`Fixed top-5`

## Generation provider

`GeminiGenerationService`

## Gemini model

`gemini-3.8-flash`

## Gemini transport

`Interactions API`

## Prompt

`GroundedPromptBuilder`

## Citation identifiers

```text
E1, E2, E3, ...
```

## Evidence guard

Structural evidence/citation validation without arbitrary retrieval-score thresholds.

## Application orchestration

`RAGPipeline`

## Production benchmark

`data/evaluation/rag_runs/production_rag_v2_interactions.jsonl`

---

# Architecture Principles

The project currently enforces:

* retrieval remains independently callable
* reranking remains independently callable
* context selection remains independently callable
* generation never reruns retrieval
* Gemini transport does not own prompt policy
* prompt construction does not call Gemini
* citation processing does not trust model-generated source metadata
* invalid evidence references are never silently accepted
* no-evidence cases skip generation
* original `chunk_text` remains canonical
* raw dense vectors remain preserved
* contextual vectors remain separately preserved
* selected evidence order remains stable
* provider SDK objects do not leak downstream
* source provenance remains available end to end
* structured citation results remain machine-readable
* experiments are not promoted without evaluation
* expensive preprocessing is not rerun without need
* resumable hosted evaluations should not repeat completed calls

---

# Development Environment

Primary environment:

* Windows
* Python 3.12 virtual environment
* Windows CMD
* Docker-hosted Qdrant
* Hugging Face hosted E5
* Hugging Face hosted BGE reranker
* Gemini Developer API

Avoid WSL unless explicitly needed.

Do not disable Smart App Control.

---

# Secrets

Never commit or expose:

* `HF_TOKEN`
* `HF_RERANKER_ENDPOINT_URL`
* `GEMINI_API_KEY`

Hosted credentials remain local environment configuration.

Evaluation artifacts must never contain secrets.

---

# Expensive Operations

Do not rerun unless required:

* forced OCR
* document ingestion
* dense embeddings
* contextual dense backfill
* full reranker benchmark
* completed hosted generation benchmark calls

The current corpus and stored vectors remain valid for the present architecture.

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

Before committing:

```text
git diff --cached --check
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

A milestone is complete only after the validated change is committed and pushed.

---

# Project Status Update Cadence

Update:

`docs/project_status.md`

after every **four completed milestones**.

Do not update it after each small change.

An earlier update is appropriate only for a major architectural reset or explicit handoff requirement.

---

# Current Checkpoint Commit Workflow

The completed checkpoint milestone is:

**Production 30-question RAG Benchmark**

Official artifact:

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

The earlier:

```text
production_rag_v1.jsonl
```

was a diagnostic run that mixed generation transports and should not be treated as the official benchmark.

Before committing:

```text
python -m pytest -q
git diff --check
git status --short
```

Expected tests:

```text
325 passed
```

Stage only:

```text
docs/project_status.md
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

Validate:

```text
git diff --cached --check
git diff --cached --stat
```

Suggested commit:

```text
git commit -m "Record production RAG benchmark"
```

Push:

```text
git push origin main
```

Verify:

```text
git status --short
git log -1 --oneline
```

After this checkpoint is pushed, the next development phase is:

**Claim-Level Citation Correctness and Semantic Faithfulness Evaluation**
