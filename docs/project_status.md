# NepalGov AI — Current Project Status

Last updated: 2026-09-20

## Purpose of This Document

This file is the project handoff and checkpoint document for NepalGov AI.

It exists so development can continue accurately across ChatGPT conversations without reconstructing architectural decisions, benchmark results, implemented milestones, and remaining work from scratch.

Technical source of truth remains:

* current repository code
* automated tests
* evaluation outputs
* Git history

Project-status updates are now performed after every **four completed milestones**, rather than after every individual milestone.

Between status checkpoints, Git history, tests, and repository code remain the live source of truth.

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

The core RAG architecture is now implemented through application-level orchestration.

Completed major layers:

1. multilingual retrieval
2. multilingual reranking
3. context selection
4. provider-independent generation contract
5. Gemini generation provider
6. grounded prompt construction
7. live Gemini generation validation
8. citation/evidence processing
9. insufficient-evidence handling
10. end-to-end RAG orchestration

Current locally validated test suite:

`295 passed`

Latest validation:

`git diff --check` — clean

Current local uncommitted milestone:

**End-to-End RAG Orchestration**

Files:

* `src/rag/pipeline.py`
* `tests/test_rag_pipeline.py`
* `docs/project_status.md`

Next major milestone after this checkpoint:

**End-to-End RAG Evaluation**

---

# Git Milestone History

Latest committed milestone on `main`:

`b67d42b — Add insufficient evidence handling`

Recent milestones:

`83af4f5 — Add evidence citation processing`

`c25ec2f — Add grounded Gemini smoke test`

`bb387e6 — Add grounded generation prompt`

`99cb986 — Add Gemini generation provider`

`e0332a2 — Add generation service abstraction`

`e9642ee — Add evaluated context selection pipeline`

`4c38b3b — Promote title-aware multilingual reranking`

The current End-to-End RAG Orchestration milestone is locally validated but not yet committed at this checkpoint.

---

# Four-Milestone Checkpoint Summary

This status update captures the following four completed development milestones.

## Milestone 1 — Live Grounded Gemini Smoke Test

Committed as:

`c25ec2f — Add grounded Gemini smoke test`

Implementation:

`src/generation/run_gemini_smoke_test.py`

The smoke test exercised the real production generation path:

```text
GenerationRequest
        |
        v
GroundedPromptBuilder
        |
        v
GeminiGenerationService
        |
        v
Gemini API
        |
        v
GenerationResult
```

The test used deterministic synthetic evidence rather than live retrieval so generation behavior could be isolated.

Synthetic evidence stated that a public service desk:

* operates Monday through Thursday
* operates from 09:00 to 15:00
* is closed Friday

The real Gemini response correctly preserved all three facts and cited:

`[E1]`

Observed successful output included:

* provider: `gemini`
* model: `gemini-3.8-flash`
* grounded answer
* valid evidence identifier `[E1]`

The first attempts returned:

`503 UNAVAILABLE`

because the model endpoint was temporarily under high demand.

A later request succeeded without changing the production model.

Conclusion:

* Gemini credentials work
* SDK connectivity works
* model access works
* grounded prompt works against the real model
* response parsing works
* provider/model provenance works
* generated evidence labels work in practice

An SDK AFC warning was also observed, but it did not prevent successful text generation.

---

## Milestone 2 — Citation / Evidence Processing

Committed as:

`83af4f5 — Add evidence citation processing`

Implementation:

`src/citations/evidence.py`

Tests:

`tests/test_evidence_citations.py`

This layer formalizes the evidence identifier contract introduced by:

`GroundedPromptBuilder`

Evidence identifiers are deterministic:

```text
E1
E2
E3
...
```

They correspond directly to selected-context order.

Example:

```text
selected_context[0] -> E1
selected_context[1] -> E2
selected_context[2] -> E3
```

Citation processing now supports:

* building evidence-ID maps
* extracting model-generated `[E#]` references
* preserving first-appearance order
* deduplicating repeated references
* detecting invalid evidence identifiers
* mapping valid IDs back to exact `RerankedResult` objects
* preserving generated answer text exactly
* rendering canonical source metadata

Core types:

```python
EvidenceCitation
CitationProcessingResult
```

Key functions:

```python
build_evidence_map()
extract_evidence_ids()
process_answer_citations()
format_evidence_citation()
render_cited_sources()
```

The model is never trusted to supply canonical source metadata.

Instead:

```text
model output:
[E1]
   |
   v
validated against selected context
   |
   v
canonical title / organization / pages / URL
```

Invalid model references such as:

```text
[E99]
```

are detected explicitly.

They do not silently resolve to a source.

---

## Milestone 3 — Insufficient-Evidence Handling

Committed as:

`b67d42b — Add insufficient evidence handling`

Implementation:

`src/generation/evidence_guard.py`

Tests:

`tests/test_evidence_guard.py`

The evidence guard introduces deterministic application-level withholding behavior.

No retrieval-score or reranker-score threshold has been introduced.

Current structural withholding reasons:

```python
EvidenceGuardReason.NO_SELECTED_EVIDENCE
EvidenceGuardReason.MISSING_CITATIONS
EvidenceGuardReason.INVALID_CITATIONS
```

### No selected evidence

If upstream context selection returns no passages:

```text
retrieval/context selection
        |
        v
no context
        |
        v
do not call LLM
        |
        v
deterministic insufficient-evidence response
```

### Missing citations

If generation occurs but the answer contains no valid evidence identifier:

```text
generated answer
        |
        v
no [E#]
        |
        v
withhold model answer
```

### Invalid citations

If the model invents a reference such as:

```text
[E99]
```

the answer is withheld even if another valid citation also appears.

Invalid citation handling therefore takes precedence.

### Deterministic messages

English:

```text
The supplied government evidence is insufficient to provide a supported answer.
```

Nepali:

```text
उपलब्ध गराइएको सरकारी प्रमाणका आधारमा पर्याप्त रूपमा समर्थित उत्तर दिन सकिएन।
```

Unknown language identifiers currently fall back to the English application-owned message.

### Important limitation

The current evidence guard treats valid citation structure as a necessary condition for presenting generated output.

It does **not** claim that citation presence alone proves semantic faithfulness.

Claim-level factual support remains an evaluation problem.

---

## Milestone 4 — End-to-End RAG Orchestration

Current locally completed milestone.

Implementation:

`src/rag/pipeline.py`

Tests:

`tests/test_rag_pipeline.py`

Current full test result:

`295 passed`

The new application-level orchestration composes existing independently tested stages rather than reimplementing their behavior.

Pipeline:

```text
query
  |
  v
context provider
  |
  v
retrieval + reranking + context selection
  |
  +---------------- no evidence ----------------+
  |                                             |
  |                                             v
  |                               deterministic withholding
  |
  v
GenerationRequest
  |
  v
GroundedPromptBuilder
  |
  v
GenerationService
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

Core orchestration types:

```python
ContextProvider
RAGResult
RAGPipeline
```

Production constructor:

```python
build_production_rag_pipeline()
```

Production generation wiring:

```python
GeminiGenerationService(
    prompt_builder=GroundedPromptBuilder(),
)
```

Production context provider:

```python
run_context_selection
```

---

# RAGResult Contract

The final application-level result contains:

```python
RAGResult(
    answer_text: str,
    accepted: bool,
    reason: EvidenceGuardReason | None,
    sources: tuple[str, ...],
    selected_context: tuple[RerankedResult, ...],
    provider: str | None,
    model: str | None,
)
```

This creates one clean boundary for future:

* FastAPI
* Streamlit
* evaluation
* structured logging
* monitoring
* MLflow

---

# End-to-End Orchestration Behavior

## Successful answer

When evidence exists and generated citations are valid:

```text
selected evidence
      |
      v
generation
      |
      v
valid citations
      |
      v
accepted answer
      |
      v
canonical rendered sources
```

The original generated answer is preserved.

Provider provenance is retained.

Selected context is retained.

---

## No selected evidence

If context selection returns:

```python
[]
```

generation is skipped completely.

The result contains:

```text
accepted = False
reason = NO_SELECTED_EVIDENCE
provider = None
model = None
sources = ()
selected_context = ()
```

This prevents evidence-free LLM generation.

---

## Missing citations

If generated output does not contain a valid evidence label:

```text
accepted = False
reason = MISSING_CITATIONS
sources = ()
```

The generated answer is replaced with the deterministic insufficient-evidence message.

Provider/model provenance remains available because generation actually occurred.

---

## Invalid citations

If generated output contains an invented evidence identifier:

```text
[E99]
```

the result becomes:

```text
accepted = False
reason = INVALID_CITATIONS
sources = ()
```

Invalid sources are never rendered to the user.

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

Preserved metadata includes:

* chunk ID
* document ID
* title
* organization
* category
* document type
* language
* publication date
* source URL
* page range
* section
* subsection
* article number
* article title
* chunk index
* original passage text
* tokenizer-derived token count
* extraction method

Current chunking remains a flat overlapping baseline.

Parent/child structure-aware chunking has not been introduced.

---

# Embeddings

Primary model:

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

Stored Qdrant representations:

* `dense`

  * original raw passage embedding
* `dense_contextual`

  * metadata-enriched passage embedding
* `bm25`

  * sparse lexical representation

Production semantic retrieval uses:

`dense_contextual`

Raw dense remains stored for controlled comparisons and diagnostics.

Original `chunk_text` remains canonical evidence.

---

# Production Retrieval

## Same-language retrieval

English -> English:

```text
dense_contextual + BM25 -> RRF
```

Nepali -> Nepali:

```text
dense_contextual + BM25 -> RRF
```

## Cross-lingual retrieval

English -> Nepali:

```text
dense_contextual only
```

Nepali -> English:

```text
dense_contextual only
```

BM25 is intentionally skipped when query and target document languages differ.

Query language detection is implemented in:

`src/retrieval/run_hybrid_retrieval.py`

Current V1 routing identifies Devanagari queries as Nepali and otherwise routes them as English.

---

# Retrieval Evaluation Dataset

Manually verified benchmark:

`30 questions`

Slices:

* 12 English -> English
* 6 Nepali -> Nepali
* 6 English -> Nepali
* 6 Nepali -> English

Metrics:

* Hit Rate
* MRR
* Recall

Persistent first-stage primary-evidence misses at @20:

* `en_en_011`
* `en_ne_002`
* `ne_en_005`

These cannot be recovered downstream because required evidence never enters the top-20 candidate pool.

---

# First-Stage Retrieval Baseline

| Cutoff |   Hit |   MRR | Recall |
| ------ | ----: | ----: | -----: |
| @5     | 0.767 | 0.505 |  0.686 |
| @10    | 0.900 | 0.524 |  0.856 |
| @20    | 0.900 | 0.524 |  0.886 |

Validated production candidate depth:

`20`

---

# Production Reranking

Provider abstraction:

`src/reranking/base.py`

Hosted implementation:

`src/reranking/hf_bge_reranker.py`

Production integration:

`src/retrieval/run_reranked_retrieval.py`

Model:

`BAAI/bge-reranker-v2-m3`

Production representation:

```text
Document: <title>

<original chunk_text>
```

The richer contextual dense representation remains retrieval-only.

Original passage text remains unchanged.

---

# Reranking Evaluation

Title-aware BGE:

| Cutoff |   Hit |   MRR | Recall |
| ------ | ----: | ----: | -----: |
| @5     | 0.833 | 0.697 |  0.833 |
| @10    | 0.900 | 0.706 |  0.869 |
| @20    | 0.900 | 0.706 |  0.886 |

Important source-sensitive case:

`en_en_007`

```text
First-stage rank: 7
Plain BGE rank: 12
Title-aware BGE rank: 2
```

Selected production reranker representation remains title-aware.

---

# Production Context Selection

Abstraction:

`src/context_selection/base.py`

Production selector:

`src/context_selection/fixed_top_k.py`

Production integration:

`src/context_selection/run_context_selection.py`

Selected strategy:

**Fixed top-5**

Experimental selectors retained:

* token-budget selector
* adjacent-chunk selector

---

# Context Selection Evaluation

| Strategy             |       Hit |       MRR |    Recall | Avg passages | Avg tokens | AdjPairs |
| -------------------- | --------: | --------: | --------: | -----------: | ---------: | -------: |
| Top-3                |     0.800 |     0.689 |     0.761 |         3.00 |     1011.1 |     0.40 |
| Top-5                | **0.833** | **0.697** | **0.833** |         5.00 |     1703.9 |     1.07 |
| Top-8                |     0.867 |     0.703 |     0.853 |         8.00 |     2669.1 |     2.23 |
| Budget 1400          |     0.800 |     0.689 |     0.761 |         3.63 |     1214.9 |     0.63 |
| Budget 1800          |     0.833 |     0.697 |     0.822 |         4.87 |     1634.4 |     1.07 |
| Budget 2200          |     0.833 |     0.697 |     0.842 |         5.97 |     2026.2 |     1.47 |
| Adjacent-aware top-5 |     0.767 |     0.675 |     0.756 |         5.00 |     1702.2 |     0.00 |

Production decision:

**Fixed top-5**

Adjacent evidence is not automatically treated as redundant because suppression reduced evidence quality.

---

# Generation Service

Provider-independent implementation:

`src/generation/base.py`

Core request:

```python
GenerationRequest(
    query: str,
    context: tuple[RerankedResult, ...],
    answer_language: str,
)
```

Core result:

```python
GenerationResult(
    answer_text: str,
    provider: str | None = None,
    model: str | None = None,
)
```

Interface:

```python
GenerationService.generate(
    request: GenerationRequest,
) -> GenerationResult
```

Generation requires selected evidence.

An empty-context generation request is rejected.

---

# Gemini Provider

Implementation:

`src/generation/gemini_service.py`

SDK dependency:

`google-genai==2.24.0`

Production model:

`gemini-3.8-flash`

Environment variable:

`GEMINI_API_KEY`

Default timeout:

`60.0` seconds

Provider provenance:

`gemini`

The provider supports:

* environment credential resolution
* explicit API-key injection
* explicit model configuration
* timeout configuration
* lazy SDK construction
* fake client injection
* response parsing
* provider-error isolation
* provider/model provenance
* SDK client cleanup

Gemini-specific SDK objects do not propagate into application code.

---

# Grounded Prompt

Implementation:

`src/generation/grounded_prompt.py`

Production callable:

```python
GroundedPromptBuilder()
```

Prompt responsibilities:

* question placement
* answer-language instruction
* evidence boundaries
* deterministic evidence IDs
* canonical evidence metadata
* evidence-only answering instructions
* prompt-injection boundary instructions
* citation requirements
* instruction to abstain instead of guessing

The prompt requires factual claims to cite supplied evidence IDs.

Example:

```text
The Constitution guarantees ... [E1]
```

Evidence IDs must be placed after the sentence or clause they support.

The prompt explicitly prohibits invented or modified evidence labels.

---

# Evidence Representation

Evidence blocks use:

```text
[E1]
Document: ...
Organization: ...
Document ID: ...
Language: ...
Pages: ...
Section: ...
Article: ...
Passage:
<original chunk_text>
[/E1]
```

Optional metadata is included only when available.

Original `chunk_text` is inserted verbatim.

Prompt construction does not:

* summarize evidence
* paraphrase evidence
* translate evidence
* alter legal wording
* alter dates
* alter numerical values
* reorder selected evidence

---

# Citation Processing

Implementation:

`src/citations/evidence.py`

Deterministic mapping:

```text
selected_context[0] -> E1
selected_context[1] -> E2
...
```

Citation parsing preserves:

* first-appearance order
* unique evidence IDs
* exact underlying `RerankedResult` identity

Rendered source metadata uses application-owned provenance:

* evidence ID
* document title
* organization
* page/page range
* source URL

The language model is not trusted to supply canonical citation metadata.

---

# Evidence Guard

Implementation:

`src/generation/evidence_guard.py`

Current structural policy:

```text
No selected evidence
    -> withhold

No valid citations
    -> withhold

Invalid evidence IDs
    -> withhold

Valid selected evidence + valid citations
    -> allow
```

Current policy deliberately does not use:

* dense similarity thresholds
* BM25 thresholds
* RRF thresholds
* reranker-score cutoffs
* model confidence estimates

Those must be evaluated before becoming production rules.

---

# Current End-to-End Production Architecture

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
RRF                                  |
   +------------------+---------------+
                      |
                      v
                Top-20 Candidates
                      |
                      v
        Title-Aware BGE Reranker
                      |
                      v
          Reranked Candidate Pool
                      |
                      v
             Fixed Top-5
           Context Selection
                      |
           +----------+----------+
           |                     |
        Empty                 Evidence
           |                     |
           v                     v
      Evidence Guard       GenerationRequest
           |                     |
           |                     v
           |            GroundedPromptBuilder
           |                     |
           |                     v
           |          GeminiGenerationService
           |                     |
           |                     v
           |                 Gemini API
           |                     |
           |                     v
           |              GenerationResult
           |                     |
           |                     v
           |            Citation Processing
           |                     |
           |                     v
           +------------ Evidence Guard
                                 |
                                 v
                              RAGResult
```

---

# Current Application API Boundary

The new RAG orchestration gives future interfaces one primary application object:

```python
RAGPipeline
```

Typical production construction:

```python
pipeline = build_production_rag_pipeline()
```

Typical answer call:

```python
result = pipeline.answer(
    query,
    answer_language="en",
)
```

The same pipeline can later be reused by:

* FastAPI
* Streamlit
* batch evaluation
* CLI workflows
* experiment runners

without duplicating retrieval/generation logic.

---

# Current Test State

Latest full local run:

```text
295 passed in 2.93s
```

Validation:

```text
git diff --check
```

Result:

clean

Current local Git state before this checkpoint update:

```text
?? src/rag/
?? tests/test_rag_pipeline.py
```

After replacing this file, expected status becomes approximately:

```text
M docs/project_status.md
?? src/rag/
?? tests/test_rag_pipeline.py
```

---

# Checkpoint Test Progression

Generation Service baseline:

`212 passed`

Gemini provider:

`232 passed`

Grounded prompt:

`246 passed`

Citation processing:

`263 passed`

Insufficient-evidence handling:

`278 passed`

End-to-end RAG orchestration:

`295 passed`

The test count progression reflects added coverage rather than removal of existing tests.

---

# Immediate Next Major Milestone

**End-to-End RAG Evaluation**

The core execution pipeline now exists.

The next step should measure whether final generated answers are actually high quality across the multilingual evaluation set.

Evaluation should go beyond retrieval metrics.

Candidate generation metrics include:

* answer relevance
* factual consistency
* faithfulness to evidence
* evidence utilization
* citation correctness
* citation validity
* citation completeness
* unsupported-claim rate
* refusal/withholding correctness
* unnecessary-refusal rate
* answer-language correctness
* English -> English quality
* Nepali -> Nepali quality
* English -> Nepali quality
* Nepali -> English quality

The evaluation should also inspect documented failure cases.

---

# Important Evaluation Limitation

Current evidence guarding verifies structural citation behavior.

Example:

```text
claim [E1]
```

can be validated as referencing a real selected passage.

However, this does not prove that:

```text
claim
```

is semantically supported by:

```text
E1
```

Therefore future evaluation must distinguish:

```text
citation validity
```

from:

```text
citation correctness / claim support
```

This distinction is essential before the system can claim strong groundedness.

---

# Remaining Major Work

1. End-to-end RAG evaluation
2. generation-quality benchmark refinement
3. citation correctness/completeness evaluation
4. insufficient-evidence evaluation
5. FastAPI application
6. Streamlit interface
7. structured logging
8. MLflow experiment tracking
9. Docker/Compose application integration
10. CI refinement
11. environment/configuration refinement
12. README architecture documentation
13. benchmark documentation
14. screenshots/demo
15. portfolio presentation

Potential future experimental work, only if evaluation justifies it:

* generation model comparison
* context-size comparison for final answer quality
* reranker-score evidence sufficiency
* claim-level grounding validation
* parent/child chunking
* retrieval-query reformulation
* more advanced prompt-injection defenses

These should not be promoted without measured evidence.

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

## First-stage depth

`20`

## Reranker

`BAAI/bge-reranker-v2-m3`

Representation:

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

## Gemini SDK

`google-genai==2.24.0`

## Prompt

`GroundedPromptBuilder`

## Citation IDs

```text
E1, E2, E3, ...
```

## Evidence guard

Structural citation/evidence validation without score thresholds.

## Application orchestration

`RAGPipeline`

---

# Architecture Principles

The project currently enforces these separation-of-concern boundaries:

```text
Retrieval
   |
   v
Reranking
   |
   v
Context Selection
   |
   v
Generation Request
   |
   v
Grounded Prompt
   |
   v
Generation Provider
   |
   v
Citation Processing
   |
   v
Evidence Guard
   |
   v
Application RAG Result
```

Rules:

* retrieval remains independently callable
* reranking remains independently callable
* context selection remains independently callable
* generation never reruns retrieval
* Gemini transport does not own prompt policy
* prompt construction does not call Gemini
* citation processing does not trust model-generated source metadata
* invalid evidence references are never silently accepted
* no-evidence cases skip LLM generation
* original `chunk_text` remains canonical
* selected evidence order remains stable
* provider SDK objects do not leak into downstream layers
* source provenance remains available end to end
* experiments are not promoted without evaluation

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

# Secrets and Environment Rules

Never commit:

* `HF_TOKEN`
* `HF_RERANKER_ENDPOINT_URL`
* `GEMINI_API_KEY`

Never paste secrets into ChatGPT.

Hosted credentials should remain local environment configuration.

---

# Expensive Operations

Do not rerun unless necessary:

* forced OCR
* document ingestion
* dense embeddings
* contextual dense backfill
* full reranker benchmark

Current corpus and vectors are already valid for the present architecture.

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

Verify after push:

```text
git status --short
git log -1 --oneline
```

A milestone is complete only after the validated change is committed and pushed.

---

# Project Status Update Cadence

`docs/project_status.md` should now normally be updated after every **four completed milestones**.

Do not update it after every minor change.

Between checkpoint updates:

* Git history is the milestone record
* tests are the validation record
* code is the architecture source of truth

A status update may still be performed earlier if a major architectural reset or handoff requires it.

---

# Current Checkpoint Commit Workflow

The current locally completed milestone is:

**End-to-End RAG Orchestration**

Before committing, run:

```text
python -m pytest -q
git diff --check
git status --short
```

Expected tests:

`295 passed`

Stage:

```text
git add docs/project_status.md
git add src/rag
git add tests/test_rag_pipeline.py
```

Validate:

```text
git status --short
git diff --cached --check
git diff --cached --stat
```

Commit:

```text
git commit -m "Add end-to-end RAG orchestration"
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

After this commit is pushed, the next development phase is:

**End-to-End RAG Evaluation**
