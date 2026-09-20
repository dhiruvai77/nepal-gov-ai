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

**Gemini generation provider completed locally; grounded prompt construction is the next milestone**

The project now has independently testable boundaries for:

* multilingual first-stage retrieval
* title-aware multilingual reranking
* fixed top-5 context selection
* provider-independent generation
* Gemini generation transport

The Gemini provider is implemented through:

`src/generation/gemini_service.py`

It sits behind the generic:

`GenerationService`

interface.

The implementation includes:

* Gemini API credential resolution
* explicit model configuration
* explicit timeout configuration
* lazy SDK client construction
* injected fake clients for unit testing
* injected prompt-builder dependency
* request execution
* provider error isolation
* response parsing
* conversion to `GenerationResult`
* provider/model provenance
* client cleanup

The provider uses:

`google-genai==2.24.0`

Selected initial model:

`gemini-3.8-flash`

Credential environment variable:

`GEMINI_API_KEY`

The provider intentionally does **not** yet define the production grounded prompt.

The next milestone is:

**Grounded Prompt Construction**

Latest full local test suite:

`232 passed`

Latest whitespace/error validation:

`git diff --check` — clean

---

## Last Validated Repository State

Latest committed milestone on `main`:

`e0332a2 — Add generation service abstraction`

Previous milestones:

`e9642ee — Add evaluated context selection pipeline`

`4c38b3b — Promote title-aware multilingual reranking`

Current Gemini provider milestone is fully implemented and locally validated but has not yet been committed at this checkpoint.

Current expected local milestone changes:

* `requirements.txt`
* `src/generation/gemini_service.py`
* `tests/test_gemini_generation_service.py`
* `docs/project_status.md`

Latest full local test suite:

`232 passed`

Previous committed Generation Service abstraction baseline:

`212 passed`

Latest whitespace/error validation:

`git diff --check` — clean

Development environment:

* Windows
* Python 3.12 virtual environment
* Windows CMD
* Qdrant via Docker
* Hosted Hugging Face inference
* Gemini Developer API
* Smart App Control remains enabled

Important:

* Do not disable Smart App Control.
* Avoid WSL unless explicitly requested.
* Never expose API keys in chat.
* Never commit API keys.
* Do not commit hosted endpoint URLs or secrets.
* Qdrant development storage is persisted through Docker Compose.
* Do not rerun expensive OCR or vector backfills unless required.

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

Current implementation:

`src/chunking/chunk_documents.py`

Tokenizer:

`intfloat/multilingual-e5-large-instruct`

Baseline chunk parameters:

* target chunk size: approximately `400` tokens
* overlap: approximately `60` tokens

Chunks preserve provenance including:

* chunk ID
* document ID
* title
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
* chunk index
* original chunk text
* tokenizer-derived token count
* extraction method

The current baseline is not yet a structure-aware parent/child chunking system.

Adjacent chunks may intentionally share content because of the configured overlap.

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

Stored Qdrant representations:

* `dense`

  * original raw passage embedding
* `dense_contextual`

  * metadata-enriched passage embedding
* `bm25`

  * sparse lexical representation

Production semantic retrieval uses:

`dense_contextual`

Raw dense remains stored for controlled diagnostics.

Original passage text remains unchanged.

---

## Qdrant Payload Metadata

Qdrant stores the complete chunk payload.

Important downstream fields include:

* `chunk_index`
* `token_count`

These were already stored in Qdrant before Context Selection but were not originally exposed through `RetrievalResult`.

`RetrievalResult` now includes optional:

```python
chunk_index: int | None = None
token_count: int | None = None
```

No reingestion was required.

No OCR rerun was required.

No vector backfill was required.

These fields now support:

* context selection
* token-cost analysis
* adjacency diagnostics
* generation diagnostics
* future context-management strategies

---

## Production First-Stage Retrieval Architecture

### Same-language retrieval

English -> English:

```text
dense_contextual + BM25 -> Reciprocal Rank Fusion
```

Nepali -> Nepali:

```text
dense_contextual + BM25 -> Reciprocal Rank Fusion
```

### Cross-lingual retrieval

English -> Nepali:

```text
dense_contextual only
```

Nepali -> English:

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

Persistent first-stage primary-evidence misses at top-20:

* `en_en_011`
* `en_ne_002`
* `ne_en_005`

These cannot be recovered downstream because the required evidence does not enter the reranker candidate pool.

---

## First-Stage Retrieval Baseline

### @5

* Hit: `0.767`
* MRR: `0.505`
* Recall: `0.686`

### @10

* Hit: `0.900`
* MRR: `0.524`
* Recall: `0.856`

### @20

* Hit: `0.900`
* MRR: `0.524`
* Recall: `0.886`

These are the validated current baseline values.

---

## Historical Retrieval Metric Correction

An earlier project-status snapshot recorded:

* `0.522` MRR @5
* `0.541` MRR @10
* `0.541` MRR @20

Fresh official evaluation and the independent reranker benchmark both reproduced:

* `0.505` @5
* `0.524` @10
* `0.524` @20

The reproduced values are the validated baseline.

The reason for the earlier small difference was not established and should not be inferred without evidence.

---

## Retrieval Experiments Completed

Completed retrieval/ranking experiments include:

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
* title-aware BGE reranking
* fixed-count context selection
* token-budget context selection
* adjacent-chunk-aware context selection

Important first-stage result:

Equal-weight fusion of raw dense + contextual dense degraded retrieval quality, especially Nepali -> English.

Therefore:

**Do not add raw dense back into production RRF unless new evaluation evidence justifies it.**

---

## Production Reranking

Provider-independent abstraction:

`src/reranking/base.py`

Hosted provider:

`src/reranking/hf_bge_reranker.py`

Production integration:

`src/retrieval/run_reranked_retrieval.py`

Selected model:

`BAAI/bge-reranker-v2-m3`

Selected reranker input:

```text
Document: <title>

<original chunk_text>
```

The original evidence text remains unchanged.

---

## Title-Aware BGE Results

### @5

* Hit: `0.833`
* MRR: `0.697`
* Recall: `0.833`

### @10

* Hit: `0.900`
* MRR: `0.706`
* Recall: `0.869`

### @20

* Hit: `0.900`
* MRR: `0.706`
* Recall: `0.886`

Important source-sensitive case:

`en_en_007`

Primary evidence rank:

```text
First-stage: 7
Plain BGE: 12
Title-aware BGE: 2
```

The title-aware representation corrected documented source confusion.

---

## Production Context Selection

Provider-independent abstraction:

`src/context_selection/base.py`

Production selector:

`src/context_selection/fixed_top_k.py`

Experimental selectors:

* `src/context_selection/token_budget.py`
* `src/context_selection/adjacent_chunk.py`

Production integration:

`src/context_selection/run_context_selection.py`

Selected production strategy:

**Fixed top-5**

---

## Context Selection Results

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

Token budgeting did not improve the measured quality/cost tradeoff.

Aggressive adjacency suppression degraded evidence quality.

---

## Context Selection Interpretation

Top-8 improved:

* Hit by `+0.034`
* MRR by `+0.006`
* Recall by `+0.020`

relative to top-5, but required approximately:

`+965`

average source-passage tokens.

The strict-prefix token budget did not outperform fixed top-5.

Adjacent-aware top-5 reduced average adjacent pairs:

```text
1.07 -> 0.00
```

but degraded:

```text
Hit:
0.833 -> 0.767

MRR:
0.697 -> 0.675

Recall:
0.833 -> 0.756
```

Adjacent chunks therefore cannot currently be treated as useless redundancy.

---

## Generation Service Architecture

Provider-independent implementation:

`src/generation/base.py`

Tests:

`tests/test_generation_base.py`

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

Provider interface:

```python
GenerationService.generate(
    request: GenerationRequest,
) -> GenerationResult
```

Shared helpers:

* `build_generation_request()`
* `build_generation_result()`

Generation consumes selected evidence rather than repeating upstream pipeline stages.

---

## Generation Request Contract

The request contains:

* normalized user query
* selected evidence
* requested answer language

The selected evidence container is:

```python
tuple[RerankedResult, ...]
```

The tuple is immutable.

The underlying `RerankedResult` objects are preserved exactly.

Generation therefore retains:

* original chunk text
* title
* organization
* document ID
* page provenance
* source URL
* language
* category
* document type
* chunk ID
* chunk index
* token count
* retrieval score
* reranker score
* original first-stage rank

---

## Generation Request Validation

Current common validation rules:

### Query

The query is trimmed.

Blank queries raise:

`ValueError`

### Answer language

The language value is:

* trimmed
* normalized to lowercase

Blank language values raise:

`ValueError`

The generic contract does not restrict future providers to only English and Nepali.

### Context

At least one selected evidence passage is required.

Empty context raises:

`ValueError`

This prevents an LLM provider from accidentally generating an ungrounded answer with no selected evidence.

Final user-facing insufficient-evidence behavior remains a later RAG responsibility.

---

## Generation Result Contract

`GenerationResult` contains:

* answer text
* optional provider identity
* optional model identity

This supports future:

* logging
* diagnostics
* model comparison
* evaluation
* MLflow tracking
* operational observability

Provider SDK response objects must not leak downstream.

---

## Gemini Generation Provider

Concrete provider:

`src/generation/gemini_service.py`

Tests:

`tests/test_gemini_generation_service.py`

SDK dependency:

`google-genai==2.24.0`

Selected model:

`gemini-3.8-flash`

Provider name recorded in results:

`gemini`

Credential environment variable:

`GEMINI_API_KEY`

Default timeout:

`60.0` seconds

---

## Gemini Provider Responsibilities

The Gemini provider currently owns:

* resolving `GEMINI_API_KEY`
* accepting an explicitly injected API key
* model configuration
* timeout configuration
* SDK client lifecycle
* lazy client creation
* provider request execution
* provider-error wrapping
* response-text extraction
* conversion to `GenerationResult`
* provider/model provenance
* client cleanup

The provider does **not** own:

* retrieval
* reranking
* context selection
* production grounded prompt policy
* citation rendering
* citation validation
* insufficient-evidence policy

---

## Gemini Prompt Boundary

The provider accepts an injected callable:

```python
PromptBuilder = Callable[
    [GenerationRequest],
    str,
]
```

Flow:

```text
GenerationRequest
      |
      v
PromptBuilder
      |
      v
prompt string
      |
      v
GeminiGenerationService
      |
      v
Gemini API
```

This means Gemini transport is independent from prompt policy.

The production grounded prompt builder has not yet been implemented.

---

## Gemini Provider Configuration

Production credentials are resolved from:

`GEMINI_API_KEY`

Explicit API-key injection is supported for tests and application configuration.

An injected test client does not require credentials.

The provider validates:

* prompt builder is callable
* model name is nonblank
* timeout is positive
* production credentials exist when no client is injected
* generated prompt is a string
* generated prompt is nonblank
* Gemini response contains usable text

---

## Gemini Client Construction

The provider creates the official SDK client lazily.

Conceptually:

```python
genai.Client(
    api_key=<configured key>,
    http_options=<configured options>,
)
```

Lazy construction means:

* importing the provider does not make network requests
* unit tests can inject fake clients
* configuration errors can be tested independently
* unused services do not allocate networking resources

The provider also exposes:

`close()`

to release SDK networking resources when a client exists.

---

## Gemini Retry Strategy

The provider does not add a second manual retry loop around the Google SDK.

This avoids nested retry policies and unexpectedly multiplying external requests.

Provider failures are converted to the generic application boundary:

```text
RuntimeError:
Gemini generation request failed.
```

The original exception remains attached as the cause for diagnostics.

Retry policy can be revisited if production behavior or measured reliability requires it.

---

## Gemini Response Handling

Expected Gemini SDK output is converted to:

```python
GenerationResult(
    answer_text=<generated text>,
    provider="gemini",
    model=<configured model>,
)
```

Response validation rejects:

* missing text
* non-string text
* whitespace-only text

Raw provider objects do not propagate into downstream RAG code.

---

## Gemini Provider Unit Tests

Provider tests currently verify:

* selected default model
* provider name
* default timeout
* missing credential rejection
* environment credential resolution
* explicit credential override
* injected client without credentials
* blank model rejection
* invalid timeout rejection
* noncallable prompt-builder rejection
* prompt-builder invocation
* configured model usage
* custom model provenance
* non-string prompt rejection
* blank prompt rejection
* provider-failure wrapping
* missing response-text rejection
* blank response-text rejection
* injected-client cleanup
* safe cleanup before lazy client creation

The provider tests use injected fake clients.

They do not require a real Gemini API key.

---

## Gemini Live Smoke Test Status

**Not yet performed**

This is intentional.

The provider transport boundary is unit tested, but the production grounded prompt builder does not yet exist.

The first meaningful live Gemini request should use the actual grounded prompt rather than a temporary prompt that will immediately be discarded.

A live smoke test should therefore occur during or immediately after the Grounded Prompt milestone.

---

## Current Production RAG Architecture

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
         GenerationRequest
                   |
                   v
         GenerationService
                   |
                   v
    GeminiGenerationService
                   |
                   v
       GroundedPromptBuilder
              [NEXT]
```

---

## Planned Generation Flow

```text
Selected top-5 evidence
        |
        v
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
        |
        v
Citation / evidence processing
```

---

## Immediate Next Milestone

**Grounded Prompt Construction**

Goal:

Create deterministic provider-independent prompt construction from:

`GenerationRequest`

The prompt builder should be independently testable and should not make network requests.

The grounded prompt should address:

* evidence-only answering
* clear evidence boundaries
* answer-language instruction
* source-document distinctions
* preservation of original evidence
* resistance to unsupported inference
* instruction not to invent policy, law, figures, or source details
* future citation identifiers
* deterministic evidence ordering

The prompt builder should consume the already selected top-5 evidence.

It must not:

* rerun retrieval
* rerun reranking
* change context selection
* call Gemini
* rewrite source metadata
* invent citation information

---

## Grounded Prompt Design Requirements

The upcoming prompt should clearly distinguish:

1. system/task instructions
2. user question
3. requested answer language
4. individual evidence passages
5. evidence metadata
6. output constraints

Evidence passages should receive deterministic identifiers that can later support citation processing.

Possible conceptual structure:

```text
Task instructions

Question:
<query>

Answer language:
<language>

Evidence:

[E1]
Document: ...
Pages: ...
Passage: ...

[E2]
...
```

The exact representation must be unit tested before production use.

---

## Citation and Evidence Generation — Future Milestone

Final citation rendering is not yet implemented.

Available provenance already includes:

* title
* organization
* document ID
* page range
* source URL
* chunk ID
* selected evidence order

The prompt builder may expose deterministic evidence IDs such as:

`E1`, `E2`, etc.

Later citation processing should map those IDs back onto actual selected evidence.

The model should not be trusted to invent source metadata.

---

## Insufficient Evidence — Future Milestone

Insufficient-evidence behavior has not yet been implemented.

It should remain separate from:

* retrieval
* reranking
* context selection
* Gemini transport

Future behavior must determine when to:

* answer normally
* answer cautiously
* report insufficient evidence
* avoid unsupported claims

Potential evidence signals may include:

* reranker scores
* retrieval coverage
* evidence agreement
* grounded-answer support
* citation support

No threshold should be introduced into production without evaluation.

---

## End-to-End Evaluation Still Required

Future generation evaluation should measure:

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

Retrieval metrics alone do not establish answer quality.

---

## Important Current Production Decisions

### First-stage retrieval

```text
same-language:
dense_contextual + BM25 -> RRF

cross-lingual:
dense_contextual only
```

### First-stage reranker pool

`20`

### Reranking

Model:

`BAAI/bge-reranker-v2-m3`

Input:

```text
Document: <title>

<original chunk_text>
```

### Context selection

**Fixed top-5**

### Generation abstraction

```text
GenerationRequest
        ->
GenerationService
        ->
GenerationResult
```

### Generation provider

`GeminiGenerationService`

### Gemini model

`gemini-3.8-flash`

### Gemini Python dependency

`google-genai==2.24.0`

### Grounded prompt

**Not yet implemented**

---

## Architecture Principles Established So Far

The project separates:

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
Generation Contract
   |
   v
Generation Provider
   |
   v
Grounded Prompt
   |
   v
Citation / Evidence
```

Important consequences:

* retrieval remains independently callable
* reranking remains independently callable
* context selection remains independently callable
* generation does not rerun retrieval
* generation does not rerun reranking
* generation does not choose new evidence
* Gemini transport does not define prompt policy
* original source text remains canonical
* provider SDK response objects do not leak downstream
* provenance is preserved end to end
* experiments remain separate from production behavior until evaluated

---

## Current Test State

Latest full local test run:

```text
232 passed in 2.54s
```

Latest validation:

```text
git diff --check
```

Result:

clean

Current expected local Git status:

```text
M docs/project_status.md
M requirements.txt
?? src/generation/gemini_service.py
?? tests/test_gemini_generation_service.py
```

This is the expected Gemini provider milestone state before staging.

---

## Git Workflow for Completing Gemini Provider Milestone

Run final validation after saving this status file:

```text
python -m pytest -q
git diff --check
git status --short
```

Expected test count:

`232 passed`

Stage exactly this milestone:

```text
git add docs/project_status.md
git add requirements.txt
git add src/generation/gemini_service.py
git add tests/test_gemini_generation_service.py
```

Validate staged changes:

```text
git status --short
git diff --cached --check
git diff --cached --stat
```

Commit:

```text
git commit -m "Add Gemini generation provider"
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

The milestone is complete only after:

* tests remain green
* staged diff check is clean
* commit succeeds
* push succeeds
* working tree is clean

---

## Remaining Major Work

1. Grounded prompt construction
2. Gemini live grounded-generation smoke test
3. Citation/evidence generation
4. Insufficient-evidence handling
5. End-to-end RAG orchestration
6. End-to-end RAG evaluation
7. FastAPI application
8. Streamlit interface
9. Structured logging
10. MLflow experiment tracking
11. Docker/Compose production integration
12. CI refinement
13. environment/configuration refinement
14. README and architecture documentation
15. benchmark documentation
16. portfolio screenshots/demo

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
* Preserve `dense_contextual` for production semantic retrieval.
* Preserve original `chunk_text` for generation and citations.
* Do not mutate evidence merely to build provider input.
* Keep first-stage retrieval independently callable.
* Keep reranking independently callable.
* Keep context selection independently callable.
* Generation must consume selected evidence rather than repeat upstream stages.
* Keep Gemini-specific behavior behind `GenerationService`.
* Keep grounded prompt construction outside Gemini transport.
* Do not leak Gemini SDK objects into downstream code.
* Preserve source provenance end to end.
* Never commit `HF_TOKEN`.
* Never commit `HF_RERANKER_ENDPOINT_URL`.
* Never commit `GEMINI_API_KEY`.
* Do not add production thresholds, deduplication rules, refusal policies, or score cutoffs without evaluation evidence.
* Treat repository code, tests, evaluation output, and Git history as technical source of truth.
* Treat this file as a handoff/checkpoint document.
