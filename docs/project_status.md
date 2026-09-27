# NepalGov AI — Current Project Status

Last updated: 2026-09-27

## Purpose of This Document

This file is the project handoff and checkpoint document for NepalGov AI.

It exists so development can continue accurately across ChatGPT conversations without reconstructing architectural decisions, benchmark results, evaluation history, production decisions, and remaining work from scratch.

Technical source of truth remains:

* current repository code
* automated tests
* persisted evaluation outputs
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

The complete V1 RAG execution path is implemented.

The system now has evaluation at multiple independent levels:

1. retrieval and selected-context coverage
2. structural citation correctness
3. human-reviewed claim-level semantic support
4. automated semantic-judge agreement with human labels
5. hard-negative and ambiguity semantic-judge evaluation
6. structural insufficient-evidence handling
7. human response-level answer / abstention / partial-answer behavior
8. targeted cross-lingual retrieval diagnostics
9. response-level answer completeness
10. response-level factual fidelity

The project now has a second deterministic consolidated production-quality checkpoint that preserves the previous checkpoint and adds the latest four-milestone evaluation cycle.

Completed major system layers:

1. document ingestion and chunking
2. multilingual dense embeddings
3. contextual dense embeddings
4. sparse BM25 representations
5. multilingual hybrid retrieval
6. multilingual reranking
7. context selection
8. provider-independent generation abstraction
9. Gemini generation provider
10. grounded evidence prompt
11. citation/evidence processing
12. structural insufficient-evidence handling
13. application-level RAG orchestration
14. deterministic RAG evaluation metrics
15. resumable production RAG benchmark runner
16. Gemini Interactions API transport
17. complete 30-question production RAG benchmark
18. deterministic claim-to-citation alignment
19. semantic citation dataset with exact passages
20. deterministic stratified human semantic-review subset
21. completed human semantic citation evaluation
22. automated semantic judge with human-label validation
23. insufficient-evidence benchmark
24. human response-level abstention evaluation
25. deterministic production-quality checkpoint V1
26. partial/mixed insufficient-evidence benchmark
27. hard semantic-judge challenge set
28. targeted NE -> EN retrieval experiments
29. full 30-question answer-completeness and factual-fidelity review
30. deterministic production-quality checkpoint V2

Current validated test suite:

```text
604 passed in 3.35s
```

Latest committed and pushed milestone on `main`:

```text
1e7619d — Add answer quality evaluation checkpoint
```

Current repository head before this status-document checkpoint update:

```text
1e7619d — Add answer quality evaluation checkpoint
```

Current development phase:

**Measured retrieval and answer-quality improvement without speculative production changes.**

---

# Corpus

The indexed corpus currently contains six Government of Nepal documents and 2,276 chunks.

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

# Embeddings and Index

Embedding model:

```text
intfloat/multilingual-e5-large-instruct
```

Embedding dimension:

```text
1024
```

Qdrant collection:

```text
nepal_gov_documents
```

Stored named vectors:

```text
dense
dense_contextual
bm25
```

The production semantic representation is:

```text
dense_contextual
```

Raw dense vectors remain preserved.

Contextual dense vectors remain separately preserved.

Sparse BM25 vectors remain preserved.

The original passage text remains preserved in the Qdrant payload.

No representation should be deleted merely because another representation currently performs better in production.

---

# Contextual Passage Representation

Current contextual passage construction uses metadata fields such as:

```text
Document
Organization
Document type
Section
Subsection
Article number
Article title
```

followed by:

```text
Content:
<original chunk text>
```

The current contextual passage implementation does **not** automatically extract table captions or table titles from inside the original chunk text.

This became important during the latest NE -> EN Economic Survey retrieval diagnostics.

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

Cross-language BM25 is intentionally skipped in current production because lexical overlap cannot be assumed across languages.

Query-language detection treats Devanagari-containing queries as Nepali and other queries as English.

Candidate depth:

```text
20
```

RRF and dense retrieval remain independently callable.

Production retrieval has **not** been modified during the latest evaluation cycle.

---

# Reranker

Production reranker:

```text
BAAI/bge-reranker-v2-m3
```

Production candidate count:

```text
20
```

Reranker receives the candidate pool returned by first-stage retrieval.

A passage missing from the top-20 first-stage candidate pool cannot be recovered by the reranker.

This distinction was important in the latest NE -> EN diagnostics.

---

# Context Selection

Final selected context:

```text
top 5
```

The context-selection stage reranks the candidate pool and returns a fixed five-passage context.

No production change to context selection was made during the current evaluation cycle.

---

# Generation Architecture

Generation package:

```text
google-genai==2.24.0
```

Production generation model:

```text
gemini-3.8-flash
```

Transport:

```text
Gemini Interactions API
```

Current production call:

```python
interaction = client.interactions.create(
    model=self.model_name,
    input=prompt,
)

answer_text = interaction.output_text
```

Generation uses:

```text
GEMINI_API_KEY
```

Hosted multilingual E5 uses:

```text
HF_TOKEN
```

Hosted reranking configuration remains environment-based.

Secrets must never be committed, printed, pasted into chats, or included in evaluation artifacts.

---

# Evidence Guard

The production evidence guard is intentionally structural.

It rejects:

* missing selected context
* missing required citations
* invalid evidence references
* references to evidence that was not selected

It does not use an arbitrary retrieval-score threshold.

The evidence guard does not itself perform semantic entailment evaluation.

Therefore it is possible for:

```text
structural guard state = accepted
```

while the generated answer itself semantically says that the available evidence is insufficient.

This distinction is intentionally measured separately.

No brittle multilingual string matching has been introduced merely to force structural guard state to mirror semantic answer behavior.

---

# Official Production RAG Benchmark

Artifact:

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

Run configuration:

```text
production-rag-v2-interactions
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

Benchmark notation means:

```text
query_language -> evidence/document language
```

It does **not** mean answer language.

Production answer language is intentionally:

```text
query language
```

Therefore:

```text
EN -> NE
```

means:

```text
English query
Nepali evidence
English answer
```

and:

```text
NE -> EN
```

means:

```text
Nepali query
English evidence
Nepali answer
```

---

# Official Production Structural Results

| Slice | N | Accept | SelHit | SelRec | CitHit | CitPrec | CitRec | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EN -> EN | 12 | 1.000 | 0.833 | 0.792 | 0.833 | 0.533 | 0.792 | 1.000 |
| NE -> NE | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.458 | 0.917 | 1.000 |
| EN -> NE | 6 | 1.000 | 0.833 | 0.861 | 0.833 | 0.347 | 0.778 | 1.000 |
| NE -> EN | 6 | 1.000 | 0.667 | 0.722 | 0.667 | 0.539 | 0.722 | 1.000 |
| **Overall** | **30** | **1.000** | **0.833** | **0.833** | **0.833** | **0.482** | **0.800** | **1.000** |

These are structural evidence-identity metrics.

They are not claim-level factual-faithfulness metrics.

---

# Human Semantic Citation Evaluation

Semantic claim dataset:

```text
data/evaluation/semantic/production_rag_v2_claims.jsonl
```

Total claims:

```text
290
```

Human semantic reference:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_labeled.jsonl
```

Reviewed claims:

```text
48/48
```

Sampling:

```text
12 claims per language pair
```

The human semantic set is a deterministic stratified diagnostic sample.

It is not a population-proportional random sample of all 290 production claims.

Overall claim-level results:

```text
Fully supported                    = 0.950
At least partially supported       = 1.000
Unsupported                        = 0.000
Required citation coverage         = 1.000
```

Individual citation evidence:

```text
Individual citations reviewed      = 48
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

Important terminology correction:

The historical:

```text
IndSup = 0.979
```

means:

```text
individual_at_least_partial_support_rate
```

It does **not** mean individual full support.

The actual individual full-support rate is:

```text
0.917
```

---

# Automated Semantic Judge — Original Human Reference

Persisted predictions:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_judge.jsonl
```

Judge provider:

```text
gemini
```

Judge model:

```text
gemini-3.8-flash
```

Agreement against the original 48 human-reviewed claims:

```text
Semantic exact agreement           = 0.979
Citation-requirement agreement     = 1.000
Individual-evidence agreement      = 0.958
```

The original 48-claim reference contains no human:

```text
unsupported
needs_review
unclear citation requirement
```

examples.

Therefore the latest development cycle added a separate hard-case challenge set.

---

# Original Insufficient-Evidence Benchmark

Dataset:

```text
data/evaluation/insufficient_evidence_questions.jsonl
```

Questions:

```text
16
```

Persisted run:

```text
data/evaluation/rag_runs/insufficient_evidence_v1.jsonl
```

Human response review:

```text
data/evaluation/rag_runs/insufficient_evidence_v1_response_review.jsonl
```

Language allocation:

```text
EN -> EN: 4
NE -> NE: 4
EN -> NE: 4
NE -> EN: 4
```

Each language pair contains:

```text
2 answerable controls
1 out-of-corpus document case
1 out-of-corpus period case
```

Total:

```text
8 expected-answer
8 expected-withhold
```

Structural results:

| Slice | N | Decision | Answer Accept | Withhold | False Accept | False Withhold |
|---|---:|---:|---:|---:|---:|---:|
| EN -> EN | 4 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| NE -> NE | 4 | 0.750 | 1.000 | 0.500 | 0.500 | 0.000 |
| EN -> NE | 4 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| NE -> EN | 4 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| **Overall** | **16** | **0.938** | **1.000** | **0.875** | **0.125** | **0.000** |

The one structural false accept was:

```text
ie_ne_ne_004
```

The model-generated answer itself correctly abstained because the requested period was not available.

Therefore structural guard state and semantic response behavior were evaluated separately.

---

# Original Human Response-Level Abstention Results

Human-reviewed responses:

```text
16/16
```

Results:

| Slice | N | Done | Behavior Accuracy | Answer Delivery | Abstention Success | Unsafe Answer | Guard Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN -> EN | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| NE -> NE | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.750 |
| EN -> NE | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| NE -> EN | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| **Overall** | **16** | **1.000** | **1.000** | **1.000** | **1.000** | **0.000** | **0.938** |

All eight answerable controls received substantive answers.

All eight expected-withhold cases semantically abstained.

Unsafe substantive answers:

```text
0
```

---

# Production Quality Checkpoint V1

Implementation:

```text
src/evaluation/production_quality_checkpoint.py
```

Artifact:

```text
data/evaluation/production_quality_checkpoint_v1.json
```

Configuration:

```text
production-quality-checkpoint-v1
```

V1 consolidates:

1. official production structural benchmark
2. human semantic citation review
3. automated semantic-judge agreement
4. original insufficient-evidence benchmark
5. original human response-behavior review

V1 was committed as:

```text
99e10df — Add production quality checkpoint
```

Status checkpoint after that four-milestone cycle:

```text
c2c6f08 — Update production quality checkpoint
```

---

# Latest Four-Milestone Cycle

The latest four-milestone cycle is complete.

Milestones:

```text
1. Expand insufficient-evidence benchmark with partial/mixed cases
2. Expand semantic-judge evaluation with hard negative and ambiguous classes
3. Benchmark targeted NE -> EN retrieval improvements
4. Add answer completeness and factual-fidelity evaluation and checkpoint V2
```

Primary commits:

```text
f44c4be — Expand insufficient evidence benchmark
1d091e0 — Add semantic judge hard cases
d6fa4e8 — Benchmark NE to EN retrieval improvements
1e7619d — Add answer quality evaluation checkpoint
```

No experimental retrieval strategy was promoted to production during this cycle.

The official 30-question production benchmark therefore remains the production baseline.

---

# Latest Cycle Milestone 1 — Partial and Mixed Evidence Benchmark

Commit:

```text
f44c4be — Expand insufficient evidence benchmark
```

The original 16-question benchmark remains unchanged.

New dataset:

```text
data/evaluation/insufficient_evidence_partial_mixed_v1.jsonl
```

Persisted hosted output:

```text
data/evaluation/rag_runs/insufficient_evidence_partial_mixed_v1.jsonl
```

Human review:

```text
data/evaluation/rag_runs/insufficient_evidence_partial_mixed_v1_response_review.jsonl
```

Extension size:

```text
8
```

Case types:

```text
4 partial_evidence
4 mixed_supported_unsupported
```

Language allocation:

```text
2 EN -> EN
2 NE -> NE
2 EN -> NE
2 NE -> EN
```

All extension cases use:

```text
expected_behavior = partial
```

The evaluator now distinguishes:

```text
answer
partial
withhold
```

For partial-evidence cases, structural acceptance is expected so the supported portion remains available.

Human review separately determines whether unsupported portions are safely limited.

## Structural Results

| Slice | N | Decision | Partial Accept | False Accept | False Withhold |
|---|---:|---:|---:|---:|---:|
| EN -> EN | 2 | 1.000 | 1.000 | 0.000 | 0.000 |
| NE -> NE | 2 | 1.000 | 1.000 | 0.000 | 0.000 |
| EN -> NE | 2 | 1.000 | 1.000 | 0.000 | 0.000 |
| NE -> EN | 2 | 1.000 | 1.000 | 0.000 | 0.000 |
| **Overall** | **8** | **1.000** | **1.000** | **0.000** | **0.000** |

## Human Response Results

| Slice | N | Done | Behavior Accuracy | Partial Success | Unsafe |
|---|---:|---:|---:|---:|---:|
| EN -> EN | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| NE -> NE | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| EN -> NE | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| NE -> EN | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| **Overall** | **8** | **1.000** | **1.000** | **1.000** | **0.000** |

The tested partial and mixed-evidence responses successfully preserved the supported answer portion while limiting unsupported material.

One generated partial answer contained a localized numerical error:

```text
ie_pm_ne_ne_002
```

Generated value:

```text
33.8%
```

Source value:

```text
33.6%
```

This error was intentionally carried forward into the later factual-fidelity evaluation.

---

# Latest Cycle Milestone 2 — Hard Semantic-Judge Challenge Set

Commit:

```text
1d091e0 — Add semantic judge hard cases
```

Human challenge reference:

```text
data/evaluation/semantic/semantic_judge_hard_cases_v1.jsonl
```

Automated judge output:

```text
data/evaluation/semantic/semantic_judge_hard_cases_v1_judge.jsonl
```

Reference configuration:

```text
semantic-judge-hard-cases-v1
```

Judge configuration:

```text
semantic-judge-hard-cases-v1-judge
```

Challenge claims:

```text
12
```

Allocation:

```text
3 per language pair
```

Hard classes:

```text
4 unsupported
4 needs_review
4 citation-requirement unclear
```

Overall challenge agreement:

```text
Semantic exact                    = 0.667
Citation-requirement exact        = 0.667
Individual-evidence exact         = 0.667
```

Hard-class behavior:

```text
unsupported:
    semantic exact                = 4/4
    citation requirement exact    = 4/4
    individual evidence exact     = 4/4

needs_review:
    semantic exact                = 0/4
    citation requirement exact    = 4/4
    individual evidence exact     = 0/4

citation-requirement unclear:
    semantic exact                = 4/4
    citation requirement exact    = 0/4
    individual evidence exact     = 4/4
```

Observed ambiguity pattern:

```text
human needs_review
    -> automated judge partially_supported

human citation requirement unclear
    -> automated judge required
```

The pattern occurred across all four language pairs.

Interpretation:

The automated semantic judge handles clear hard negatives correctly but is over-decisive when the human reference intentionally represents ambiguity.

Human labels remain authoritative.

The semantic judge remains evaluation-only.

---

# Latest Cycle Milestone 3 — Targeted NE -> EN Retrieval Experiments

Commit:

```text
d6fa4e8 — Benchmark NE to EN retrieval improvements
```

Persisted result artifact:

```text
data/evaluation/retrieval_runs/ne_en_targeted_retrieval_v1.json
```

Evaluation slice:

```text
6 NE -> EN production benchmark questions
```

Production baseline:

```text
original Nepali query
    -> English dense_contextual retrieval
```

Controlled experimental path:

```text
manually controlled English counterpart
    -> English dense_contextual top 100
    + English BM25 top 100
    -> RRF
    -> final top 20
    -> BGE reranking using original Nepali query
```

The controlled English counterparts are evaluation controls only.

No automatic translation component has been implemented or validated.

No production query translation was introduced.

---

# NE -> EN Persistent Failure

The persistent production failure investigated was:

```text
ne_en_005
```

Question:

```text
आर्थिक सर्वेक्षणले नेपालको आर्थिक वृद्धिबारे के जानकारी दिएको छ?
```

Gold passage:

```text
332cdfc6-3b19-564d-ac21-64a6fae52238
```

Document:

```text
economic_survey_2023_24_en
```

Gold pages:

```text
314-315
```

The passage is table-heavy and contains information around:

```text
Gross Domestic Product (GDP)
Annual Growth Rate of GDP by Economic Activities
```

---

# NE -> EN Translation Diagnostic

Primary gold ranks:

```text
Original Nepali contextual dense      >100
Controlled English contextual dense     85
```

Translation alone therefore did not recover the primary evidence inside the production candidate pool.

---

# NE -> EN Gold-Document Oracle Diagnostic

Gold-document restriction produced:

```text
ne_en_005 normal retrieval             >100
ne_en_005 gold-document-only retrieval >100
```

Therefore cross-document competition was not the main cause.

The retrieval weakness exists inside the correct Economic Survey document.

---

# Raw vs Contextual Representation Diagnostic

Within the correct document, depth 300:

```text
Nepali query:
    raw dense                  = 124
    dense_contextual           = >300

Controlled English query:
    raw dense                  = 36
    dense_contextual           = 80
```

For this table-heavy passage, the current contextual representation hurts retrieval substantially.

Even the strongest simple raw-dense result still remained outside the production top-20 candidate pool.

---

# Generic Query Expansion Diagnostic

Generic GDP-related reformulations were tested.

Representative results:

```text
generic GDP query:
    raw dense global rank          = 58
    contextual global rank         = 145

GDP annual-growth query:
    raw dense global rank          = 64
    contextual global rank         = 150

oracle-like table-title query:
    raw dense global rank          = 11
    contextual global rank         = 50
```

Generic query expansion did not solve the problem.

Only an oracle-like query containing vocabulary close to the actual table title moved the raw passage into global top 20.

Therefore a deployable query rewriter should not simply guess answer-specific table vocabulary.

---

# Table-Aware Representation Diagnostic

Three gold-passage representations were tested:

```text
1. raw
2. caption moved to front
3. structured metadata:
   Document: Economic Survey 2023/24
   Table: Annual Growth Rate of GDP by Economic Activities
   Content:
   <raw table passage>
```

Estimated global raw-dense ranks:

```text
Nepali query:
    raw                         = 182
    caption front               = 208
    structured table           = 29

Controlled English query:
    raw                         = 71
    caption front               = 72
    structured table           = 7
```

Merely moving the table caption to the front was not helpful.

Explicit labeled table structure substantially improved semantic representation.

However, the Nepali query still produced approximately:

```text
rank 29
```

which remains outside production candidate depth 20.

This is a promising future representation experiment, but no index backfill was performed.

---

# Translated English Hybrid Diagnostic

Using the controlled English query allows lexical BM25 retrieval against the English corpus.

For `ne_en_005`:

```text
controlled English contextual dense rank = 85
controlled English BM25 rank             = 46
```

At deeper component retrieval, hybrid RRF recovered the gold passage.

Candidate-depth diagnostic:

```text
Component depth 60:
    final top-20 gold rank = absent

Component depth 100:
    final top-20 gold rank = 13

Component depth 160:
    final top-20 gold rank = 14

Component depth 200:
    final top-20 gold rank = 15

Component depth 300:
    final top-20 gold rank = 19

Component depth 400:
    final top-20 gold rank = absent
```

RRF behavior was not monotonic with increasing component depth.

Depth 100 was the shallowest tested component depth that recovered the gold passage inside the final top-20 pool.

---

# Six-Question NE -> EN First-Stage Benchmark

Primary-evidence ranks:

| Question | Production | Controlled English | Translated Hybrid | Dual Route |
|---|---:|---:|---:|---:|
| ne_en_001 | 5 | 1 | 7 | 5 |
| ne_en_002 | 2 | 2 | 1 | 1 |
| ne_en_003 | 2 | 5 | 6 | 3 |
| ne_en_004 | 1 | 2 | 1 | 1 |
| ne_en_005 | >100 | 85 | 13 | >20 |
| ne_en_006 | 1 | 1 | 1 | 1 |

First-stage metrics at @20:

```text
production:
    Hit                         = 0.833
    MRR                         = 0.533
    Recall                      = 0.778

controlled English:
    Hit                         = 0.833
    MRR                         = 0.533
    Recall                      = 0.833

translated hybrid:
    Hit                         = 1.000
    MRR                         = 0.564
    Recall                      = 1.000

dual-route RRF:
    Hit                         = 0.833
    MRR                         = 0.589
    Recall                      = 0.833
```

Translated hybrid was the first tested candidate-generation strategy that recovered every primary passage inside the final top-20 candidate pool.

---

# NE -> EN BGE Reranker Benchmark

Both candidate pools were reranked using the **original Nepali user query**.

This keeps the BGE query condition identical and isolates candidate-generation quality.

Primary-rank movement:

| Question | Production First | Production BGE | Translated First | Translated BGE |
|---|---:|---:|---:|---:|
| ne_en_001 | 5 | 1 | 7 | 1 |
| ne_en_002 | 2 | 1 | 1 | 1 |
| ne_en_003 | 2 | 6 | 6 | 5 |
| ne_en_004 | 1 | 1 | 1 | 1 |
| ne_en_005 | >20 | >20 | 13 | 6 |
| ne_en_006 | 1 | 1 | 1 | 2 |

At @5:

```text
production BGE:
    Hit                         = 0.667
    MRR                         = 0.667
    Recall                      = 0.722

translated-hybrid BGE:
    Hit                         = 0.833
    MRR                         = 0.617
    Recall                      = 0.833
```

At @10:

```text
production BGE:
    Hit                         = 0.833
    MRR                         = 0.694
    Recall                      = 0.778

translated-hybrid BGE:
    Hit                         = 1.000
    MRR                         = 0.644
    Recall                      = 1.000
```

At @20:

```text
production BGE:
    Hit                         = 0.833
    MRR                         = 0.694
    Recall                      = 0.778

translated-hybrid BGE:
    Hit                         = 1.000
    MRR                         = 0.644
    Recall                      = 1.000
```

Important persistent case:

```text
ne_en_005
```

moved from:

```text
production first stage   >20
production BGE           >20
```

to:

```text
translated first stage    13
translated BGE              6
```

This confirms that candidate generation was the limiting factor.

BGE could meaningfully promote the passage once it was available in the candidate pool.

---

# NE -> EN Production Decision

The translated-hybrid experiment showed a real measured improvement.

However, it depends on:

```text
manually controlled English query counterparts
```

It does not yet establish that an automatic production translation/reformulation mechanism would provide the same benefit.

Unmeasured factors include:

* automatic translation accuracy
* translation latency
* translation cost
* query meaning preservation
* behavior on all 30 benchmark questions
* regressions on same-language retrieval
* production failure modes

Therefore:

```text
production retrieval changed = False
```

The experiment remains evaluation-only.

No retrieval backfill or production route change was performed.

---

# Latest Cycle Milestone 4 — Answer Completeness and Factual Fidelity

Commit:

```text
1e7619d — Add answer quality evaluation checkpoint
```

Implementation:

```text
src/evaluation/answer_quality_review.py
```

Tests:

```text
tests/test_answer_quality_review.py
```

Human review artifact:

```text
data/evaluation/rag_runs/production_rag_v2_answer_quality_review_v1.jsonl
```

Reviewed responses:

```text
30/30
```

Needs-review responses:

```text
0
```

The review uses exact original Qdrant passages for both:

```text
gold reference evidence
selected RAG evidence
```

The evaluation keeps two dimensions separate:

```text
completeness
factual fidelity
```

---

# Answer Completeness Labels

Supported labels:

```text
complete
mostly_complete
incomplete
needs_review
```

Interpretation:

```text
complete
    Central answer and all material verified gold information are covered.

mostly_complete
    Central answer is present, but secondary material information is omitted.

incomplete
    Central evidence, primary answer, or another major requested component is missing.

needs_review
    Human reviewer cannot classify completeness confidently.
```

---

# Factual Fidelity Labels

Supported labels:

```text
fully_faithful
minor_issue
major_issue
needs_review
```

Interpretation:

```text
fully_faithful
    No material factual error or unsupported factual assertion.

minor_issue
    Localized factual imprecision that does not change the central answer.

major_issue
    Material unsupported, contradicted, or substantially incorrect answer content.

needs_review
    Fidelity cannot be classified confidently.
```

---

# Human Answer-Quality Results

Per language pair:

| Slice | N | Done | Scored | FullComp | >=Mostly | Faithful | NoMajor | Strong | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EN -> EN | 12 | 1.000 | 1.000 | 0.667 | 0.667 | 0.917 | 1.000 | 0.667 | 0.667 |
| NE -> NE | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 0.833 | 1.000 |
| EN -> NE | 6 | 1.000 | 1.000 | 0.667 | 0.833 | 1.000 | 1.000 | 0.667 | 0.833 |
| NE -> EN | 6 | 1.000 | 1.000 | 0.500 | 0.833 | 1.000 | 1.000 | 0.500 | 0.833 |
| **Overall** | **30** | **1.000** | **1.000** | **0.700** | **0.800** | **0.933** | **1.000** | **0.667** | **0.800** |

Completeness counts:

```text
complete              = 21
mostly_complete       = 3
incomplete            = 6
needs_review          = 0
```

Completeness rates:

```text
Fully complete                    = 21/30 = 0.700
At least mostly complete          = 24/30 = 0.800
Incomplete                        =  6/30 = 0.200
```

Factual-fidelity counts:

```text
fully_faithful        = 28
minor_issue           = 2
major_issue           = 0
needs_review          = 0
```

Factual-fidelity rates:

```text
Fully faithful                    = 28/30 = 0.933
No major factual error            = 30/30 = 1.000
Major factual issue               =  0/30 = 0.000
```

Combined quality:

```text
Strong answer                     = 20/30 = 0.667
Acceptable answer                 = 24/30 = 0.800
```

A strong answer requires:

```text
complete
+
fully_faithful
```

An acceptable answer allows:

```text
complete OR mostly_complete
+
fully_faithful OR minor_issue
```

---

# Important Answer-Quality Findings

The new review shows that:

```text
factual correctness is stronger than answer completeness
```

No reviewed answer contained a major factual error.

However:

```text
6/30
```

answers were materially incomplete.

Therefore the next major quality improvement should not focus only on hallucination prevention.

It should also focus on getting the correct and sufficiently complete evidence into the answer.

The weakest full-completeness slice was:

```text
NE -> EN = 0.500
```

This aligns with the previously measured NE -> EN retrieval weakness.

---

# Numerical Fidelity Finding

The answer-quality review successfully surfaced the known localized numerical mismatch:

```text
Generated:
33.8%

Evidence:
33.6%
```

This is classified as:

```text
minor_issue
```

because it is a localized numerical error rather than a central-answer failure.

The second minor factual-fidelity issue is associated with partially visible/truncated evidence around an education indicator.

No major factual error was observed in the 30-question review.

---

# Production Quality Checkpoint V2

Implementation:

```text
src/evaluation/production_quality_checkpoint_v2.py
```

Tests:

```text
tests/test_production_quality_checkpoint_v2.py
```

Artifact:

```text
data/evaluation/production_quality_checkpoint_v2.json
```

Configuration:

```text
production-quality-checkpoint-v2
```

V2 preserves:

```text
production-quality-checkpoint-v1
```

and adds the latest four-milestone cycle.

The V2 checkpoint is deterministic over persisted evaluation artifacts.

It performs no:

* hosted generation
* embedding
* retrieval
* reranking
* OCR
* ingestion
* vector backfill

---

# Production Quality Checkpoint V2 Headline

Partial/mixed evidence:

```text
Structural decision accuracy       = 1.000
Partial-case acceptance            = 1.000
Human behavior accuracy            = 1.000
Partial-response success           = 1.000
```

Hard semantic-judge challenge:

```text
Semantic exact                     = 0.667
Citation-requirement exact         = 0.667
Individual-evidence exact          = 0.667
```

Targeted NE -> EN BGE @10:

```text
Production:
    Hit                            = 0.833
    Recall                         = 0.778

Experimental translated hybrid:
    Hit                            = 1.000
    Recall                         = 1.000
```

Human answer quality:

```text
Fully complete                    = 0.700
At least mostly complete          = 0.800
Fully faithful                    = 0.933
No major factual error            = 1.000
Strong answer                     = 0.667
Acceptable answer                 = 0.800
```

Production decision:

```text
Production retrieval changed          = False
Hosted production rerun performed     = False
Official production benchmark retained = True
```

---

# Why the Official Production Benchmark Was Not Rerun

The latest retrieval improvement was experimental only.

No production retrieval component was changed.

No production contextual vector representation was changed.

No new query translation component was added.

No production candidate depth was changed.

No reranker was changed.

No context-selection behavior was changed.

No generation prompt or generation model was changed.

Therefore a new 30-question hosted Gemini production run would not measure a new production system.

The current official benchmark remains:

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

A new production benchmark should be generated only after an actual production component is deliberately promoted.

---

# Current Quality Headline

Official production structural benchmark:

```text
Selected primary hit              = 0.833
Selected relevant recall          = 0.833
Valid citation references         = 1.000
```

Human semantic claim review:

```text
Fully supported                   = 0.950
At least partially supported      = 1.000
Unsupported                       = 0.000
Required citation coverage        = 1.000
Individual full support           = 0.917
Individual at least partial       = 0.979
```

Automated semantic judge on original human set:

```text
Semantic exact                    = 0.979
Citation requirement             = 1.000
Individual evidence              = 0.958
```

Automated judge on hard challenge set:

```text
Semantic exact                    = 0.667
Citation requirement             = 0.667
Individual evidence              = 0.667
```

Original insufficient-evidence structural benchmark:

```text
Decision accuracy                 = 0.938
Withholding success               = 0.875
Structural false accept           = 0.125
Structural false withhold         = 0.000
```

Original human response behavior:

```text
Behavior accuracy                 = 1.000
Semantic abstention success       = 1.000
Unsafe substantive answer rate    = 0.000
Structural/behavior agreement     = 0.938
```

Partial/mixed evidence extension:

```text
Structural decision accuracy      = 1.000
Partial response success          = 1.000
Unsafe answer rate                = 0.000
```

Human answer quality:

```text
Fully complete                    = 0.700
At least mostly complete          = 0.800
Fully faithful                    = 0.933
No major factual error            = 1.000
Strong answer                     = 0.667
Acceptable answer                 = 0.800
```

---

# Current Measured Strengths

## 1. Citation References Are Structurally Reliable

Official benchmark:

```text
Valid citation reference ratio = 1.000
```

Invalid evidence references are not silently accepted.

---

## 2. Claim-Level Semantic Support Is Strong

Human-reviewed factual claims:

```text
At least partial support = 1.000
```

No jointly unsupported factual claim was found in the 48-claim stratified semantic sample.

---

## 3. Response-Level Safety Is Strong

Original insufficient-evidence benchmark:

```text
Unsafe substantive answers = 0
```

Partial/mixed extension:

```text
Unsafe responses = 0
```

Human answer-quality review:

```text
Major factual errors = 0/30
```

---

## 4. Partial-Evidence Behavior Is Strong

All eight new partial/mixed cases produced the expected limited partial-answer behavior.

```text
Partial response success = 1.000
```

---

## 5. Clear Hard Negatives Are Correctly Detected by the Automated Judge

Hard challenge:

```text
unsupported cases = 4/4 correct
```

---

## 6. NE -> EN Candidate Recall Can Be Improved

The translated-hybrid experiment recovered:

```text
all six NE -> EN primary passages
```

inside the final top-20 candidate pool.

This shows the persistent cross-lingual failures are not necessarily unsolvable with the existing corpus and reranker.

---

# Current Measured Weaknesses

## 1. Answer Completeness

Only:

```text
70.0%
```

of reviewed production answers were fully complete.

Only:

```text
80.0%
```

were at least mostly complete.

This is now a more important measured quality gap than major factual hallucination.

---

## 2. NE -> EN Retrieval

Official production:

```text
NE -> EN selected primary hit      = 0.667
NE -> EN selected relevant recall  = 0.722
```

This remains the weakest production language direction.

Experimental translated hybrid improves candidate recall substantially, but the translation component is not yet production-ready.

---

## 3. Table-Heavy Evidence Representation

The current contextual passage representation can perform poorly on table-heavy Economic Survey chunks.

The `ne_en_005` diagnostic showed:

```text
contextual representation can rank substantially worse than raw dense
```

for the relevant GDP table.

Structured table metadata appears promising.

---

## 4. Automated Judge Ambiguity Handling

The automated semantic judge is too decisive for intentionally ambiguous cases.

Observed:

```text
needs_review -> partially_supported
unclear citation requirement -> required
```

Human review remains necessary for ambiguity-sensitive evaluation.

---

## 5. Numerical Fidelity

At least one clear generated numerical mismatch was observed:

```text
33.8% generated
33.6% source
```

The new factual-fidelity layer now detects this explicitly.

---

## 6. OCR Evidence Quality

The forced-OCR Nepali Economic Survey contains occasional OCR corruption.

This can affect:

* retrieval
* table interpretation
* individual citation support
* numerical fidelity

Do not assume all OCR text is clean.

---

## 7. Citation Efficiency

Some answers cite evidence that is related but does not individually establish the associated claim.

Overall joint support can remain correct while individual citation quality is weaker.

---

# Architecture Principles

The project currently follows these principles:

* retrieval remains independently callable
* reranking remains independently callable
* context selection remains independently callable
* generation never reruns retrieval
* Gemini transport does not own prompt policy
* prompt construction does not call Gemini
* citation processing does not trust model-generated source metadata
* invalid evidence references are never silently accepted
* no-selected-evidence cases skip generation
* original `chunk_text` remains canonical evidence
* raw dense vectors remain preserved
* contextual vectors remain separately preserved
* BM25 vectors remain preserved
* selected evidence order remains stable
* provider SDK objects do not leak downstream
* source provenance remains available end to end
* structured citation results remain machine-readable
* semantic evaluation remains separate from production generation
* human semantic labels remain distinct from automated predictions
* human labels remain the semantic judge reference standard
* retrieval gold overlap is not treated as semantic entailment
* structural guard state is not treated as equivalent to semantic answer behavior
* answer completeness is measured separately from factual fidelity
* experiments are not promoted without evaluation
* expensive preprocessing is not rerun without need
* resumable hosted evaluation avoids repeating completed calls
* prompt hashes protect automated judge predictions from stale prompt reuse
* persisted evaluation artifacts are preferred over unnecessary stochastic reruns
* production changes should be driven by measured failures rather than metric cosmetics
* manually controlled evaluation queries must not be confused with production-ready translation
* production baselines remain immutable until a measured intervention is actually promoted

---

# Test Progression

Generation abstraction:

```text
212 passed
```

Gemini provider:

```text
232 passed
```

Grounded prompt:

```text
246 passed
```

Citation/evidence:

```text
263 passed
```

Insufficient-evidence handling:

```text
278 passed
```

End-to-end RAG orchestration:

```text
295 passed
```

Deterministic RAG evaluation:

```text
310 passed
```

Production RAG runner and subsequent validation:

```text
325 passed
```

Claim citation and semantic dataset work:

```text
359 passed
```

Stratified semantic review:

```text
372 passed
```

Human semantic review tooling:

```text
385 passed
```

Combined citation parser regression coverage:

```text
388 passed
```

Original insufficient-evidence benchmark:

```text
445 passed
```

Human response-level abstention evaluation:

```text
458 passed
```

Production quality checkpoint V1:

```text
467 passed
```

After partial/mixed evidence benchmark:

```text
504 passed
```

After hard semantic-judge challenge set:

```text
543 passed
```

After initial targeted NE -> EN retrieval tests:

```text
558 passed
```

Current validated full suite after answer-quality evaluation and checkpoint V2:

```text
604 passed in 3.35s
```

---

# Git Milestone History

## Latest four-milestone cycle

Partial/mixed insufficient-evidence benchmark:

```text
f44c4be — Expand insufficient evidence benchmark
```

Semantic judge hard cases:

```text
1d091e0 — Add semantic judge hard cases
```

Targeted NE -> EN retrieval benchmark:

```text
d6fa4e8 — Benchmark NE to EN retrieval improvements
```

Answer-quality evaluation and checkpoint V2:

```text
1e7619d — Add answer quality evaluation checkpoint
```

---

## Previous production-quality cycle

Automated semantic judge:

```text
7a62a77 — Add automated semantic judge evaluation
```

Insufficient-evidence benchmark:

```text
c3d60b7 — Add insufficient evidence benchmark
```

Human response-level abstention evaluation:

```text
d622873 — Add response abstention evaluation
```

Response-review metadata correction:

```text
ab91220 — Correct response review language notes
```

Production quality checkpoint:

```text
99e10df — Add production quality checkpoint
```

Status checkpoint:

```text
c2c6f08 — Update production quality checkpoint
```

---

## Previous semantic-evaluation cycle

Status checkpoint:

```text
2de6b86 — Update semantic evaluation checkpoint
```

Human semantic citation evaluation:

```text
750966f — Add human semantic citation evaluation
```

Stratified semantic review subset:

```text
47cab25 — Add stratified semantic review subset
```

Multilingual claim/parser correction:

```text
984710e — Fix multilingual claim extraction artifacts
```

Semantic dataset:

```text
796cce7 — Add semantic citation evaluation dataset
```

Claim citation alignment:

```text
a0c0e75 — Add claim citation alignment evaluation
```

---

## Production benchmark cycle

```text
1bc404b — Record production RAG benchmark
e9ab5ec — Migrate Gemini generation to Interactions API
9d14f05 — Add resumable RAG evaluation runner
80058cf — Add deterministic RAG evaluation metrics
```

---

## Earlier architecture milestones

```text
4691192 — Add end-to-end RAG orchestration
b67d42b — Add insufficient evidence handling
83af4f5 — Add evidence citation processing
c25ec2f — Add grounded Gemini smoke test
bb387e6 — Add grounded generation prompt
99cb986 — Add Gemini generation provider
e0332a2 — Add generation service abstraction
e9642ee — Add evaluated context selection pipeline
4c38b3b — Promote title-aware multilingual reranking
```

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

Original automated judge:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_judge.jsonl
```

Original insufficient-evidence benchmark:

```text
data/evaluation/insufficient_evidence_questions.jsonl
```

Original insufficient-evidence hosted run:

```text
data/evaluation/rag_runs/insufficient_evidence_v1.jsonl
```

Original human response review:

```text
data/evaluation/rag_runs/insufficient_evidence_v1_response_review.jsonl
```

Partial/mixed benchmark:

```text
data/evaluation/insufficient_evidence_partial_mixed_v1.jsonl
```

Partial/mixed hosted run:

```text
data/evaluation/rag_runs/insufficient_evidence_partial_mixed_v1.jsonl
```

Partial/mixed human response review:

```text
data/evaluation/rag_runs/insufficient_evidence_partial_mixed_v1_response_review.jsonl
```

Hard semantic judge reference:

```text
data/evaluation/semantic/semantic_judge_hard_cases_v1.jsonl
```

Hard semantic judge predictions:

```text
data/evaluation/semantic/semantic_judge_hard_cases_v1_judge.jsonl
```

Targeted NE -> EN retrieval results:

```text
data/evaluation/retrieval_runs/ne_en_targeted_retrieval_v1.json
```

Human answer-quality review:

```text
data/evaluation/rag_runs/production_rag_v2_answer_quality_review_v1.jsonl
```

Production checkpoint V1:

```text
data/evaluation/production_quality_checkpoint_v1.json
```

Production checkpoint V2:

```text
data/evaluation/production_quality_checkpoint_v2.json
```

---

# Expensive Operations

Do not rerun without a clear reason:

* forced OCR
* corpus ingestion
* dense embedding ingestion
* contextual-vector backfill
* full retrieval benchmark involving hosted embedding calls
* full BGE reranker benchmark
* completed 30-question hosted Gemini production benchmark
* completed semantic-judge calls

The current corpus and stored vectors remain valid for the current production architecture.

No contextual-vector backfill should be performed merely because the structured-table representation diagnostic looked promising.

A new representation must first be benchmarked carefully.

---

# Development Environment

Primary development environment:

```text
Windows
Python 3.12 virtual environment
Windows CMD
Docker-hosted Qdrant
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

Avoid WSL unless explicitly required.

Do not disable Smart App Control.

---

# Git and OneDrive Note

The working clone is stored under OneDrive.

Git automatic housekeeping previously produced deletion/retry problems under:

```text
.git/objects
```

Repository integrity was checked with:

```text
git fsck --full
```

No corruption was found.

Automatic Git GC was disabled locally:

```text
git config --local gc.auto 0
```

Keep this workaround for the current clone.

Do not routinely run:

```text
git gc
```

Do not manually delete:

```text
.git/objects/*
```

If a stale Git lock appears after an interrupted command:

1. verify that no Git process is still active
2. remove only the specific stale lock if necessary

A future fresh clone outside OneDrive may be safer, but it is not required for current milestone work.

---

# Secrets

Never commit, print, expose, or ask the user to paste:

```text
HF_TOKEN
HF_RERANKER_ENDPOINT_URL
GEMINI_API_KEY
```

Hosted credentials remain local environment configuration.

Evaluation artifacts must not contain credentials.

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

Avoid:

```text
git add .
```

when unrelated files may exist.

Before commit:

```text
git diff --cached --check
git diff --cached --stat
git status --short
```

Commit the completed milestone.

Push explicitly:

```text
git push origin main
```

Verify after push:

```text
git status --short
git log -1 --oneline
```

A milestone is complete only after it is committed and pushed.

---

# Project Status Update Cadence

Update:

```text
docs/project_status.md
```

after every **four completed milestones**.

Do not update it after every individual milestone.

An earlier update is appropriate only when required by:

* a major architectural reset
* an explicit handoff
* correction of project state that would otherwise mislead future work

When this document is updated, replace the entire file rather than editing scattered sections.

---

# Current Four-Milestone Checkpoint

Checkpoint title:

**Hard-Case Evaluation, Cross-Lingual Retrieval Diagnostics, and Answer Quality**

Completed milestones:

```text
1. Partial/mixed insufficient-evidence benchmark
2. Hard semantic-judge challenge evaluation
3. Targeted NE -> EN retrieval benchmark
4. Answer completeness and factual-fidelity evaluation
```

Milestone commits:

```text
f44c4be — Expand insufficient evidence benchmark
1d091e0 — Add semantic judge hard cases
d6fa4e8 — Benchmark NE to EN retrieval improvements
1e7619d — Add answer quality evaluation checkpoint
```

Current validated full suite:

```text
604 passed in 3.35s
```

Current pushed repository head before this status-document checkpoint commit:

```text
1e7619d — Add answer quality evaluation checkpoint
```

---

# Current Production Decisions

## Retrieval

Same-language:

```text
dense_contextual + BM25 -> RRF
```

Cross-language:

```text
dense_contextual only
```

## Candidate Depth

```text
20
```

## Reranker

```text
BAAI/bge-reranker-v2-m3
```

## Final Context

```text
top 5
```

## Generation

```text
Gemini
gemini-3.8-flash
Interactions API
```

## Prompt

```text
GroundedPromptBuilder
```

## Evidence IDs

```text
E1
E2
E3
...
```

Combined citation syntax is also supported:

```text
[E1, E2]
```

## Guard

```text
Structural evidence/citation validation
```

No arbitrary retrieval threshold.

## Production Benchmark

```text
production_rag_v2_interactions.jsonl
```

No new production run has replaced it.

---

# Recommended Next Development Cycle

The latest evaluation cycle identified two principal measured gaps:

```text
1. NE -> EN candidate retrieval
2. answer completeness
```

A sensible next four-milestone cycle is:

## Milestone 1 — Automatic Cross-Lingual Query Reformulation Benchmark

Replace the manually controlled English evaluation query with one or more realistic automatic strategies.

Possible controlled experiments:

```text
Nepali original only
automatic English translation only
original + translated fusion
translated dense + BM25
```

Measure:

* retrieval hit
* MRR
* recall
* latency
* hosted-call cost
* translation failures
* meaning preservation

Do not change production yet.

---

## Milestone 2 — Table-Aware Passage Representation Benchmark

Develop an evaluation-only representation for table-heavy passages.

Potential format:

```text
Document: ...
Table: ...
Section: ...
Content:
...
```

Benchmark it against:

```text
raw dense
current contextual dense
table-aware contextual representation
```

Focus initially on table-heavy Economic Survey failures.

Do not backfill the production collection until the representation is shown to improve the benchmark without harmful regressions.

---

## Milestone 3 — Answer Completeness Improvement Experiment

Use the 30-question answer-quality review as a diagnostic set.

Focus specifically on the six currently incomplete answers.

Possible causes should be separated:

```text
retrieval miss
gold passage outside selected top 5
generator omission despite correct evidence
overly broad question
table evidence interpretation
```

Potential interventions may include:

* retrieval improvements
* context-selection changes
* completeness-aware generation instructions
* evidence coverage checks

Preserve factual fidelity while improving completeness.

Do not optimize only for the six failure cases without checking regression behavior.

---

## Milestone 4 — Production Promotion Decision

If a retrieval, representation, context, or generation intervention shows a clear improvement:

1. promote exactly one measured production change
2. rerun the official 30-question benchmark
3. rerun answer-quality evaluation
4. compare against the current immutable baseline
5. create the next production-quality checkpoint
6. update this status document

If no intervention justifies promotion, keep production unchanged and record the negative result rather than forcing a change.

---

# Remaining Major Work

Evaluation and quality:

1. automatic query translation/reformulation evaluation
2. table-aware passage representation benchmark
3. answer completeness improvement
4. numerical fidelity regression tests
5. date fidelity evaluation
6. legal qualification and exception preservation
7. citation-efficiency evaluation
8. larger human semantic evaluation
9. OCR quality diagnostics
10. automated judge ambiguity calibration

Retrieval:

11. production-worthy NE -> EN retrieval improvement
12. targeted table-heavy retrieval improvement
13. automatic translated-query fusion
14. candidate-depth cost/performance evaluation
15. possible query-instruction experiments

Generation:

16. completeness-aware prompt experiments
17. evidence-coverage preservation
18. citation-efficiency improvements
19. controlled model comparison only if justified by evaluation

Application:

20. FastAPI application layer
21. Streamlit or equivalent interface
22. source rendering
23. structured logging
24. error handling
25. user feedback mechanisms

Operations:

26. Docker/Compose application integration
27. CI refinement
28. environment/configuration refinement
29. MLflow or equivalent experiment tracking

Documentation and portfolio:

30. README architecture documentation
31. benchmark methodology documentation
32. system architecture diagram
33. screenshots/demo
34. portfolio presentation

Potential experimental work only if justified:

* parent/child chunking
* alternative multilingual embedding models
* alternative rerankers
* automatic semantic validation
* advanced prompt-injection defenses
* richer document/table parsing

Experiments remain separate from production until measured.

---

# Immediate Next Action

The latest four milestones are complete and pushed.

The only remaining task for this checkpoint is to commit and push this updated:

```text
docs/project_status.md
```

After that, begin the next development cycle with an evaluation-only automatic cross-lingual query reformulation benchmark.

Do not modify production retrieval before that benchmark demonstrates a deployable improvement.