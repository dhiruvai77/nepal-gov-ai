# NepalGov AI — Current Project Status

Last updated: 2026-09-21

## Purpose of This Document

This file is the project handoff and checkpoint document for NepalGov AI.

It exists so development can continue accurately across ChatGPT conversations without reconstructing architectural decisions, benchmark results, evaluation history, and remaining work from scratch.

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

The project now has evaluation at five distinct levels:

1. retrieval and selected-context coverage
2. structural citation correctness
3. human-reviewed claim-level semantic support
4. automated semantic-judge agreement with human labels
5. response-level answer-versus-abstention behavior

The project also has a deterministic consolidated production-quality checkpoint that combines all persisted evaluation layers without rerunning hosted generation.

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
11. structural insufficient-evidence handling
12. application-level RAG orchestration
13. deterministic RAG evaluation metrics
14. resumable production RAG benchmark runner
15. Gemini Interactions API transport
16. complete 30-question production RAG benchmark
17. deterministic claim-to-citation alignment
18. semantic citation dataset with exact passages
19. deterministic stratified human semantic-review subset
20. completed human semantic citation evaluation
21. automated semantic judge with human-label validation
22. dedicated insufficient-evidence benchmark
23. human response-level abstention evaluation
24. deterministic consolidated production-quality checkpoint

Current validated test suite:

```text
467 passed
```

Latest committed and pushed milestone on `main`:

```text
99e10df — Add production quality checkpoint
```

Current repository head before this status-document checkpoint update:

```text
99e10df — Add production quality checkpoint
```

Current official production RAG benchmark:

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

Production benchmark questions:

```text
30
```

Current semantic claim dataset:

```text
data/evaluation/semantic/production_rag_v2_claims.jsonl
```

Semantic claim rows:

```text
290
```

Current human semantic reference set:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_labeled.jsonl
```

Human-reviewed claims:

```text
48/48
```

Current automated semantic-judge output:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_judge.jsonl
```

Automated judge predictions:

```text
48/48
```

Current insufficient-evidence benchmark:

```text
data/evaluation/insufficient_evidence_questions.jsonl
```

Questions:

```text
16
```

Persisted insufficient-evidence run:

```text
data/evaluation/rag_runs/insufficient_evidence_v1.jsonl
```

Human response-level review:

```text
data/evaluation/rag_runs/insufficient_evidence_v1_response_review.jsonl
```

Current consolidated quality checkpoint:

```text
data/evaluation/production_quality_checkpoint_v1.json
```

Next major development direction:

**Expand hard-case evaluation coverage and improve measured retrieval weaknesses without changing production speculatively.**

---

# Latest Four-Milestone Checkpoint

This checkpoint captures the following completed milestones:

1. Automated Semantic Judge Validation
2. Insufficient-Evidence Benchmark
3. Human Response-Level Abstention Evaluation
4. Consolidated Production-Quality Checkpoint

This evaluation cycle established an important distinction between:

```text
structural guard state
```

and:

```text
actual user-visible semantic behavior
```

It also validated an automated semantic judge against human labels before using it as a broader evaluation instrument.

No production retrieval, reranking, context-selection, prompt, generation, citation-processing, or evidence-guard behavior was changed during this four-milestone cycle.

Therefore the existing official 30-question hosted production benchmark remains the current production baseline.

A new 30-question Gemini generation run was intentionally not performed because it would have been another stochastic sample of an unchanged production system rather than a measurement of a production intervention.

---

# Milestone 1 — Automated Semantic Judge Validation

Committed and pushed as:

```text
7a62a77 — Add automated semantic judge evaluation
```

Implementation:

```text
src/evaluation/semantic_judge.py
src/evaluation/run_semantic_judge.py
```

Tests:

```text
tests/test_semantic_judge.py
tests/test_run_semantic_judge.py
```

Persisted predictions:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_judge.jsonl
```

Reference set:

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_labeled.jsonl
```

Human reference claims:

```text
48
```

Automated predictions:

```text
48
```

Judge provider:

```text
gemini
```

Judge model:

```text
gemini-3.8-flash
```

Run configuration:

```text
production-rag-v2-human-review-v1-semantic-judge-v1
```

The automated judge evaluates three properties:

1. joint semantic support
2. citation requirement
3. individual evidence support

Semantic labels:

```text
supported
partially_supported
unsupported
not_a_factual_claim
needs_review
```

Citation-requirement labels:

```text
required
not_required
unclear
```

Individual-evidence labels:

```text
supported
partially_supported
unsupported
needs_review
```

Human labels remain the reference standard.

Human labels are deliberately excluded from judge prompts.

Persisted predictions include prompt SHA-256 values so predictions generated with an older judge prompt cannot silently be reused after prompt changes.

## Automated Judge Agreement

Final overall agreement:

```text
Semantic exact agreement:       0.979
Citation-requirement agreement: 1.000
Individual-evidence agreement:  0.958
```

Semantic exact count:

```text
47/48
```

Observed human-to-judge semantic confusion:

```text
Human not_a_factual_claim:
    8 -> not_a_factual_claim

Human supported:
    38 -> supported

Human partially_supported:
    1 -> partially_supported
    1 -> supported
```

The only joint semantic disagreement was therefore one human:

```text
partially_supported
```

claim classified by the automated judge as:

```text
supported
```

## Automated Judge Limitation

The 48-claim human reference contains:

```text
0 human unsupported examples
0 human needs_review examples
0 human unclear citation-requirement examples
```

Therefore high overall exact agreement does **not** establish judge reliability on those absent classes.

The semantic judge remains evaluation-only.

It is not part of the production evidence guard.

It must not be treated as authoritative for unsupported or ambiguous classes until those classes have human reference examples.

---

# Milestone 2 — Insufficient-Evidence Benchmark

Committed and pushed as:

```text
c3d60b7 — Add insufficient evidence benchmark
```

Dataset:

```text
data/evaluation/insufficient_evidence_questions.jsonl
```

Persisted run:

```text
data/evaluation/rag_runs/insufficient_evidence_v1.jsonl
```

Implementation:

```text
src/evaluation/insufficient_evidence_evaluator.py
src/evaluation/run_insufficient_evidence_evaluation.py
```

Tests:

```text
tests/test_insufficient_evidence_evaluator.py
tests/test_run_insufficient_evidence_evaluation.py
```

Benchmark size:

```text
16 questions
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
8 expected-answer cases
8 expected-withhold cases
```

Current benchmark case types:

```text
answerable_control
out_of_corpus_document
out_of_corpus_period
```

Supported evaluator case types that are not yet represented in the dataset:

```text
partial_evidence
mixed_supported_unsupported
```

## Structural Insufficient-Evidence Results

| Slice | N | Decision | Answer Accept | Withhold | False Accept | False Withhold |
|---|---:|---:|---:|---:|---:|---:|
| EN -> EN | 4 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| NE -> NE | 4 | 0.750 | 1.000 | 0.500 | 0.500 | 0.000 |
| EN -> NE | 4 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| NE -> EN | 4 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| **Overall** | **16** | **0.938** | **1.000** | **0.875** | **0.125** | **0.000** |

By case type:

| Case type | N | Decision | Withhold |
|---|---:|---:|---:|
| answerable_control | 8 | 1.000 | n/a |
| out_of_corpus_document | 4 | 1.000 | 1.000 |
| out_of_corpus_period | 4 | 0.750 | 0.750 |

The one structural false accept was:

```text
ie_ne_ne_004
```

Query:

```text
आर्थिक सर्वेक्षण २०८२/८३ ले बेरोजगारीबारे के तथ्याङ्क दिएको छ?
```

Expected behavior:

```text
withhold
```

Case type:

```text
out_of_corpus_period
```

The structural evidence guard classified the generation as:

```text
accepted
```

because the generated text contained a valid citation.

However, the generated answer itself explicitly stated that the requested Economic Survey 2082/83 evidence was unavailable and that only 2081/82 evidence was present.

This motivated a separate response-level human evaluation rather than changing production immediately based on structural metrics alone.

---

# Milestone 3 — Human Response-Level Abstention Evaluation

Primary milestone commit:

```text
d622873 — Add response abstention evaluation
```

Follow-up metadata correction:

```text
ab91220 — Correct response review language notes
```

Implementation:

```text
src/evaluation/response_behavior_review.py
```

Tests:

```text
tests/test_response_behavior_review.py
```

Human-reviewed artifact:

```text
data/evaluation/rag_runs/insufficient_evidence_v1_response_review.jsonl
```

The review reuses the exact 16 persisted application outputs from the insufficient-evidence benchmark.

No additional hosted generation calls are required.

The review separates:

```text
structural guard decision
```

from:

```text
application-visible answer behavior
```

Review labels:

```text
substantive_answer
abstained
partial_answer_with_limitation
needs_review
```

Review is resumable and written atomically after each completed item.

## Human Response-Level Results

| Slice | N | Done | Behavior Accuracy | Answer Delivery | Abstention Success | Unsafe Answer | Guard Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN -> EN | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| NE -> NE | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.750 |
| EN -> NE | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| NE -> EN | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| **Overall** | **16** | **1.000** | **1.000** | **1.000** | **1.000** | **0.000** | **0.938** |

Additional counts:

```text
Reviewed responses: 16/16
Partial answers: 0
Needs-review responses: 0

Unsafe substantive answers on expected-withhold cases: 0
Unnecessary abstentions on answerable controls: 0

Structural false accepts: 1
Safe semantic abstentions among structural false accepts: 1
```

All eight answerable controls received substantive answers.

All eight expected-withhold cases semantically abstained.

The structurally false-accepted `ie_ne_ne_004` response was therefore a:

```text
safe semantic abstention
```

rather than an unsafe answer.

## Structural Guard vs User-Visible Behavior

The current evidence guard is intentionally structural.

A generated answer containing a valid citation can pass the guard even when the model-written answer is itself an abstention.

Therefore:

```text
structural accepted
```

does not necessarily mean:

```text
substantive answer delivered
```

The current 16-case human review found:

```text
Structural agreement with response behavior = 0.938
Actual response-behavior accuracy           = 1.000
```

This distinction is now explicitly measured.

No production evidence-guard change was made based on the single structural mismatch because the application-visible behavior was safe.

Adding brittle multilingual string matching merely to make the structural state mirror model-written abstention text would risk metric-driven complexity without demonstrated user benefit.

## Language-Pair Meaning

In benchmark notation:

```text
query_language -> target_language
```

the second language is the **retrieval corpus language**, not the requested output language.

Production intentionally uses:

```python
answer_language = record.query_language
```

and:

```python
filters = {
    "language": record.target_language
}
```

Therefore:

```text
EN -> NE
```

means:

```text
English query
retrieval from Nepali-language evidence
English answer
```

and:

```text
NE -> EN
```

means:

```text
Nepali query
retrieval from English-language evidence
Nepali answer
```

This interpretation was explicitly corrected in the response-review notes in:

```text
ab91220 — Correct response review language notes
```

---

# Milestone 4 — Consolidated Production-Quality Checkpoint

Committed and pushed as:

```text
99e10df — Add production quality checkpoint
```

Implementation:

```text
src/evaluation/production_quality_checkpoint.py
```

Tests:

```text
tests/test_production_quality_checkpoint.py
```

Output:

```text
data/evaluation/production_quality_checkpoint_v1.json
```

Checkpoint configuration:

```text
production-quality-checkpoint-v1
```

Schema version:

```text
1
```

The checkpoint deterministically consolidates:

1. official production RAG structural benchmark
2. human semantic citation review
3. automated semantic-judge agreement
4. structural insufficient-evidence benchmark
5. human response-level abstention review

No:

* hosted generation
* retrieval
* reranking
* embedding
* Gemini production answer generation

is executed by the checkpoint builder.

It recomputes metrics from persisted artifacts using the current evaluation code.

## Consolidated Headline Results

### Official production RAG benchmark

```text
Selected primary hit:       0.833
Selected relevant recall:   0.833
Cited primary hit:          0.833
Cited relevant precision:   0.482
Cited relevant recall:      0.800
Valid citation references:  1.000
```

### Human semantic citation review

```text
Fully supported factual claims:       0.950
At least partially supported claims:  1.000
Unsupported factual claims:           0.000
Required citation coverage:           1.000
```

Individual citation support:

```text
Fully supported:              0.917
At least partially supported: 0.979
Unsupported:                  0.021
```

Counts:

```text
Individual citations reviewed:   48
Fully supported:                 44
Partially supported:              3
Unsupported:                      1
Needs review:                     0
```

Important terminology correction:

The `IndSup` column printed by:

```text
python -m src.evaluation.semantic_review summary
```

uses:

```text
individual_at_least_partial_support_rate
```

Therefore the previously reported:

```text
IndSup = 0.979
```

means:

```text
97.9% of reviewed individual citations had at least partial support
```

It does **not** mean that 97.9% were fully supported.

The actual individual full-support rate is:

```text
0.917
```

### Automated semantic judge

```text
Semantic exact agreement:       0.979
Citation-requirement agreement: 1.000
Individual-evidence agreement:  0.958
```

### Structural insufficient-evidence evaluation

```text
Decision accuracy:               0.938
Answer acceptance:               1.000
Withholding success:             0.875
Structural false-accept rate:    0.125
Structural false-withhold rate:  0.000
```

### Human response-level behavior

```text
Behavior accuracy:               1.000
Answer delivery:                 1.000
Semantic abstention success:     1.000
Unsafe substantive-answer rate: 0.000
Structural behavior agreement:   0.938
```

---

# Why the 30-Question Hosted Production Benchmark Was Not Rerun

No production component changed during this four-milestone evaluation cycle.

The unchanged production path includes:

* retrieval
* BM25/RRF routing
* candidate depth
* reranking
* context selection
* grounded prompt
* Gemini generation model
* Gemini transport
* citation processing
* evidence guard
* RAG orchestration

Therefore a fresh 30-question hosted Gemini run would have been another stochastic sample of the same production configuration.

It would not measure an architectural or policy intervention.

The checkpoint explicitly records:

```text
hosted_production_rerun_performed = false
official_production_benchmark_retained = true
```

The official production benchmark remains:

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

A new hosted benchmark should be run when a production component affecting answers is deliberately changed and needs comparison against this baseline.

---

# Previous Human Semantic Evaluation Checkpoint

The preceding four-milestone cycle established claim-level semantic evaluation.

Completed milestones were:

1. claim-level citation alignment
2. semantic dataset with exact evidence
3. stratified human semantic-review subset
4. human semantic citation evaluation

Key implementation:

```text
src/evaluation/claim_citation_evaluator.py
src/evaluation/build_semantic_evaluation_dataset.py
src/evaluation/build_semantic_review_subset.py
src/evaluation/semantic_review.py
```

Primary outputs:

```text
data/evaluation/semantic/production_rag_v2_claims.jsonl
data/evaluation/semantic/production_rag_v2_human_review_v1.jsonl
data/evaluation/semantic/production_rag_v2_human_review_v1_labeled.jsonl
```

Semantic dataset:

```text
Questions:              30
Claims:                290
Cited claims:          252
Citation assignments: 305
```

Human review subset:

```text
48 claims
12 per language pair
```

Strata per language pair:

```text
2 multi-citation
3 single-citation numeric
5 single-citation non-numeric
1 uncited numeric
1 uncited non-numeric
```

Sampling is deterministic.

The subset is deliberately stratified.

It is not a simple random or population-proportional sample of all 290 claims.

Therefore human semantic rates are diagnostic rates for the reviewed sample rather than unbiased population estimates.

---

# Human Semantic Review Results

| Slice | N | Done | Full | Any Support | Unsupported | Required Citation Coverage | Individual Any Support |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN -> EN | 12 | 1.000 | 0.900 | 1.000 | 0.000 | 1.000 | 1.000 |
| NE -> NE | 12 | 1.000 | 0.900 | 1.000 | 0.000 | 1.000 | 1.000 |
| EN -> NE | 12 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.917 |
| NE -> EN | 12 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| **Overall** | **48** | **1.000** | **0.950** | **1.000** | **0.000** | **1.000** | **0.979** |

Additional counts:

```text
Reviewed claims: 48/48

Supported factual claims:            38
Partially supported factual claims:   2
Unsupported factual claims:           0
Not-a-factual-claim labels:            8
Semantic needs-review claims:          0

Citation requirement unclear:          0
Unnecessary citation rate:         0.000
```

Individual evidence:

```text
Supported:           44
Partially supported:  3
Unsupported:          1
Needs review:         0

Full support rate:              0.917
At least partial support rate:  0.979
Unsupported rate:               0.021
```

---

# Semantic Failure Modes Identified

Human review identified several failure modes that structural citation validity cannot detect.

## Numerical mismatch

A generated claim reported:

```text
33.8%
```

for institutional-school enrollment.

The cited evidence contained:

```text
33.6%
```

The claim was labeled:

```text
partially_supported
```

This demonstrates the need for explicit numerical fidelity evaluation.

---

## Truncated evidence

A compound claim included both:

* an out-of-school indicator
* a reading-proficiency statement

The cited passage established the first but was truncated while beginning the second.

The result was:

```text
partially_supported
```

This demonstrates that relevant retrieval does not guarantee that a selected chunk contains the complete evidence needed for a compound answer.

---

## Joint support across multiple citations

A health-institution claim combined:

* a primary-health-centre count
* a statement that centres were being upgraded to hospitals

Different passages established different portions.

The joint claim was:

```text
supported
```

while individual citations could be only:

```text
partially_supported
```

This is why joint claim support and individual citation support remain separate metrics.

---

## Individual over-citation

One vaccination claim was established by one cited passage.

A second cited passage contained related vaccination statistics but did not establish the exact generated proposition.

Result:

```text
Joint claim: supported
One citation: supported
One citation: unsupported
```

This is a citation-efficiency issue rather than an unsupported-answer issue.

---

## OCR-corrupted evidence

One Nepali evidence passage rendered a male percentage incorrectly due to OCR corruption while another cited passage contained the correct values.

The answer remained jointly supported, but the corrupted evidence assignment was only:

```text
partially_supported
```

OCR quality therefore remains an evidence-quality concern even when answer-level semantics remain correct.

---

# Combined Citation Parser Correction

The original claim-level citation parser recognized:

```text
[E1]
```

and:

```text
[E1], [E2]
```

but originally failed to expand combined groups such as:

```text
[E1, E2]
```

The parser was corrected to support:

```text
[E1]
[E1], [E2]
[E1, E2]
[E1, E2, E3]
```

The corrected parser:

* expands all evidence IDs in a group
* preserves first-appearance order
* deduplicates repeated evidence IDs
* removes the complete citation expression when producing cleaned claim text

Full benchmark inspection found seven generated claims using combined citation syntax.

After correction:

```text
Claims:
290 -> 290

Cited claims:
245 -> 252

Citation assignments:
291 -> 305
```

The correction changed citation structure, not claim segmentation.

Relevant commit:

```text
984710e — Fix multilingual claim extraction artifacts
```

---

# Official Production RAG Benchmark

Official persisted benchmark:

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

Run configuration:

```text
production-rag-v2-interactions
```

Questions:

```text
30/30
```

Language slices:

```text
EN -> EN: 12
NE -> NE:  6
EN -> NE:  6
NE -> EN:  6
```

Final deterministic structural results:

| Slice | N | Accept | SelHit | SelRec | CitHit | CitPrec | CitRec | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EN -> EN | 12 | 1.000 | 0.833 | 0.792 | 0.833 | 0.533 | 0.792 | 1.000 |
| NE -> NE | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.458 | 0.917 | 1.000 |
| EN -> NE | 6 | 1.000 | 0.833 | 0.861 | 0.833 | 0.347 | 0.778 | 1.000 |
| NE -> EN | 6 | 1.000 | 0.667 | 0.722 | 0.667 | 0.539 | 0.722 | 1.000 |
| **Overall** | **30** | **1.000** | **0.833** | **0.833** | **0.833** | **0.482** | **0.800** | **1.000** |

Metric meanings:

```text
Accept
    accepted-answer rate

SelHit
    selected context contains at least one primary annotated passage

SelRec
    recall of annotated relevant passages in selected context

CitHit
    cited evidence contains at least one primary annotated passage

CitPrec
    fraction of cited passage identities appearing in the manually
    annotated retrieval-relevance set

CitRec
    recall of annotated relevant passages among cited evidence

Valid
    fraction of generated evidence identifiers that resolve to selected
    evidence
```

These are structural and retrieval-identity metrics.

They must not be interpreted as direct semantic factuality metrics.

---

# Structural vs Semantic Citation Quality

Production structural benchmark:

```text
CitPrec = 0.482
```

This does **not** mean:

```text
48.2% factual accuracy
```

or:

```text
48.2% semantically correct citations
```

It means that cited passage identities overlapped the manually annotated retrieval-relevance set at that macro-averaged rate.

Human semantic review found:

```text
Fully supported factual claims = 0.950
At least partial support        = 1.000
Unsupported                     = 0.000
```

Individual citations:

```text
Full support                    = 0.917
At least partial support        = 0.979
Unsupported                     = 0.021
```

Therefore retrieval-gold overlap and semantic support remain separate evaluation dimensions.

Possible reasons a cited passage falls outside the annotated relevant set include:

* legitimate supplementary evidence
* overlapping chunks
* incomplete retrieval annotations
* multiple passages supporting the same proposition
* over-citation
* genuinely weak citation selection

Human semantic evaluation is the appropriate layer for distinguishing those cases.

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

```text
2,276
```

The Nepali Economic Survey requires forced OCR.

Do not rerun OCR unless corpus changes require it.

---

# Chunking

Implementation:

```text
src/chunking/chunk_documents.py
```

Tokenizer:

```text
intfloat/multilingual-e5-large-instruct
```

Parameters:

```text
target chunk size: approximately 400 tokens
overlap: approximately 60 tokens
```

Original passage text is preserved.

Current chunking remains a flat overlapping baseline.

Parent/child structure-aware chunking has not been promoted to production.

---

# Embeddings and Qdrant

Embedding model:

```text
intfloat/multilingual-e5-large-instruct
```

Dimension:

```text
1024
```

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

```text
nepal_gov_documents
```

Stored representations:

```text
dense
dense_contextual
bm25
```

Production semantic retrieval uses:

```text
dense_contextual
```

Raw:

```text
dense
```

vectors remain stored for baseline comparison and diagnostics.

Original:

```text
chunk_text
```

remains canonical evidence.

All:

```text
2,276
```

points have contextual vectors.

Do not rerun contextual-vector backfill unless required by a corpus or representation change.

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

## Cross-language retrieval

English -> Nepali:

```text
dense_contextual only
```

Nepali -> English:

```text
dense_contextual only
```

BM25 is intentionally skipped when query language and target evidence language differ.

Production candidate depth:

```text
20
```

---

# Retrieval Evaluation

Benchmark:

```text
data/evaluation/retrieval_questions.jsonl
```

Questions:

```text
30
```

First-stage baseline:

| Cutoff | Hit | MRR | Recall |
|---|---:|---:|---:|
| @5 | 0.767 | 0.505 | 0.686 |
| @10 | 0.900 | 0.524 | 0.856 |
| @20 | 0.900 | 0.524 | 0.886 |

Persistent top-20 primary-evidence misses:

```text
en_en_011
en_ne_002
ne_en_005
```

Evidence that never enters the top-20 candidate pool cannot be recovered by reranking or generation.

NE -> EN remains the weakest production language direction in the existing structural benchmark.

---

# Production Reranking

Implementation:

```text
src/reranking/hf_bge_reranker.py
```

Model:

```text
BAAI/bge-reranker-v2-m3
```

Hosted through Hugging Face TEI.

Production representation:

```text
Document: <title>

<original chunk_text>
```

Candidate count:

```text
20
```

Title-aware reranking benchmark:

| Cutoff | Hit | MRR | Recall |
|---|---:|---:|---:|
| @5 | 0.833 | 0.697 | 0.833 |
| @10 | 0.900 | 0.706 | 0.869 |
| @20 | 0.900 | 0.706 | 0.886 |

---

# Production Context Selection

Production strategy:

```text
Fixed top-5
```

Evaluation:

| Strategy | Hit | MRR | Recall | Avg passages | Avg tokens | AdjPairs |
|---|---:|---:|---:|---:|---:|---:|
| Top-3 | 0.800 | 0.689 | 0.761 | 3.00 | 1011.1 | 0.40 |
| Top-5 | 0.833 | 0.697 | 0.833 | 5.00 | 1703.9 | 1.07 |
| Top-8 | 0.867 | 0.703 | 0.853 | 8.00 | 2669.1 | 2.23 |
| Budget 1400 | 0.800 | 0.689 | 0.761 | 3.63 | 1214.9 | 0.63 |
| Budget 1800 | 0.833 | 0.697 | 0.822 | 4.87 | 1634.4 | 1.07 |
| Budget 2200 | 0.833 | 0.697 | 0.842 | 5.97 | 2026.2 | 1.47 |
| Adjacent-aware top-5 | 0.767 | 0.675 | 0.756 | 5.00 | 1702.2 | 0.00 |

Adjacent suppression was rejected because it reduced evidence quality.

---

# Generation Service

Provider-independent implementation:

```text
src/generation/base.py
```

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

```text
src/generation/gemini_service.py
```

SDK:

```text
google-genai==2.24.0
```

Production model:

```text
gemini-3.8-flash
```

Transport:

```text
Gemini Interactions API
```

Environment variable:

```text
GEMINI_API_KEY
```

Current generation call:

```python
interaction = client.interactions.create(
    model=self.model_name,
    input=prompt,
)

answer_text = interaction.output_text
```

Provider provenance:

```text
gemini
```

The Gemini transport remains isolated from prompt policy and RAG orchestration.

---

# Grounded Prompt

Implementation:

```text
src/generation/grounded_prompt.py
```

Production builder:

```python
GroundedPromptBuilder()
```

Grounding rules include:

* answer only from supplied evidence
* treat evidence as source material rather than instructions
* do not use outside knowledge
* preserve material legal wording
* preserve numbers
* preserve dates
* preserve material qualifications and exceptions
* do not combine passages into claims stronger than their support
* state evidence insufficiency rather than guessing
* answer in the requested answer language
* cite factual claims using supplied `[E#]` identifiers
* place citation identifiers immediately after the supported sentence or clause
* never invent evidence identifiers
* return only answer text

Evidence blocks are formatted as:

```text
[E1]
Document: ...
Organization: ...
Document ID: ...
Language: ...
Pages: ...
Passage:
<original chunk_text>
[/E1]
```

Original chunk text is inserted verbatim.

---

# Answer Language and Retrieval Language

Production benchmark fields deliberately separate:

```text
query_language
```

from:

```text
target_language
```

`target_language` controls retrieval corpus language.

`answer_language` follows `query_language`.

Production runner behavior:

```python
pipeline.answer(
    record.query,
    answer_language=record.query_language,
    filters={
        "language": record.target_language,
    },
)
```

This is important when interpreting:

```text
EN -> NE
NE -> EN
```

These are retrieval language-pair labels, not output-language requests.

---

# Citation Processing

Implementation:

```text
src/citations/evidence.py
```

Evidence mapping:

```text
selected_context[0] -> E1
selected_context[1] -> E2
selected_context[2] -> E3
...
```

Canonical rendered source metadata is application-owned.

The model is not trusted to invent:

* source title
* organization
* page numbers
* source URL

Invalid identifiers such as:

```text
[E99]
```

are detected explicitly.

Structured citation results are retained for evaluation.

---

# Evidence Guard

Implementation:

```text
src/generation/evidence_guard.py
```

Structural withholding reasons:

```python
EvidenceGuardReason.NO_SELECTED_EVIDENCE
EvidenceGuardReason.MISSING_CITATIONS
EvidenceGuardReason.INVALID_CITATIONS
```

Current structural policy:

```text
No selected evidence
    -> skip generation and withhold

Generated answer with no valid citations
    -> withhold

Generated answer with any invalid citation IDs
    -> withhold

Selected evidence + structurally valid citations
    -> structurally accept
```

No arbitrary retrieval or reranking score threshold is used.

The evidence guard intentionally does **not** perform semantic entailment classification.

The response-level abstention milestone established that:

```text
structurally accepted
```

and:

```text
semantically substantive
```

are not equivalent concepts.

---

# RAG Orchestration

Implementation:

```text
src/rag/pipeline.py
```

Production constructor:

```python
build_production_rag_pipeline()
```

Current result structure includes:

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

Structured:

```text
citation_result
```

is deliberately preserved so evaluation can inspect evidence references without reparsing rendered source output.

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
Same-language                    Cross-language
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

# Current Evaluation Architecture

```text
                    Persisted Production RAG Output
                               |
           +-------------------+-------------------+
           |                                       |
           v                                       v
 Structural RAG Evaluation                 Claim Extraction
                                                   |
                                                   v
                                      Claim/Citation Alignment
                                                   |
                                                   v
                                     Exact Evidence Materialization
                                                   |
                                                   v
                                        Semantic Claim Dataset
                                                   |
                                                   v
                                     Stratified Human Reference
                                                   |
                                  +----------------+---------------+
                                  |                                |
                                  v                                v
                         Human Semantic Review            Automated Judge
                                  |                                |
                                  +---------------+----------------+
                                                  |
                                                  v
                                       Human/Judge Agreement


        Dedicated Insufficient-Evidence Questions
                         |
                         v
                Production RAG Pipeline
                         |
                         v
              Structural Guard Evaluation
                         |
                         v
              Persisted Application Output
                         |
                         v
              Human Response-Level Review


All persisted evaluation layers
             |
             v
Production Quality Checkpoint
```

---

# Consolidated Evaluation Interpretation

The current evaluation layers answer different questions.

## Retrieval evaluation

Question:

> Did the retrieval system surface manually annotated relevant evidence?

Primary metrics:

```text
Hit
MRR
Recall
```

---

## Production structural RAG evaluation

Question:

> Did selected and cited passage identities overlap the annotated relevant passages, and were citation references structurally valid?

Primary metrics:

```text
SelHit
SelRec
CitHit
CitPrec
CitRec
Valid
```

---

## Human semantic citation review

Question:

> Does the actual cited evidence semantically establish the generated claim?

Primary metrics:

```text
fully supported
at least partially supported
unsupported
required citation coverage
individual evidence support
```

---

## Automated semantic judge evaluation

Question:

> How closely does the automated evaluator reproduce human semantic labels?

Primary metrics:

```text
semantic exact agreement
citation-requirement agreement
individual-evidence agreement
```

---

## Structural insufficient-evidence evaluation

Question:

> Did the structural evidence guard produce the expected accepted/withheld state?

Primary metrics:

```text
decision accuracy
withholding success
false accept
false withhold
```

---

## Human response-level evaluation

Question:

> What did the user-visible answer actually do?

Primary labels:

```text
substantive_answer
abstained
partial_answer_with_limitation
needs_review
```

This layer is required because structural guard state alone does not completely characterize generated semantic behavior.

---

# Current Measured Weaknesses

## 1. NE -> EN retrieval

In the official production structural benchmark:

```text
NE -> EN SelHit = 0.667
NE -> EN SelRec = 0.722
```

This remains the weakest language direction.

Candidate retrieval is the primary concern because missing top-20 evidence cannot be recovered by reranking or generation.

---

## 2. Automated Judge Class Coverage

The automated semantic judge has strong agreement on the current human set:

```text
Semantic = 0.979
Requirement = 1.000
Individual = 0.958
```

but the reference set contains no human examples of:

```text
unsupported
needs_review
unclear citation requirement
```

Judge reliability on those classes remains unvalidated.

---

## 3. Partial/Mixed Insufficient-Evidence Cases

The 16-question insufficient-evidence dataset currently lacks:

```text
partial_evidence
mixed_supported_unsupported
```

These cases are important because a system may need to:

* answer only the supported portion
* explicitly qualify unsupported portions
* avoid presenting related evidence as complete evidence

---

## 4. Structural Guard / Semantic Abstention Mismatch

One benchmark response was structurally accepted but semantically abstained.

This is not currently an unsafe-answer failure.

It is an observability distinction.

Changing the production guard purely to eliminate that metric mismatch is not justified by current evidence.

---

## 5. Numerical Fidelity

Human review observed a specific numerical mismatch.

A dedicated numerical fidelity benchmark has not yet been implemented.

---

## 6. OCR Evidence Quality

The forced-OCR Nepali Economic Survey contains occasional OCR corruption.

This can reduce individual citation quality even when another citation jointly supports the answer.

---

## 7. Citation Efficiency

Some answers cite passages that are related but do not individually establish the associated claim.

This creates over-citation or weak individual citation assignments even when the overall answer remains supported.

---

## 8. Answer Completeness

The current semantic review measures claim support.

It does not yet systematically measure whether the generated answer omitted important evidence needed for a complete response.

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

## Candidate depth

```text
20
```

## Reranker

```text
BAAI/bge-reranker-v2-m3
```

Input:

```text
Document: <title>

<original chunk_text>
```

## Context selection

```text
Fixed top-5
```

## Generation provider

```text
GeminiGenerationService
```

## Gemini model

```text
gemini-3.8-flash
```

## Gemini transport

```text
Interactions API
```

## Prompt

```text
GroundedPromptBuilder
```

## Citation identifiers

```text
E1, E2, E3, ...
```

Claim-level parsing additionally supports combined syntax:

```text
[E1, E2]
```

## Evidence guard

Structural citation/evidence validation without arbitrary retrieval-score thresholds.

## Application orchestration

```text
RAGPipeline
```

## Production benchmark

```text
data/evaluation/rag_runs/production_rag_v2_interactions.jsonl
```

## Semantic evaluation dataset

```text
data/evaluation/semantic/production_rag_v2_claims.jsonl
```

## Human semantic reference

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_labeled.jsonl
```

## Automated semantic judge output

```text
data/evaluation/semantic/production_rag_v2_human_review_v1_judge.jsonl
```

## Insufficient-evidence benchmark

```text
data/evaluation/insufficient_evidence_questions.jsonl
```

## Human response behavior review

```text
data/evaluation/rag_runs/insufficient_evidence_v1_response_review.jsonl
```

## Consolidated quality checkpoint

```text
data/evaluation/production_quality_checkpoint_v1.json
```

---

# Architecture Principles

The project currently enforces the following design principles:

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
* selected evidence order remains stable
* provider SDK objects do not leak downstream
* source provenance remains available end to end
* structured citation results remain machine-readable
* semantic evaluation remains separate from production generation
* human semantic labels remain distinct from automated predictions
* human labels remain the semantic judge reference standard
* retrieval gold overlap is not treated as semantic entailment
* structural guard state is not treated as equivalent to semantic response behavior
* experiments are not promoted without evaluation
* expensive preprocessing is not rerun without need
* resumable hosted evaluation avoids repeating completed calls
* prompt hashes protect automated judge predictions from stale prompt reuse
* persisted evaluation artifacts are preferred over unnecessary stochastic reruns
* production changes should be driven by measured failures rather than metric cosmetics

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

Production RAG evaluation runner and subsequent validation:

```text
325 passed
```

Claim citation evaluation and semantic dataset work:

```text
359 passed
```

Stratified semantic review subset:

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

After insufficient-evidence benchmark:

```text
445 passed
```

After human response-level abstention evaluation:

```text
458 passed
```

After consolidated production-quality checkpoint:

```text
467 passed
```

Current validated full suite:

```text
467 passed in 3.93s
```

---

# Git Milestone History

## Latest evaluation cycle

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

Review metadata correction:

```text
ab91220 — Correct response review language notes
```

Consolidated production-quality checkpoint:

```text
99e10df — Add production quality checkpoint
```

## Previous semantic-evaluation checkpoint

Status checkpoint:

```text
2de6b86 — Update semantic evaluation checkpoint
```

Human semantic evaluation:

```text
750966f — Add human semantic citation evaluation
```

Stratified human subset:

```text
47cab25 — Add stratified semantic review subset
```

Multilingual claim/parser cleanup:

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

## Production benchmark cycle

```text
1bc404b — Record production RAG benchmark
e9ab5ec — Migrate Gemini generation to Interactions API
9d14f05 — Add resumable RAG evaluation runner
80058cf — Add deterministic RAG evaluation metrics
```

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

# Recommended Next Four-Milestone Cycle

The next cycle should continue to prioritize measured coverage gaps rather than speculative architecture changes.

Recommended sequence:

1. **Expand insufficient-evidence benchmark with partial and mixed-evidence cases**
2. **Expand semantic human reference with hard negative / ambiguous classes**
3. **Benchmark targeted NE -> EN retrieval improvements**
4. **Add answer-completeness and factual-fidelity evaluation, then checkpoint**

The exact sequence may change if the first experiments identify a more important failure mode.

---

# Proposed Milestone 1 — Partial and Mixed Evidence Benchmark

Extend the insufficient-evidence dataset with:

```text
partial_evidence
mixed_supported_unsupported
```

Examples should require the model to distinguish:

```text
fully answerable
partially answerable
not answerable
```

Evaluation should preserve separate labels for:

* substantive answer
* semantic abstention
* partial answer with explicit limitation
* unsafe unsupported completion

This will exercise the currently unused:

```text
partial_answer_with_limitation
```

review path.

The dataset should remain balanced across language pairs where practical.

---

# Proposed Milestone 2 — Hard Semantic Judge Reference Cases

The existing automated judge reference set lacks human:

```text
unsupported
needs_review
unclear
```

examples.

Build a small targeted human-reviewed challenge set containing:

* deliberately unsupported claim/evidence pairs
* partially supported numeric/date cases
* OCR-corrupted ambiguous evidence
* citation-requirement ambiguity
* conflicting or incomplete evidence
* semantically related but non-supporting passages

Use this set to measure class-specific judge behavior before using the automated evaluator more broadly.

Human labels must remain the ground truth.

---

# Proposed Milestone 3 — NE -> EN Retrieval Improvement

Current official production result:

```text
NE -> EN selected primary hit = 0.667
NE -> EN selected relevant recall = 0.722
```

Potential experiments may include:

* multilingual query reformulation
* translated retrieval query as an additional signal
* alternative contextual query instruction
* candidate-depth experiments
* fusion of original and translated semantic queries
* model comparison only if justified
* language-aware retrieval strategies

Experiments must remain separate from production.

Promotion requires improvement on the retrieval benchmark without materially degrading the other language slices.

A production rerun should occur only after an actual retrieval change is promoted.

---

# Proposed Milestone 4 — Completeness and Fidelity Evaluation

Current semantic review answers:

> Are generated claims supported?

It does not fully answer:

> Did the answer include the important supported information that should have been present?

Add targeted evaluation for:

* answer completeness
* numerical fidelity
* date fidelity
* legal qualification preservation
* exception preservation
* material omission
* unnecessary citation use

This milestone should end with another consolidated checkpoint and an updated status document.

---

# Remaining Major Work

Evaluation and quality:

1. partial-evidence benchmark
2. mixed supported/unsupported benchmark
3. hard-negative automated-judge validation
4. answer completeness evaluation
5. larger semantic faithfulness evaluation
6. numerical fidelity evaluation
7. date fidelity evaluation
8. legal qualification and exception preservation
9. citation-efficiency evaluation
10. OCR evidence-quality diagnostics

Retrieval:

11. NE -> EN retrieval improvements
12. targeted investigation of persistent top-20 misses
13. query reformulation experiments
14. possible cross-lingual query translation/fusion
15. candidate-depth experiments where justified

Application:

16. FastAPI application
17. Streamlit interface
18. structured logging
19. user-facing source rendering refinement

Experiment and operations:

20. MLflow experiment tracking
21. Docker/Compose application integration
22. CI refinement
23. environment/configuration refinement

Documentation and portfolio:

24. README architecture documentation
25. benchmark documentation
26. screenshots/demo
27. portfolio presentation

Potential experimental work, only if evaluation justifies it:

* generation model comparison
* context-size comparison
* citation-aware generation prompting
* reranker-score evidence sufficiency experiments
* parent/child chunking
* automatic semantic validation
* advanced prompt-injection defenses

Experiments must remain separate from production until measured.

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

This clone is stored under OneDrive.

Git automatic housekeeping previously produced deletion/retry problems under:

```text
.git/objects
```

Repository integrity was checked successfully with:

```text
git fsck --full
```

Local automatic Git GC was disabled for this clone:

```text
git config --local gc.auto 0
```

Keep this workaround unless the repository is moved to a fresh clone outside OneDrive.

Do not manually delete:

```text
.git/objects/*
```

Do not rerun:

```text
git gc
```

as routine project work in this OneDrive clone.

If a Git lock appears after an interrupted command, first verify that no Git process is active before removing only the specific stale lock file.

---

# Secrets

Never commit, print, or expose:

```text
HF_TOKEN
HF_RERANKER_ENDPOINT_URL
GEMINI_API_KEY
```

Hosted credentials remain local environment configuration.

Evaluation artifacts must never contain secrets.

---

# Expensive Operations

Do not rerun unless required:

* forced OCR
* document ingestion
* dense embeddings
* contextual dense-vector backfill
* full reranker benchmark
* completed hosted production benchmark generation
* completed semantic-judge calls

The current corpus and stored vectors remain valid for the present architecture.

The semantic evaluation dataset can be rebuilt from:

* persisted production RAG output
* current Qdrant evidence

without rerunning the production Gemini benchmark.

The consolidated production-quality checkpoint is also deterministic over persisted evaluation artifacts.

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

Before committing:

```text
git diff --cached --check
```

Inspect staged state:

```text
git status --short
git diff --cached --stat
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

Avoid:

```text
git add .
```

when unrelated working-tree files might exist.

---

# Project Status Update Cadence

Update:

```text
docs/project_status.md
```

after every **four completed milestones**.

Do not update it after every small implementation change.

An earlier update is appropriate only for:

* a major architectural reset
* an explicit handoff requirement
* a correction necessary to prevent future work from using invalid project state

When this file is updated, replace the complete document rather than maintaining partial fragments across conversations.

---

# Current Checkpoint

The completed four-milestone checkpoint is:

**Semantic Judge Validation, Abstention Safety, and Consolidated Production Quality**

Completed milestones:

```text
1. Automated semantic judge validation
2. Insufficient-evidence benchmark
3. Human response-level abstention evaluation
4. Consolidated production-quality checkpoint
```

Primary milestone commits:

```text
7a62a77 — Add automated semantic judge evaluation
c3d60b7 — Add insufficient evidence benchmark
d622873 — Add response abstention evaluation
99e10df — Add production quality checkpoint
```

Supporting correction:

```text
ab91220 — Correct response review language notes
```

Current quality headline:

```text
Official production structural benchmark:
    Selected primary hit              = 0.833
    Selected relevant recall          = 0.833
    Valid citation references         = 1.000

Human semantic review:
    Fully supported                   = 0.950
    At least partially supported      = 1.000
    Unsupported                       = 0.000
    Required citation coverage        = 1.000
    Individual full support           = 0.917
    Individual at least partial       = 0.979

Automated semantic judge:
    Semantic exact agreement          = 0.979
    Citation requirement agreement    = 1.000
    Individual evidence agreement     = 0.958

Insufficient-evidence structural:
    Decision accuracy                 = 0.938
    Withholding success               = 0.875
    Structural false accept           = 0.125
    Structural false withhold         = 0.000

Human response behavior:
    Behavior accuracy                 = 1.000
    Answer delivery                   = 1.000
    Semantic abstention success       = 1.000
    Unsafe substantive answer rate    = 0.000
    Structural/behavior agreement     = 0.938
```

Current validated test suite:

```text
467 passed
```

Current pushed repository head before this status-document update:

```text
99e10df — Add production quality checkpoint
```

The next development phase is:

**Hard-case evaluation expansion and measured cross-lingual retrieval improvement.**