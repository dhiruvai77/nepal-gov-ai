# NepalGov AI — V1 Production Baseline Report

**Baseline tag:** `v1.0-production-baseline`  
**Commit:** `24b148c` — Improve hosted provider error diagnostics  
**Validation date:** 2026-10-05  
**Public application:** https://nepal-gov-ai-production.up.railway.app

## 1. Baseline scope

This report records the validated V1 production baseline of NepalGov AI, an evidence-grounded multilingual retrieval-augmented generation system for Nepal Government documents.

The baseline includes document ingestion, multilingual embeddings, hybrid retrieval, cross-language dense retrieval, hosted reranking, context selection, Gemini generation, citation validation, FastAPI endpoints, a static web interface, Qdrant Cloud, and Railway deployment.

The baseline is frozen by the Git tag `v1.0-production-baseline`. Future experiments should be compared against this tag and should not silently change the production configuration.

## 2. Indexed corpus

The production Qdrant collection is `nepal_gov_documents`.

- Documents: 6
- Indexed chunks: 2,276
- Qdrant points: 2,276
- Dense vector dimension: 1,024
- Named vectors: `dense`, `dense_contextual`, and `bm25`
- Production semantic vector: `dense_contextual`
- Payload indexes: `language`, `category`, `organization`, `document_id`, and `document_type`

Indexed document identifiers:

```text
constitution_nepal_current_en
public_health_service_act_2075_en
compulsory_free_education_act_2075_en
economic_survey_2023_24_en
economic_survey_2081_82_ne
budget_speech_2025_26_en
```

The Constitution of Nepal is indexed in English. Nepali questions can retrieve the English evidence through cross-language dense retrieval and receive answers in Nepali.

## 3. Production architecture

- Embeddings: `intfloat/multilingual-e5-large-instruct`
- Same-language retrieval: contextual dense retrieval plus BM25 with reciprocal-rank fusion
- Cross-language retrieval: contextual dense retrieval
- Candidate depth: 20
- Reranker: `BAAI/bge-reranker-v2-m3`
- Final context: top 5 passages
- Generator: Gemini Interactions API using `gemini-3.8-flash`
- API: FastAPI
- Vector database: Qdrant Cloud
- Deployment: Railway with Docker

The evidence guard rejects responses when selected context or required citations are missing, invalid, or inconsistent with the selected evidence.

## 4. Offline evaluation baseline

The repository contains the official 30-question multilingual RAG benchmark:

| Evaluation slice | Questions |
|---|---:|
| English query → English evidence | 12 |
| Nepali query → Nepali evidence | 6 |
| English query → Nepali evidence | 6 |
| Nepali query → English evidence | 6 |
| **Total** | **30** |

Recorded structural results:

| Slice | Accept | Selection hit | Selection recall | Citation hit | Citation precision | Citation recall | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN → EN | 1.000 | 0.833 | 0.792 | 0.833 | 0.533 | 0.792 | 1.000 |
| NE → NE | 1.000 | 1.000 | 1.000 | 1.000 | 0.458 | 0.917 | 1.000 |
| EN → NE | 1.000 | 0.833 | 0.861 | 0.833 | 0.347 | 0.778 | 1.000 |
| NE → EN | 1.000 | 0.667 | 0.722 | 0.667 | 0.539 | 0.722 | 1.000 |
| **Overall** | **1.000** | **0.833** | **0.833** | **0.833** | **0.482** | **0.800** | **1.000** |

These are structural evidence-identity metrics. They do not replace claim-level human assessment.

Recorded human semantic citation review covered 48 claims:

- Fully supported: 95.0%
- At least partially supported: 100%
- Unsupported: 0%
- Required citation coverage: 100%

Recorded human answer-quality review covered 30 answers:

- Fully complete: 70.0%
- At least mostly complete: 80.0%
- Fully faithful: 93.3%
- No major factual error: 100%
- Strong answer: 66.7%
- Acceptable answer: 80.0%

The primary measured quality gap is answer completeness, not major factual hallucination.

## 5. Automated and safety evaluation

The recorded insufficient-evidence evaluations show:

- Original structural decision accuracy: 93.8%
- Original withholding success: 87.5%
- Partial/mixed extension structural accuracy: 100%
- Partial/mixed extension human behavior accuracy: 100%
- Unsafe substantive response rate in the reviewed response behavior evaluation: 0%

Human labels remain authoritative where automated semantic judgments are uncertain.

## 6. Local and cloud validation

The complete local test suite passed:

```text
727 passed, 1 warning
```

The warning was an unrelated Starlette/AnyIO deprecation warning.

The public Railway deployment passed readiness validation:

- `GET /health`: HTTP 200
- `GET /health/ready`: HTTP 200
- Pipeline initialized: true
- Gemini configured: true
- Hugging Face token configured: true
- Reranker endpoint configured: true
- Qdrant reachable: true
- Qdrant collection ready: true

Successful public RAG validations included:

- English Article 31 question with the Constitution filter: accepted, cited, two sources, five selected passages.
- Nepali Article 31 question with the English Constitution filter: accepted, Nepali answer, two sources, five selected passages.
- Unfiltered English right-to-information question: accepted with Constitution sources.
- Unrelated Stockholm weather question: safely withheld because the government evidence was insufficient.

The deployed provider diagnostic hardening records bounded provider status and messages without logging query or passage contents or credential values.

## 7. Known limitations

1. The indexed Constitution document is English-only. Nepali answers currently rely on cross-language retrieval from English evidence.
2. End-to-end hosted answer latency is approximately 55–59 seconds in observed production tests.
3. Citation precision and answer completeness remain the main areas for future improvement.
4. The public smoke tests validate representative behavior; they are not a substitute for a full repeated production benchmark.
5. Hosted Hugging Face providers depend on valid tokens and active endpoints.

## 8. Reproducibility rule

Any future retrieval, reranking, prompt, embedding, corpus, or deployment change must be evaluated against this frozen baseline and recorded under a new commit or tag. The tag `v1.0-production-baseline` remains the comparison point for V1 results.
