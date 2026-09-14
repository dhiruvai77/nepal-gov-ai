# Retrieval Evaluation Dataset

This directory contains manually curated evaluation queries for the NepalGov AI
retrieval system.

## Purpose

The dataset is used to evaluate retrieval quality before and after changes such as:

- embedding-model changes
- hybrid retrieval
- cross-lingual routing
- reranking
- chunking changes

Retrieval quality should be measured before generation quality so that failures in
document retrieval can be distinguished from failures in answer generation.

## Evaluation slices

The initial dataset targets 30 questions:

| Slice | Target |
|---|---:|
| English → English | 12 |
| Nepali → Nepali | 6 |
| English → Nepali | 6 |
| Nepali → English | 6 |
| Total | 30 |

## JSONL schema

Each line in `retrieval_questions.jsonl` represents one evaluation query.

Required fields:

- `question_id`: Stable unique identifier.
- `query`: User-style search question.
- `query_language`: Language of the query (`en` or `ne`).
- `target_language`: Language of documents expected to contain the evidence.
- `category`: Corpus domain such as constitution_law, finance_economy,
  health_population, or education.
- `expected_document_ids`: Documents expected to contain relevant evidence.
- `primary_relevant_chunk_ids`: Strongest manually verified chunks that directly answer
  the query.
- `relevant_chunk_ids`: All manually verified chunks that directly answer or materially
  support the query, including the primary relevant chunks.
- `notes`: Short annotation explaining the relevance judgment.

## Relevance policy

Primary relevance should be assigned conservatively. A primary relevant chunk should
contain enough evidence to substantially answer the query on its own.

The broader relevant set may contain supporting chunks that provide additional
directly useful evidence.

Every chunk listed in `primary_relevant_chunk_ids` must also appear in
`relevant_chunk_ids`.

A chunk should be marked relevant only when it contains evidence that directly
answers or materially supports the query.

Do not mark a chunk relevant merely because it contains the same keywords.

More than one chunk may be relevant for a question.

Relevance judgments should be based on the actual indexed corpus, not on assumptions
about what a document ought to contain.

## Cross-lingual evaluation

Cross-lingual queries are evaluated separately from same-language retrieval.

For V1:

- English → English: hybrid retrieval
- Nepali → Nepali: hybrid retrieval
- English → Nepali: dense retrieval
- Nepali → English: dense retrieval

This matches the current production retrieval-routing policy.