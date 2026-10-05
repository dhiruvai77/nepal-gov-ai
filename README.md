# NepalGov AI

NepalGov AI is a production-oriented multilingual RAG application for asking questions about official Government of Nepal documents. It retrieves relevant evidence, generates an answer in English or Nepali, and displays the supporting sources so the answer can be checked.

**Live demo:** https://nepal-gov-ai-production.up.railway.app

## What it does

- Answers questions over Nepal Government documents.
- Supports English and Nepali queries and answers.
- Supports same-language and cross-language retrieval.
- Combines contextual dense retrieval with BM25 for same-language search.
- Uses multilingual dense retrieval for cross-language search.
- Reranks candidates with `BAAI/bge-reranker-v2-m3`.
- Returns structured evidence with document, page, organization, and source URL.
- Withholds answers when the selected evidence or required citations are insufficient.
- Provides a browser interface and JSON API.
- Runs locally with Docker and in production on Railway with Qdrant Cloud.

## Architecture

```text
User question
    ↓
FastAPI API / web interface
    ↓
Language-aware retrieval
    ├─ contextual E5 dense retrieval
    ├─ BM25 sparse retrieval for same-language queries
    └─ Qdrant reciprocal-rank fusion
    ↓
BGE multilingual reranking
    ↓
Top-five context selection
    ↓
Gemini Interactions API
    ↓
Citation and evidence guard
    ↓
Answer with structured sources or safe refusal
```

## Current corpus

The production collection contains six official documents and 2,276 indexed chunks in the Qdrant collection `nepal_gov_documents`.

- Constitution of Nepal
- Public Health Service Act 2075
- Compulsory and Free Education Act 2075
- Economic Survey 2023/24
- Economic Survey 2081/82
- Budget Speech 2025/26

The indexed Constitution document is in English. Nepali questions can retrieve that evidence through cross-language retrieval and receive answers in Nepali.

## Technology

- Python 3.12
- FastAPI and Uvicorn
- Qdrant Cloud
- `intfloat/multilingual-e5-large-instruct`
- `BAAI/bge-reranker-v2-m3`
- BM25 sparse retrieval
- Gemini Interactions API with `gemini-3.8-flash`
- Docker and Docker Compose
- Railway deployment
- Pytest

## API

Health check:

```text
GET /health
```

Readiness check:

```text
GET /health/ready
```

Answer endpoint:

```text
POST /v1/answer
```

Example request:

```bash
curl -X POST https://nepal-gov-ai-production.up.railway.app/v1/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What rights relating to education are guaranteed by Article 31 of the Constitution of Nepal?",
    "answer_language": "en",
    "filters": {
      "document_id": "constitution_nepal_current_en"
    }
  }'
```

The response includes the answer, acceptance status, withholding reason when applicable, provider/model metadata, selected context count, and structured source records.

## Local development

Create and activate a virtual environment, then install the development requirements:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Set the required environment variables without committing their values:

```text
GEMINI_API_KEY
HF_TOKEN
HF_RERANKER_ENDPOINT_URL
QDRANT_URL
QDRANT_API_KEY
```

Run the API locally:

```bash
uvicorn src.api.app:app --reload
```

The local Docker stack can be started with:

```bash
docker compose up --build
```

## Testing

The complete automated test suite currently passes:

```text
727 passed, 1 warning
```

Run it with:

```bash
python -m pytest -q
```

## Evaluation

The repository includes a 30-question multilingual RAG benchmark covering:

- English query to English evidence
- Nepali query to Nepali evidence
- English query to Nepali evidence
- Nepali query to English evidence

It also includes retrieval evaluation, insufficient-evidence evaluation, human citation review, semantic evaluation, and production quality checkpoints under `data/evaluation/`.

The current baseline recorded 100% structural answer acceptance and 100% validity across the 30-question benchmark. Human review found 95% fully supported claims, 100% at least partially supported claims, and no major factual answer errors. The main measured quality gap is answer completeness.

## Deployment

The application is deployed from the `main` branch to Railway using the Dockerfile. Qdrant Cloud stores the production collection and Railway provides the public HTTPS service.

The validated production baseline is tagged:

```text
v1.0-production-baseline
```

The baseline commit passed the full test suite and public English, Nepali, unfiltered, and insufficient-evidence smoke tests.

## Project structure

```text
src/              Application, retrieval, embeddings, reranking, RAG, indexing, and evaluation code
data/             Corpus metadata and evaluation datasets
tests/             Unit, integration, API, readiness, and deployment tests
docs/             Project status and production baseline documentation
Dockerfile        Production runtime image
docker-compose.yml Local application and Qdrant stack
```

## Safety and limitations

- Answers are limited to the indexed government-document corpus.
- The system can refuse an answer when the evidence does not support it.
- Hosted Hugging Face providers require valid credentials and an active endpoint.
- End-to-end hosted response latency is approximately 55–59 seconds in observed production tests.
- Production secrets are supplied through environment configuration and are never committed to the repository.

## License and source documents

This project is a personal engineering and learning project. Source documents are official Government of Nepal publications, and their original source URLs are returned with evidence records where available.
