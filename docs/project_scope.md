# NepalGov AI — Project Scope

## 1. Project Overview

NepalGov AI is a multilingual, evidence-grounded knowledge platform for public Government of Nepal documents.

The system will allow users to search official documents and ask natural-language questions about Nepal's constitution, laws, public policies, government programs, ministries, and selected areas of public administration.

Answers must be grounded in retrieved official documents and accompanied by source citations.

The project is intended to demonstrate production-oriented Retrieval-Augmented Generation (RAG), semantic search, multilingual NLP, information retrieval, evaluation, API development, testing, and MLOps practices.

---

## 2. Business Problem

Government information is distributed across many ministries, commissions, departments, reports, acts, policies, and public websites.

Users may need to search multiple documents manually to answer questions such as:

- What rights are guaranteed by the Constitution of Nepal?
- What does a particular law say about a specific issue?
- What are the objectives of a government policy?
- What does the national budget allocate to a particular sector?
- What policies govern health or education?

Traditional keyword search can be insufficient when the user's wording differs from the terminology used in the source document.

Documents may also differ in:

- language;
- structure;
- terminology;
- publication format;
- length;
- metadata quality.

This creates a need for a system capable of retrieving relevant information semantically while preserving the source evidence required to verify an answer.

---

## 3. Proposed Solution

Develop a production-oriented Retrieval-Augmented Generation platform that:

1. acquires publicly available Government of Nepal documents;
2. extracts and cleans document text;
3. preserves document and page-level metadata;
4. indexes documents using keyword and vector-based retrieval;
5. retrieves relevant evidence for natural-language questions;
6. reranks candidate passages;
7. generates answers using retrieved evidence;
8. provides source citations;
9. evaluates retrieval and generation quality quantitatively.

The system should distinguish between supported and unsupported questions and avoid generating confident answers when sufficient evidence cannot be retrieved.

---

## 4. Target Users

Potential users include:

- citizens seeking information from government documents;
- students and researchers;
- journalists;
- policy analysts;
- legal and public-administration researchers;
- government employees;
- organizations working with Nepalese public policy.

The portfolio implementation is a research and demonstration system and is not intended to replace official legal or governmental advice.

---

## 5. Version 1 Knowledge Domains

Version 1 will focus on four domains.

### 5.1 Constitution and Law

Examples:

- Constitution of Nepal;
- selected Acts;
- selected regulations;
- selected legal and administrative documents.

Primary source candidates include the Nepal Law Commission and other official Government of Nepal sources.

### 5.2 Finance and Economy

Examples:

- national budget speeches;
- economic surveys;
- fiscal reports;
- economic policies;
- selected Ministry of Finance publications.

### 5.3 Health and Population

Examples:

- national health policies;
- public-health Acts and regulations;
- health-sector strategies;
- population policies;
- official guidelines.

### 5.4 Education

Examples:

- education policies;
- strategic plans;
- government education programs;
- selected education Acts and regulations;
- official ministry reports.

---

## 6. Data Scope

The initial target is approximately 40–60 official public documents.

Documents may include both English and Nepali material.

Each source document should have metadata such as:

- document ID;
- title;
- issuing organization;
- document category;
- language;
- publication date;
- source URL;
- download URL;
- retrieval date;
- local filename.

Large raw documents will not be committed directly to the Git repository where this is impractical.

Instead, the project should maintain a reproducible source manifest and acquisition pipeline.

---

## 7. Multilingual Scope

NepalGov AI should eventually support both Nepali and English.

Initial retrieval scenarios include:

- English query → English document;
- Nepali query → Nepali document.

Advanced experiments may include:

- English query → Nepali document;
- Nepali query → English document.

This will allow the project to evaluate multilingual embedding and retrieval models.

---

## 8. Core Use Cases

### UC1 — Government Document Search

A user enters a natural-language query and receives relevant passages from official documents.

### UC2 — Evidence-Grounded Question Answering

A user asks a question and receives an answer generated from retrieved official evidence.

### UC3 — Source Verification

The user can inspect:

- source document;
- page number;
- retrieved passage;
- retrieval or reranking score where appropriate.

### UC4 — Cross-Document Questions

The system can retrieve evidence from several documents when a question requires information from more than one source.

### UC5 — Multilingual Retrieval

Users can search government documents using Nepali or English queries.

### UC6 — Retrieval Comparison

The system can compare multiple retrieval strategies such as:

- BM25 keyword retrieval;
- dense semantic retrieval;
- hybrid retrieval;
- reranked retrieval.

---

## 9. Evaluation Goals

The system should be evaluated rather than judged only through example conversations.

A manually curated evaluation dataset will contain questions with known relevant documents or passages.

Potential retrieval metrics include:

- Precision@K;
- Recall@K;
- Mean Reciprocal Rank (MRR);
- Hit Rate.

Generation evaluation may include:

- answer relevance;
- context relevance;
- faithfulness;
- citation correctness;
- unsupported-answer rate.

Performance measurements may also include:

- retrieval latency;
- end-to-end response latency;
- indexing time.

---

## 10. Non-Functional Requirements

The system should be:

### Reproducible
Document acquisition and processing should be repeatable.

### Traceable
Generated answers should be linked to their supporting evidence.

### Modular
Ingestion, preprocessing, retrieval, reranking, generation, evaluation, API, and UI components should remain logically separated.

### Configurable
Important parameters such as chunk size, retrieval method, top-k, embedding model, and reranking configuration should not be hard-coded.

### Testable
Core functionality should have automated tests.

### Observable
Important system events, errors, latency, and retrieval behavior should be logged.

### Maintainable
The codebase should follow a clear modular structure and documented configuration.

---

## 11. Version 1 Boundaries

Version 1 will not attempt to cover every Government of Nepal organization or every public document.

The first release will focus only on:

- constitution and selected laws;
- finance and economy;
- health and population;
- education.

The system will not initially provide:

- live government-news monitoring;
- every provincial or local-government document;
- personalized legal advice;
- authoritative legal interpretation;
- transactional government services;
- real-time government databases.

These may be considered future extensions.

---

## 12. Project Objective

The objective of NepalGov AI is to design, implement, and evaluate a production-oriented multilingual RAG system capable of retrieving information from heterogeneous Government of Nepal documents and generating transparent, evidence-grounded answers with verifiable source attribution.