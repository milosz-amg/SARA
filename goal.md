# SARA: Search and Research Agent

## Overview
**SARA (Search and Research Agent)** is an **LLM-based recommendation system** designed to support **scientific collaborations** and **grant applications**.  
It integrates intelligent search, semantic analysis, and recommendation pipelines to help researchers discover **collaborators**, **projects**, and **funding opportunities** more effectively than with traditional academic search tools.

---

## Motivation
- Finding collaborators and suitable grant calls is **time-consuming, biased, and inefficient**.  
- Early-career researchers are especially disadvantaged by reliance on informal networks.  
- SARA aims to increase **equity, efficiency, and transparency** in academic collaboration.

---

## Core Approach
- **Large Language Models (LLMs)** + **Retrieval-Augmented Generation (RAG)**.  
- **Vector databases** for semantic similarity search.  
- Construction of a **knowledge graph** enriched with embeddings, linking:
  - Publications  
  - Researchers and institutions  
  - Research projects  
  - Grant calls  

This allows **semantic matching** that goes beyond keyword search, capturing thematic and contextual relevance.

---

## Features
- **Natural Language Queries**: e.g.  
  - *“I want to apply for a project on AI and sustainability—who should I work with?”*  
  - *“Which calls in the next 6 months best match my expertise in deep learning?”*  
- **Tailored Recommendations**: collaborators, calls, funding opportunities.  
- **Cross-domain Matching**: supports interdisciplinary discovery.  
- **Context-aware Ranking**: considers publication history, grant success likelihood, institutional fit.  
- **Validation Agents**: reduce hallucination and ensure factual alignment with databases.  
- **Explainability**: highlights why a researcher or call was suggested.  

---

## System Architecture
SARA consists of interacting agents, each responsible for different stages of the pipeline:

1. **Agent 1 (NLU)**: interprets the user query, decides which databases to search.  
2. **Agent 2 (Search)**: retrieves candidate results from structured/unstructured sources.  
3. **Agent 3 (Validation)**: checks results against ground-truth databases to limit hallucinations.  
4. **ElasticSearch / Vector DB**: semantic search and embedding-based retrieval.  
5. **Agent 4 (Display)**: decides how to present results (tables, graphs, lists).  
6. **Agent 5 (Final Validation)**: ensures clarity, readability, and alignment with the user’s intent.  

---

## Expected Impact
- **Faster, fairer discovery** of collaborators and grants.  
- **Improved interdisciplinarity** by surfacing less obvious but semantically relevant matches.  
- **Transparency & equity** in research opportunities.  
- **Prototype results** show high-quality, explainable, and user-friendly recommendations.  

---

## Conference Poster Summary
SARA represents a **scalable, generalizable framework** for enhancing academic collaboration and funding discovery.  
The poster presents:
- Full pipeline (data ingestion → vectorization → semantic search → recommendation).  
- Architecture (LLM + RAG + embeddings + validation agents).  
- Evaluation results on real academic/grant datasets.  
- Ethical considerations: bias mitigation, privacy, hallucination control.  

**SARA = intelligent advisor for the future of academic collaboration and funding.**
