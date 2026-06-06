<div align="center">

# A-Law AI Server

> AI-powered lease contract OCR, legal clause analysis, and risk detection server

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-1C3C3C)](https://www.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?logo=pinecone&logoColor=white)](https://www.pinecone.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://openai.com)
[![Celery](https://img.shields.io/badge/Celery-5.3.6-37814A)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-5.0+-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-AMQP-FF6600?logo=rabbitmq&logoColor=white)](https://www.rabbitmq.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com)

</div>

---

**A-Law AI Server** is the AI backend for the A-Law lease contract analysis platform.  
It performs OCR processing, contract clause analysis, legal risk detection, RAG-based legal verification, and asynchronous AI analysis through RabbitMQ, Celery, and Redis.

---

## Features

### Contract OCR

Receives contract images from either AWS S3 or direct file upload and extracts structured OCR text.  
The OCR result can include word-level coordinate overlays for displaying recognized text on the original contract image.

### AI Contract Analysis

Analyzes extracted contract text and performs:

- Clause segmentation
- Risk analysis
- Illegal clause detection
- Missing clause detection
- Fraud pattern detection
- Contract summary generation
- Recommendation generation

### Asynchronous Analysis Pipeline

Spring Boot can submit OCR text to the AI server after OCR completion.  
The AI server creates a Celery task, stores job status in Redis, and processes the analysis asynchronously through RabbitMQ.

### RAG-based Legal Verification

The server is designed to use Pinecone as a vector database for legal document retrieval.  
RAG endpoints are currently scaffolded and prepared for integration with the internal RAG module.

### Spring Boot Integration

The AI server is designed to work as a downstream AI processing service for the A-Law backend.  
Spring Boot handles user-facing APIs, authentication, storage, and contract lifecycle management, while this server handles OCR and AI analysis.

---

## Architecture

```text
┌─────────────────────────────────────────┐
│        Frontend / Spring Boot API       │
│ Contract Upload · OCR Request · Status  │
└────────────────────┬────────────────────┘
                     │ REST / RabbitMQ
                     ▼
┌─────────────────────────────────────────┐
│          A-Law AI Server                 │
│        FastAPI / Python 3.11             │
│ OCR · Analysis · RAG · Risk Detection    │
└──────────────┬──────────────────────────┘
               │
               ├── OpenAI / LangChain / LangGraph
               ├── Pinecone Vector DB
               ├── Redis Result Store
               ├── RabbitMQ Message Queue
               ├── Celery Worker
               └── AWS S3 Contract Images
