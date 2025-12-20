# HR GenAI Assistant – RAG-based Policy Chatbot

## Overview
This project implements an **HR Policy Assistant** using a **Retrieval-Augmented Generation (RAG)** architecture.  
Employees can ask natural language questions about HR policies (leave, benefits, conduct, etc.), and the system provides **grounded, explainable answers** sourced from internal policy documents.

The solution is designed to be **scalable, local-first, and production-oriented**, with built-in evaluation and observability.

---

## Architecture

**Core components:**
- **Document ingestion**: Multi-document loader (supports adding more policies)
- **Chunking**: Paragraph-based chunking using LangChain text splitters
- **Embeddings**: Sentence Transformers (local, open-source)
- **Vector Store**: FAISS
- **LLM**:
  - Primary: Local LLM via **Ollama (Mistral)**
  - Fallback: Pure Python HuggingFace model (no system dependency)
- **RAG Pipeline**: Retriever + context-grounded generation
- **Web UI**: Chainlit
- **Persistence**: SQLite for chat history
- **Evaluation**: ROUGE-L & BLEU logged via MLflow

---

## Project Structure