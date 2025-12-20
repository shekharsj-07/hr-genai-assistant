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
hr-genai-assistant/
├── app/
│   └── app.py                # Chainlit application
├── chatbot/
│   ├── loader.py             # Multi-document ingestion
│   ├── chunking.py           # Text chunking logic
│   ├── embeddings.py         # Embedding backend
│   ├── vectorstore.py        # FAISS vector store
│   ├── rag_chain.py          # RAG pipeline
│   ├── history.py            # SQLite chat history
│   ├── evaluation.py         # ROUGE/BLEU + MLflow
│   ├── ollama_utils.py       # Ollama auto-bootstrap
│   └── llm_factory.py        # LLM backend selection
├── data/hr_policies/
│   └── acme_hr_policy.txt
├── storage/
│   └── history.db
├── requirements.txt
└── README.md


---

## How to Run

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt



### 2. Run the App
chainlit run app/app.py

The app will be available at:
http://localhost:8000



LLM Backend Strategy

The application automatically selects the best available local LLM:
	•	Primary: Ollama (Mistral) – high quality local inference
	•	Fallback: HuggingFace local model – pure Python, no system install

This ensures the app runs out-of-the-box in most environments.

⸻

Evaluation & Observability
	•	ROUGE-L F1: Measures overlap between generated answer and retrieved context
	•	BLEU: Measures lexical precision (reported for completeness)
	•	MLflow: Tracks metrics, parameters, and artifacts per user query

Metrics are logged automatically for each question.

⸻

Key Design Decisions
	•	No paid or proprietary APIs required
	•	Modular, extensible architecture
	•	Explicit grounding to reduce hallucinations
	•	Transparent evaluation metrics

⸻

Future Enhancements
	•	Semantic similarity metrics
	•	Faithfulness / grounding score
	•	Admin dashboard for analytics
	•	Role-based access control

⸻

Author

Shekhar
GitHub: https://github.com/shekharsj-07