# HR GenAI Assistant – RAG-based Policy Chatbot

## Overview
This project implements an **HR Policy Assistant** using a **Retrieval-Augmented Generation (RAG)** architecture.  
Employees can ask natural language questions about HR policies (leave, benefits, conduct, etc.), and the system provides **grounded, explainable answers** sourced from internal policy documents ONLY.

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
    app/
        app.py                # Chainlit application
    chatbot/
        loader.py             # Multi-document ingestion
        chunking.py           # Text chunking logic
        embeddings.py         # Embedding backend
        vectorstore.py        # FAISS vector store
        rag_chain.py          # RAG pipeline
        history.py            # SQLite chat history
        evaluation.py         # ROUGE/BLEU + MLflow
        ollama_utils.py       # Ollama auto-bootstrap
        llm_factory.py        # LLM backend selection
    mlruns                    #for logging Q/A session
        1                     #keeps every Q/A asked
    data/hr_policies/
        acme_hr_policy.txt    # Policy doc
    storage/
        vectorstore
            index.faiss       #faiss vector store index
            index.pkl         #metadata for faiss document mapping
        history.db            #stores chat history (SQLite db)
requirements.txt
README.md


---

## How to Run

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Local LLM Ollama Setup (Optional – Recommended)##
Important: Ollama is optional.
If Ollama is not installed, the application automatically falls back to a pure Python HuggingFace-based local model, so the app will still run successfully.


macOS / Linux
	1.	Download Ollama from the official website:
👉 https://ollama.com/download
	2.	Install and launch the application
	•	macOS users must open Ollama.app once (menu bar icon should appear)
	3.	Verify installation

```bash
ollama --version

```

Windows
	1.	Download the Windows installer from:
👉 https://ollama.com/download
	2.	Run the installer and complete setup
	3.	Restart your terminal (PowerShell / Command Prompt)
	4.	Verify installation

```powershell
ollama --version
```

```bash / powershell

ollama pull mistral
ollama serve
ollama run mistral "Hello, are you running?"

```



### 2. Run the app
```bash
chainlit run app/app.py

The app will be available at:
http://localhost:8000


```

### 3. Run the below command in a separate terminal to initiate mlflow tracking
```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1
```
*use ctrl + c to quit*

then use 

```bash
mlflow ui
```


### 4. LLM Backend Strategy

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

### Future Enhancements ###
	•	Semantic similarity metrics
	•	Faithfulness / grounding score
	•	Admin dashboard for analytics
	•	Role-based access control

⸻


Author

Shekhar S. Jana
Contact: 9087486777
GitHub: https://github.com/shekharsj-07