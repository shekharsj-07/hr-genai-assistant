HR GenAI Assistant – Acme RAG Application

Overview
The Acme RAG application is an HR Policy Assistant built using a Retrieval-Augmented Generation (RAG) architecture.
Employees can ask natural-language questions about internal HR policies (leave, benefits, working hours, code of conduct, etc.), and the system generates grounded, explainable answers strictly from the provided policy documents.

The application is designed to be:
- Local-first (no paid APIs required)
- Scalable and modular
- Production-oriented
- Auditable, with evaluation and usage tracking



## PYTHON 3.10.x recommended ##

To install via homebrew:

install homebrew if not installed: 

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

```

Then, install python 3.10.x

```bash

brew install python@3.10
brew link --overwrite python@3.10

```

Architecture Overview

Core Components
- Document Ingestion: Supports ingestion of one or multiple HR policy documents.
- Chunking: Paragraph-based text chunking using LangChain text splitters.
- Embeddings: Local sentence embeddings using Sentence Transformers.
- Vector Store: FAISS for fast semantic retrieval.
- Large Language Model (LLM):
  * Primary (Recommended): Ollama with Mistral
  * Fallback: Local HuggingFace model (pure Python, no system dependency)
- RAG Pipeline: Retriever + context-grounded answer generation.
- Web UI: Chainlit-based chat interface.
- Persistence: SQLite database for chat history.
- Evaluation & Observability: ROUGE-L and BLEU metrics logged using MLflow.
- Insights Layer: FAQ insights derived from historical user questions.


Project Structure

hr-genai-assistant/
   app/
      app.py
   chatbot/
      loader.py
      chunking.py
      embeddings.py
      vectorstore.py
      rag_chain.py
      hf_client.py
      ollama_client.py
      history.py
      faq_insights.py
      evaluation.py
      ollama_utils.py
      llm_factory.py
      data/hr_policies/
│        acme_hr_policy.txt
      storage/
│        history.db
      requirements.txt
      README.md


## How to Run

1. Environment Setup

macOS / Linux:
``` bash 
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Windows (PowerShell):
``` shell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Note: If python or pip does not work, use python3 or pip3.

2. Local LLM Setup – Ollama (Optional but Recommended)

Important:
Ollama is optional. If Ollama is not installed or running, the application automatically falls back to a local HuggingFace model. No API keys are required.

macOS / Linux:

1. Download Ollama from https://ollama.com/download
2. Install and launch Ollama (macOS users must open Ollama.app once)
3. Verify installation:
   ollama --version

Windows:
1. Download the Windows installer from https://ollama.com/download
2. Run the installer and complete setup
3. Restart PowerShell
4. Verify installation:
   ollama --version

Download and run Mistral (one-time):
```bash
ollama pull mistral
ollama run mistral "Hello, are you running?"
```
Note:
Ollama runs as a background service. Do not run ollama serve if the service is already running.

3. Run the Application
```bash
chainlit run app/app.py
```

The application will be available at:
http://localhost:8000

4. MLflow Tracking (Optional)

In a separate terminal:
``` bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1
mlflow ui


Open:
http://127.0.0.1:5000

```


## FAQ Insights

On application startup, the system displays a list of the most frequently asked HR questions.
These insights are derived by analyzing historical user queries stored in the SQLite database and grouping semantically similar questions.

This feature:
- Improves discoverability
- Provides usage insights
- Does not affect the core RAG pipeline


## Evaluation & Observability

- ROUGE-L F1: Measures overlap between generated answers and retrieved context.
- BLEU: Measures lexical precision (reported for completeness).
- MLflow: Logs metrics, parameters, and artifacts per user query.

Metrics are logged automatically for each interaction.


## Key Design Decisions

- No paid or proprietary APIs
- Explicit grounding to prevent hallucinations
- Modular, extensible architecture
- Local-first execution for sensitive HR data


## Future Enhancements

- Semantic similarity evaluation metrics
- Admin analytics dashboard
- Role-based access control


## Author

Shekhar S. Jana
GitHub: https://github.com/shekharsj-07

