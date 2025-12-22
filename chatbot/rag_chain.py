from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from chatbot.config import config
from chatbot.llm_factory import generate_response


class HRPolicyRAG:
    """
    Retrieval-Augmented Generation (RAG) chain
    for answering HR policy questions.
    """

    def __init__(self, vectorstore: FAISS):
        # Retriever setup
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )

    def _build_context(self, docs: List[Document]) -> str:
        """
        Combine retrieved documents into a single context string.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Generate an answer for the given question using RAG.
        """
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)

        # Build context
        context = self._build_context(docs)

        # Prompt template
        prompt = f"""
You are an HR policy assistant for Acme Corporation.

Rules:
- If the user says "hello", "hi", or greets you, respond ONLY with:
  "Hello! How can I help you today? You can ask me questions about Acme's leave policies and employee benefits programs."
- Answer the question strictly using the context below.
- If the answer is NOT present in the context, respond with:
  "Sorry, I don't have an answer to this as it is not specified in the policy."

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate response (Ollama or HuggingFace fallback)
        answer, backend_used = generate_response(prompt)

        return {
            "question": question,
            "answer": answer.strip(),
            "sources": docs,
            "backend": backend_used,
        }