from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from chatbot.config import config
from chatbot.llm_factory import get_llm


class HRPolicyRAG:
    """
    Stable, local RAG pipeline using Ollama (LangChain >=0.3 compatible).
    """

    def __init__(self, vectorstore: FAISS):
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )

        self.llm = get_llm()

    def _build_context(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self, question: str) -> Dict[str, Any]:
        # ✅ CORRECT way in new LangChain
        docs = self.retriever.invoke(question)

        context = self._build_context(docs)

        prompt = f"""
        You are an HR policy assistant.
        Answer the question strictly using the context below.
        If the answer is not present, say "Not specified in policy".

        Context:
        {context}

        Question:
        {question}
        """

        answer = self.llm.invoke(prompt).content

        return {
            "question": question,
            "answer": answer,
            "sources": docs,
        }