from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


from chatbot.config import config
from chatbot.llm_factory import get_llm


class HRPolicyRAG:
    """
    RAG chain for answering HR policy questions.
    """

    def __init__(self, vectorstore: FAISS):
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )

        self.llm = get_llm()

    def _build_context(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self, question: str) -> Dict[str, Any]:
        docs = self.retriever.invoke(question)

        context = self._build_context(docs)

        prompt = f"""
        You are an HR policy assistant. If someone says "Hello/hi", you respond just with "Hello! How can I help you today? You can ask me questions about Acme's leave policies and employee benfits programs."
        Answer the question strictly using the context below.
        If the answer is not present, say "Sorry, I don't have an answer to this as it is not specified in the policy".

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