from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from chatbot.embeddings import get_embeddings
from chatbot.config import config


class VectorStoreManager:
    """
    this block manages FAISS vector store lifecycle.
    """

    def __init__(self, persist_dir: Path = config.VECTORSTORE_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = get_embeddings()

    def build_vectorstore(self, chunks: List[Document]) -> FAISS:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        vectorstore.save_local(str(self.persist_dir))
        return vectorstore

    def load_vectorstore(self) -> FAISS:
        return FAISS.load_local(
            str(self.persist_dir),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

    def get_or_create(self, chunks: List[Document]) -> FAISS:
        if any(self.persist_dir.iterdir()):
            return self.load_vectorstore()
        return self.build_vectorstore(chunks)