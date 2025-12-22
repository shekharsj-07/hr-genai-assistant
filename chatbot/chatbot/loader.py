from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from chatbot.config import config


class MultiDocumentLoader:
    """
    Loads all .txt documents from the HR policies directory.
    """

    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = data_dir

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

    def load_documents(self) -> List[Document]:
        documents: List[Document] = []

        for file_path in self.data_dir.iterdir():
            if file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file_path.name

                documents.extend(docs)

        if not documents:
            raise ValueError("No documents loaded.")

        return documents