from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chatbot.config import config


class DocumentChunker:
    """
    Splits documents into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No chunks created from documents.")

        return chunks