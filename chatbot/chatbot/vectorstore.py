from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


VECTORSTORE_PATH = Path("storage/vectorstore")


class VectorStoreManager:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def get_or_create(self, documents):
        if VECTORSTORE_PATH.exists():
            try:
                return self.load_vectorstore()
            except Exception as e:
                print("⚠️ Failed to load existing vectorstore. Rebuilding...")
                print(e)
                return self.create_vectorstore(documents)
        else:
            return self.create_vectorstore(documents)

    def create_vectorstore(self, documents):
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding,
        )
        VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        return vectorstore

    def load_vectorstore(self):
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True,
        )