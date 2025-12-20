from pydantic import BaseModel
from pathlib import Path


class AppConfig(BaseModel):
    # Paths
    DATA_DIR: Path = Path("data/hr_policies")
    VECTORSTORE_DIR: Path = Path("storage/vectorstore")

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Retrieval
    TOP_K: int = 4

    # Models
    LLM_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Tracking
    MLFLOW_EXPERIMENT_NAME: str = "hr_rag_assistant"


# Singleton config object
config = AppConfig()