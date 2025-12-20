from collections import Counter
from typing import Dict, List

from langchain_core.documents import Document


class PolicyInsights:
    def __init__(self, documents: List[Document], chunks: List[Document]):
        self.documents = documents
        self.chunks = chunks

    def document_stats(self) -> Dict[str, int]:
        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "total_words": sum(len(d.page_content.split()) for d in self.documents),
        }

    def frequent_terms(self, top_k: int = 10) -> Dict[str, int]:
        words = []
        for chunk in self.chunks:
            for w in chunk.page_content.lower().split():
                if w.isalpha() and len(w) > 4:
                    words.append(w)
        return dict(Counter(words).most_common(top_k))