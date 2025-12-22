from collections import defaultdict
from typing import Dict, List

from sentence_transformers import SentenceTransformer, util


class FAQInsights:
    def __init__(self, questions: List[str], similarity_threshold: float = 0.85):
        self.questions = questions
        self.threshold = similarity_threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def top_faqs(self, top_k: int = 5) -> Dict[str, int]:
        if not self.questions:
            return {}

        embeddings = self.model.encode(self.questions, convert_to_tensor=True)
        visited = set()
        clusters = defaultdict(int)

        for i, q in enumerate(self.questions):
            if i in visited:
                continue

            clusters[q] += 1
            visited.add(i)

            for j in range(i + 1, len(self.questions)):
                if j in visited:
                    continue

                score = util.cos_sim(embeddings[i], embeddings[j]).item()
                if score >= self.threshold:
                    clusters[q] += 1
                    visited.add(j)

        return dict(
            sorted(clusters.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )