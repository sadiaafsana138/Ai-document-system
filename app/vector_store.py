import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, vectors, texts):
        self.index.add(np.array(vectors).astype("float32"))
        self.texts.extend(texts)

    def search(self, query_vector, k=5):
        k = min(k, len(self.texts))
        D, I = self.index.search(np.array([query_vector]).astype("float32"), k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

    def search_with_scores(self, query_vector, k=5):
        k = min(k, len(self.texts))
        D, I = self.index.search(np.array([query_vector]).astype("float32"), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.texts):
                results.append({"text": self.texts[idx], "score": float(dist)})
        return results