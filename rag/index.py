"""TF-IDF vector index + cosine similarity retrieval (no external API needed)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .corpus import get_corpus


class VectorIndex:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_ids = []
        self.matrix = None
        self._built = False

    def build(self, docs):
        texts = [d["text"] for d in docs]
        self.doc_ids = [d["id"] for d in docs]
        self.matrix = self.vectorizer.fit_transform(texts)  # (n_docs, vocab)
        self._built = True

    def query(self, text, k=3):
        """Return the top-k doc ids ranked by cosine similarity to the query."""
        if not self._built:
            raise RuntimeError("Index not built yet.")
        qvec = self.vectorizer.transform([text])
        sims = cosine_similarity(qvec, self.matrix)[0]  # (n_docs,)
        ranked = sorted(zip(self.doc_ids, sims), key=lambda t: t[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:k]]

    def scores(self, text, k=3):
        qvec = self.vectorizer.transform([text])
        sims = cosine_similarity(qvec, self.matrix)[0]
        ranked = sorted(zip(self.doc_ids, sims), key=lambda t: t[1], reverse=True)
        return ranked[:k]
