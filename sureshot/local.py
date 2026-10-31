"""System 1 as a small classifier trained on your graded decisions, running in-process: no server, no prompt."""
from sureshot.adapters.base import normalize

DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"


def sentence_embedder(name=DEFAULT_EMBEDDING):
    """A local embedding model as a function from a list of texts to a 2D array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


class LocalDecider:
    """Embeds the state with a frozen embedding model, then one logistic regression per question (trained here) gives
    the probability of each answer. It answers only the questions it was trained on (leaving the rest out of its result,
    so in a Router chain they pass to the next tier), and not images.

    LocalDecider.train(store.labeled()) trains from a log's graded decisions; `rows` need "state", "key" and "truth"."""

    def __init__(self, heads, embed=None, embedding=DEFAULT_EMBEDDING):
        self.heads, self.embedding, self._embed = heads, embedding, embed

    @property
    def identity(self):
        """Changes whenever the heads or the embedding model do, so retraining never reuses cached answers."""
        import joblib

        return f"local|{self.embedding}|{joblib.hash(self.heads)}"

    @property
    def embed(self):
        if self._embed is None:
            self._embed = sentence_embedder(self.embedding)
        return self._embed

    @classmethod
    def train(cls, rows, embed=None, embedding=DEFAULT_EMBEDDING, C=10.0):
        from sklearn.linear_model import LogisticRegression

        decider = cls({}, embed, embedding)
        by_key = {}
        for row in rows:
            by_key.setdefault(row["key"], []).append(row)
        for key, group in by_key.items():
            answers = [row["truth"] for row in group]
            if len(set(answers)) < 2:
                raise ValueError(f"question {key!r} needs at least two different answers to train on")
            decider.heads[key] = LogisticRegression(C=C, max_iter=1000).fit(decider.embed([row["state"] for row in group]), answers)
        return decider

    def decide(self, state, questions, images=()):
        if images:
            raise ValueError("LocalDecider can't take images")
        known = {key: q for key, q in questions.items() if key in self.heads}
        if not known:
            return {}
        vector = self.embed([state])
        result = {}
        for key, question in known.items():
            head = self.heads[key]
            probs = dict(zip(head.classes_.tolist(), head.predict_proba(vector)[0].tolist()))
            if question["type"] == "noul":
                raw = {"noul": probs.get(True, 0.0)}
            else:
                raw = {"probabilities": {str(answer): p for answer, p in probs.items()}}
            result[key] = normalize(question, raw)
        return result

    def save(self, path):
        import joblib

        joblib.dump({"heads": self.heads, "embedding": self.embedding}, path)

    @classmethod
    def load(cls, path, embed=None):
        import joblib

        saved = joblib.load(path)
        return cls(saved["heads"], embed, saved["embedding"])
