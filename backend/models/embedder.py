"""
Local embedding generation using sentence-transformers.
Model: all-MiniLM-L6-v2 (~80MB, runs fast on CPU, 384-dim output).
"""
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class Embedder:
    """Generate embeddings using sentence-transformers locally."""

    def __init__(self):
        self.model = _get_model()
        self.dimension = 384  # all-MiniLM-L6-v2 output dim

    def embed(self, text):
        """Embed a single text string. Returns list of floats."""
        vec = self.model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts):
        """Embed a list of texts. Returns list of list of floats."""
        vecs = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [v.tolist() for v in vecs]
