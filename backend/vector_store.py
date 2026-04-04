"""
FAISS-based vector store replacing ChromaDB.
Stores document chunks with embeddings for semantic search.
Persists index + metadata to disk as JSON + .faiss files.
"""
import os
import json
import numpy as np
import faiss


class FAISSStore:
    """Manages FAISS indices per document with persistence."""

    def __init__(self, persist_dir="faiss_store", dimension=384):
        self.persist_dir = os.path.abspath(persist_dir)
        self.dimension = dimension
        os.makedirs(self.persist_dir, exist_ok=True)
        # In-memory state: { doc_id: { "index": faiss.Index, "chunks": [...] } }
        self._docs = {}
        self._load_all()

    def _index_path(self, doc_id):
        return os.path.join(self.persist_dir, f"{doc_id}.faiss")

    def _meta_path(self, doc_id):
        return os.path.join(self.persist_dir, f"{doc_id}.json")

    def _load_all(self):
        """Load all persisted indices on startup."""
        if not os.path.exists(self.persist_dir):
            return
        for fname in os.listdir(self.persist_dir):
            if fname.endswith(".faiss"):
                doc_id = fname[:-6]
                self._load_doc(doc_id)

    def _load_doc(self, doc_id):
        idx_path = self._index_path(doc_id)
        meta_path = self._meta_path(doc_id)
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            index = faiss.read_index(idx_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            self._docs[doc_id] = {"index": index, "chunks": chunks}

    def has_document(self, doc_id):
        return doc_id in self._docs and len(self._docs[doc_id]["chunks"]) > 0

    def add_document(self, doc_id, texts, embeddings):
        """Add chunks for a document. Overwrites if doc_id exists."""
        arr = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after L2 norm)
        faiss.normalize_L2(arr)
        index.add(arr)
        chunks = [{"text": t, "doc_id": doc_id} for t in texts]
        self._docs[doc_id] = {"index": index, "chunks": chunks}
        self._save_doc(doc_id)

    def delete_document(self, doc_id):
        if doc_id in self._docs:
            del self._docs[doc_id]
        for path in (self._index_path(doc_id), self._meta_path(doc_id)):
            if os.path.exists(path):
                os.remove(path)

    def search(self, query_embedding, doc_id=None, top_k=5):
        """Search for similar chunks. If doc_id given, search only that doc."""
        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)

        results = []
        doc_ids = [doc_id] if doc_id else list(self._docs.keys())

        for did in doc_ids:
            if did not in self._docs:
                continue
            entry = self._docs[did]
            index = entry["index"]
            chunks = entry["chunks"]
            if index.ntotal == 0:
                continue
            k = min(top_k, index.ntotal)
            scores, indices = index.search(q, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(chunks):
                    continue
                results.append({
                    "text": chunks[idx]["text"],
                    "doc_id": did,
                    "score": float(score),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_chunks(self, doc_id):
        """Return all chunks for a document."""
        if doc_id not in self._docs:
            return []
        return list(self._docs[doc_id]["chunks"])

    def _save_doc(self, doc_id):
        if doc_id not in self._docs:
            return
        entry = self._docs[doc_id]
        faiss.write_index(entry["index"], self._index_path(doc_id))
        with open(self._meta_path(doc_id), "w", encoding="utf-8") as f:
            json.dump(entry["chunks"], f, ensure_ascii=False)

    def save(self):
        """Persist all indices to disk."""
        for doc_id in self._docs:
            self._save_doc(doc_id)
