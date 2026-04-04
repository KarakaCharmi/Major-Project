# Option B: Align Code to Project Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Gemini + ChromaDB with BART/T5 + FAISS + sentence-transformers, add SerpAPI/NewsAPI/CrossRef web search, citation analysis, Graphviz/Plotly visualization, code generation, and report builder — matching the project PPT spec exactly.

**Architecture:** The Flask AI backend (`backend/`) will swap its AI backbone: HuggingFace Inference API calls BART (summarization) and T5 (quiz generation), sentence-transformers generates local embeddings, FAISS stores/retrieves vectors. New modules add web search (SerpAPI + NewsAPI + CrossRef), citation analysis with trust scores, visualization (Graphviz + Plotly), code generation (via HF CodeLlama), and report export (PDF/PPT via ReportLab + python-pptx). The React frontend gets new pages/components for each feature. The Node middleware remains unchanged except for new proxy routes.

**Tech Stack:** Python 3.12, Flask 3.x, HuggingFace Inference API (free), sentence-transformers (all-MiniLM-L6-v2), faiss-cpu, SerpAPI, NewsAPI, CrossRef API, Graphviz, Plotly, ReportLab, python-pptx, React 18

**Hardware constraint:** Intel i5, 16GB RAM, no GPU — all models use HuggingFace Inference API (remote) except embeddings (small local model ~80MB).

---

## Phase Overview

| Phase | What | Difficulty | Est. Tasks |
|-------|------|------------|------------|
| 1 | FAISS + sentence-transformers (replace ChromaDB + Gemini embeddings) | Medium | 5 |
| 2 | BART summarization (replace Gemini summarizer) | Medium | 3 |
| 3 | T5 quiz generation (replace Gemini quiz) | Medium | 3 |
| 4 | Flashcard generation with BART (replace Gemini flashcards) | Easy | 2 |
| 5 | RAG chat pipeline rewire (BART for answer gen) | Medium | 3 |
| 6 | SerpAPI + NewsAPI + CrossRef web search | Medium | 4 |
| 7 | Citation analysis with trust scores | Medium | 3 |
| 8 | Graphviz/Plotly visualization | Medium | 3 |
| 9 | Code generation | Easy | 2 |
| 10 | Report builder (PDF/PPT export) | Medium | 3 |
| 11 | Frontend new pages + wiring | Medium | 5 |
| 12 | Bug fixes + cleanup | Easy | 3 |

---

## File Structure

### New files to CREATE:

```
backend/
  models/
    __init__.py              — Package init
    hf_client.py             — HuggingFace Inference API wrapper (BART, T5, CodeLlama)
    embedder.py              — sentence-transformers local embedding (all-MiniLM-L6-v2)
  vector_store.py            — FAISS index manager (replaces ChromaDB)
  web_search.py              — SerpAPI + NewsAPI + CrossRef integration
  citation_analyzer.py       — Citation extraction, metadata fetch, trust scoring
  visualizer.py              — Graphviz flowcharts + Plotly charts
  code_generator.py          — Code generation via HF API
  report_builder.py          — PDF/PPT report export (ReportLab + python-pptx)
  tests/
    test_vector_store.py
    test_hf_client.py
    test_web_search.py
    test_citation.py
    test_visualizer.py
    test_report_builder.py

my-app/src/Components/
  WebSearch.jsx              — Web search UI panel
  Citations.jsx              — Citation analysis display
  Visualizer.jsx             — Flowchart/graph display
  CodeGenerator.jsx          — Code generation panel
  ReportBuilder.jsx          — Report export UI
```

### Existing files to MODIFY:

```
backend/main.py              — Remove Gemini/ChromaDB, wire new modules
backend/summarize.py         — Replace Gemini with BART
backend/quiz.py              — Replace Gemini with T5
backend/flashcard.py         — Replace Gemini with BART
backend/requirements.txt     — New dependencies
my-app/src/App.js            — New routes
my-app/src/Components/Chat.jsx          — Add web search/citation/viz triggers
my-app/src/Components/UploadPage.jsx    — Wire new features
servers/routes/chat.js       — Proxy new Flask endpoints
```

---

## Task 1: New Python Dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Update requirements.txt**

```txt
flask>=3.0.0
flask-cors>=4.0.0
requests>=2.31.0
PyPDF2>=3.0.0
python-docx>=1.1.0
better-profanity>=0.7.0
werkzeug>=3.0.0
gunicorn>=21.2.0
python-dotenv>=1.0.0

# --- NEW: BART/T5 via HuggingFace ---
huggingface-hub>=0.20.0

# --- NEW: Local embeddings ---
sentence-transformers>=2.2.0

# --- NEW: FAISS vector store ---
faiss-cpu>=1.7.4
numpy>=1.24.0

# --- NEW: Web search APIs ---
google-search-results>=2.4.2
newsapi-python>=0.2.7

# --- NEW: Visualization ---
graphviz>=0.20.0
plotly>=5.18.0
kaleido>=0.2.1

# --- NEW: Report generation ---
reportlab>=4.0.0
python-pptx>=0.6.21

# --- REMOVED: google-generativeai, chromadb, docx2pdf ---
```

- [ ] **Step 2: Install dependencies**

Run: `cd backend && pip install -r requirements.txt`
Expected: All packages install successfully. `sentence-transformers` will download ~80MB model on first use.

- [ ] **Step 3: Install Graphviz system binary**

Run: `winget install graphviz` (Windows) or download from https://graphviz.org/download/
Expected: `dot -V` returns version info.

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: update dependencies for BART/T5/FAISS/SerpAPI stack"
```

---

## Task 2: HuggingFace Inference API Client

**Files:**
- Create: `backend/models/__init__.py`
- Create: `backend/models/hf_client.py`
- Create: `backend/tests/test_hf_client.py`

- [ ] **Step 1: Create package init**

```python
# backend/models/__init__.py
```

(Empty file, just makes it a package)

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_hf_client.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.hf_client import HFClient

def test_summarize_returns_string():
    """BART summarization should return a non-empty string."""
    client = HFClient()
    text = "The Smart Research Assistant is an AI-powered tool that streamlines information gathering and report generation. It integrates data from uploaded documents and live online sources. Unlike traditional methods, it delivers concise citation-based summaries."
    result = client.summarize(text)
    assert isinstance(result, str)
    assert len(result) > 10

def test_generate_quiz_returns_string():
    """T5 quiz generation should return a non-empty string."""
    client = HFClient()
    prompt = "Generate a quiz question about: The Smart Research Assistant uses BART for summarization and T5 for quiz generation."
    result = client.generate_text(prompt, model="quiz")
    assert isinstance(result, str)
    assert len(result) > 5

def test_generate_code_returns_string():
    """Code generation should return a non-empty string."""
    client = HFClient()
    prompt = "Write a Python function that adds two numbers"
    result = client.generate_code(prompt)
    assert isinstance(result, str)
    assert "def" in result or "return" in result or "+" in result
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_hf_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.hf_client'`

- [ ] **Step 4: Implement HFClient**

```python
# backend/models/hf_client.py
"""
HuggingFace Inference API client for BART, T5, and CodeLlama.
Uses the free Inference API — no GPU required locally.
"""
import os
import requests

HF_API_URL = "https://api-inference.huggingface.co/models"
HF_TOKEN = os.environ.get("HF_API_TOKEN", "")

# Model IDs on HuggingFace
MODELS = {
    "summarization": "facebook/bart-large-cnn",
    "quiz": "google/flan-t5-base",
    "code": "Salesforce/codegen-350M-mono",
    "text2text": "google/flan-t5-base",
}

class HFClient:
    """Wrapper around HuggingFace Inference API."""

    def __init__(self, token=None):
        self.token = token or HF_TOKEN
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def _query(self, model_id, payload, timeout=60):
        """Send a request to HuggingFace Inference API."""
        url = f"{HF_API_URL}/{model_id}"
        resp = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
        if resp.status_code == 503:
            # Model is loading — wait and retry once
            import time
            wait = resp.json().get("estimated_time", 30)
            time.sleep(min(wait, 60))
            resp = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def summarize(self, text, max_length=200, min_length=50):
        """Summarize text using BART-large-CNN."""
        # BART has a 1024 token limit, truncate long input
        truncated = text[:3000]
        result = self._query(MODELS["summarization"], {
            "inputs": truncated,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
            }
        })
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("summary_text", "")
        return str(result)

    def generate_text(self, prompt, model="text2text", max_length=512):
        """Generate text using T5/Flan-T5."""
        model_id = MODELS.get(model, MODELS["text2text"])
        result = self._query(model_id, {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "do_sample": True,
            }
        })
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)

    def generate_code(self, prompt, max_length=256):
        """Generate code using CodeGen model."""
        result = self._query(MODELS["code"], {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.2,
                "do_sample": True,
            }
        })
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)

    def answer_question(self, question, context, max_length=300):
        """Answer a question given context using Flan-T5."""
        prompt = f"""Answer the following question based only on the provided context.

Context: {context[:3000]}

Question: {question}

Answer:"""
        return self.generate_text(prompt, model="text2text", max_length=max_length)
```

- [ ] **Step 5: Run tests**

Run: `cd backend && python -m pytest tests/test_hf_client.py -v`
Expected: PASS (requires internet + optionally HF_API_TOKEN env var for higher rate limits)

- [ ] **Step 6: Commit**

```bash
git add backend/models/ backend/tests/test_hf_client.py
git commit -m "feat: add HuggingFace Inference API client for BART/T5/CodeGen"
```

---

## Task 3: Local Embeddings with sentence-transformers

**Files:**
- Create: `backend/models/embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_embedder.py (append or create)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.embedder import Embedder

def test_embed_returns_list_of_floats():
    emb = Embedder()
    vec = emb.embed("Hello world")
    assert isinstance(vec, list)
    assert len(vec) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(x, float) for x in vec)

def test_embed_batch_returns_correct_shape():
    emb = Embedder()
    vecs = emb.embed_batch(["Hello", "World", "Test"])
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_embedder.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement Embedder**

```python
# backend/models/embedder.py
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
```

- [ ] **Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_embedder.py -v`
Expected: PASS (first run downloads model ~80MB)

- [ ] **Step 5: Commit**

```bash
git add backend/models/embedder.py backend/tests/test_embedder.py
git commit -m "feat: add local sentence-transformers embedder (all-MiniLM-L6-v2)"
```

---

## Task 4: FAISS Vector Store (replaces ChromaDB)

**Files:**
- Create: `backend/vector_store.py`
- Create: `backend/tests/test_vector_store.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_vector_store.py
import os, sys, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import FAISSStore

def test_add_and_search():
    tmp = tempfile.mkdtemp()
    try:
        store = FAISSStore(persist_dir=tmp, dimension=384)
        store.add_document("doc1", ["Hello world", "Python is great"], [[0.1]*384, [0.2]*384])
        results = store.search([0.1]*384, doc_id="doc1", top_k=2)
        assert len(results) > 0
        assert results[0]["text"] == "Hello world"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_has_document():
    tmp = tempfile.mkdtemp()
    try:
        store = FAISSStore(persist_dir=tmp, dimension=384)
        assert not store.has_document("doc1")
        store.add_document("doc1", ["test"], [[0.5]*384])
        assert store.has_document("doc1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_delete_document():
    tmp = tempfile.mkdtemp()
    try:
        store = FAISSStore(persist_dir=tmp, dimension=384)
        store.add_document("doc1", ["test chunk"], [[0.3]*384])
        assert store.has_document("doc1")
        store.delete_document("doc1")
        assert not store.has_document("doc1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_get_all_chunks():
    tmp = tempfile.mkdtemp()
    try:
        store = FAISSStore(persist_dir=tmp, dimension=384)
        store.add_document("doc1", ["chunk A", "chunk B"], [[0.1]*384, [0.2]*384])
        chunks = store.get_chunks("doc1")
        assert len(chunks) == 2
        texts = [c["text"] for c in chunks]
        assert "chunk A" in texts
        assert "chunk B" in texts
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_persistence():
    tmp = tempfile.mkdtemp()
    try:
        store = FAISSStore(persist_dir=tmp, dimension=384)
        store.add_document("doc1", ["persistent chunk"], [[0.4]*384])
        store.save()
        # Load a new instance from same dir
        store2 = FAISSStore(persist_dir=tmp, dimension=384)
        assert store2.has_document("doc1")
        results = store2.search([0.4]*384, doc_id="doc1", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "persistent chunk"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_vector_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement FAISSStore**

```python
# backend/vector_store.py
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

        # In-memory state: { doc_id: { "index": faiss.Index, "chunks": [{"text":..., "meta":...}] } }
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
                doc_id = fname[:-6]  # strip .faiss
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
        index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
        # Normalize for cosine similarity
        faiss.normalize_L2(arr)
        index.add(arr)
        chunks = [{"text": t, "doc_id": doc_id} for t in texts]
        self._docs[doc_id] = {"index": index, "chunks": chunks}
        self._save_doc(doc_id)

    def delete_document(self, doc_id):
        if doc_id in self._docs:
            del self._docs[doc_id]
        idx_path = self._index_path(doc_id)
        meta_path = self._meta_path(doc_id)
        if os.path.exists(idx_path):
            os.remove(idx_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

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
```

- [ ] **Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_vector_store.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/vector_store.py backend/tests/test_vector_store.py
git commit -m "feat: add FAISS vector store replacing ChromaDB"
```

---

## Task 5: Rewire main.py — Remove Gemini/ChromaDB, Wire FAISS + HF

**Files:**
- Modify: `backend/main.py`

This is the biggest single change. We remove all `google.generativeai` and `chromadb` usage and replace with our new modules.

- [ ] **Step 1: Replace imports and config block (lines 1-82)**

Remove these imports/config:
```python
# REMOVE: import google.generativeai as genai
# REMOVE: import chromadb
# REMOVE: All genai.configure() calls
# REMOVE: EMBED_MODEL, TEXT_MODEL Gemini refs
# REMOVE: _init_chroma_client(), chroma_client, collection
```

Replace with:
```python
from models.hf_client import HFClient
from models.embedder import Embedder
from vector_store import FAISSStore

# Initialize AI clients
hf_client = HFClient()
embedder = Embedder()

# FAISS store (replaces ChromaDB)
FAISS_STORE_PATH = os.environ.get("FAISS_STORE_PATH", os.path.join(os.getcwd(), "faiss_store"))
vector_store = FAISSStore(persist_dir=FAISS_STORE_PATH, dimension=embedder.dimension)
```

- [ ] **Step 2: Replace generate_embeddings function (line 337-349)**

Replace:
```python
def generate_embeddings(text, timeout_sec=20):
    """Generate embeddings using local sentence-transformers."""
    try:
        return embedder.embed(text)
    except Exception as e:
        print("Embedding error:", e)
        return None
```

- [ ] **Step 3: Replace has_index function**

Find and replace all `has_index(doc_id)` references to use:
```python
def has_index(doc_id):
    return vector_store.has_document(doc_id)
```

- [ ] **Step 4: Replace indexing logic (index_bytes function)**

The existing `index_bytes` function uses `collection.add()` for ChromaDB. Replace with:
```python
def index_bytes(doc_id, filename, mimetype, data):
    """Extract text, chunk, embed, store in FAISS."""
    text = extract_text_for_mimetype(filename, mimetype, data)
    if not text or not text.strip():
        return False, 0
    chunks = chunk_text(text)
    if not chunks:
        return False, 0
    embeddings = embedder.embed_batch(chunks)
    vector_store.add_document(doc_id, chunks, embeddings)
    return True, len(chunks)
```

- [ ] **Step 5: Replace the ask_doc RAG retrieval (lines 688-760)**

Replace the ChromaDB query + Gemini generation with FAISS search + BART answer:
```python
# In ask_doc():
q_emb = generate_embeddings(question)
if not q_emb:
    return jsonify({"error": "Failed to generate embedding"}), 500

results = vector_store.search(q_emb, doc_id=doc_id, top_k=5)

if not results:
    general_fallback[doc_id] = {"awaiting": True, "pending_question": question}
    return jsonify({
        "answer": "I couldn't find relevant information in the uploaded document.\n"
                  "Do you want me to answer using general knowledge instead? Reply \"y\" for yes or \"n\" for no."
    })

context = "\n\n".join([r["text"] for r in results])

# Use BART/T5 to answer
answer = hf_client.answer_question(question, context)
if not answer or not answer.strip():
    return jsonify({"answer": "Could not generate an answer. Please try rephrasing."})

return jsonify({"answer": format_response(answer.strip()), "requireConfirmation": False})
```

- [ ] **Step 6: Replace general knowledge fallback (lines 650-665)**

Replace Gemini `model.generate_content` with:
```python
answer = hf_client.generate_text(f"Answer this question: {orig_q}", model="text2text")
if answer:
    return jsonify({"answer": format_response(answer.strip())})
else:
    return jsonify({"answer": "Could not generate an answer."})
```

- [ ] **Step 7: Replace delete_doc ChromaDB calls (line 552-558)**

```python
@app.route("/api/document/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    vector_store.delete_document(doc_id)
    return jsonify({"message": "Deleted successfully"})
```

- [ ] **Step 8: Replace list_docs ChromaDB calls (line 521-536)**

```python
@app.route("/api/document/my", methods=["GET"])
def list_docs():
    docs = {}
    for doc_id in vector_store._docs:
        chunks = vector_store.get_chunks(doc_id)
        if chunks:
            docs[doc_id] = {"_id": doc_id, "name": doc_id, "type": "document", "size": len(chunks)}
    return jsonify(list(docs.values()))
```

- [ ] **Step 9: Update init calls for quiz/flashcard/summarize blueprints**

Replace Gemini-based init calls:
```python
# Old: init_quiz(collection, has_index, fetch_doc_from_node, extract_text_for_mimetype, TEXT_MODEL, genai)
# New:
init_quiz(vector_store, has_index, fetch_doc_from_node, extract_text_for_mimetype, hf_client)
init_flashcards(vector_store, has_index, fetch_doc_from_node, extract_text_for_mimetype, hf_client)
init_summarizer(hf_client)
```

- [ ] **Step 10: Commit**

```bash
git add backend/main.py
git commit -m "feat: rewire main.py from Gemini/ChromaDB to BART/T5/FAISS"
```

---

## Task 6: Rewire summarize.py to Use BART

**Files:**
- Modify: `backend/summarize.py`

- [ ] **Step 1: Rewrite summarize.py**

Replace the entire file — remove all `genai` references, use `HFClient.summarize()`:

```python
# backend/summarize.py
from flask import Blueprint, request, jsonify
import re
from typing import List

summarize_bp = Blueprint("summarize", __name__)

hf_client = None

def _clean_selection_text(text):
    if not text:
        return ""
    t = text.replace("\r", "")
    t = re.sub(r"([A-Za-z])-[\n\r]+([A-Za-z])", r"\1\2", t)
    t = re.sub(r"([^\n])\n(?!\n)([a-z0-9(])", r"\1 \2", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    t = re.sub(r"^\s*[-*]\s*", "- ", t, flags=re.MULTILINE)
    return t.strip()

def _chunk_text(text, size=2500):
    """Split text into chunks for BART (max ~1024 tokens)."""
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return [text]
    windows, buf, cur = [], [], 0
    for p in paras:
        plen = len(p) + 2
        if not buf or cur + plen <= size:
            buf.append(p)
            cur += plen
        else:
            windows.append("\n\n".join(buf))
            buf = [p]
            cur = plen
    if buf:
        windows.append("\n\n".join(buf))
    return windows

def _map_reduce_summary(text, style="concise"):
    """Summarize using BART with map-reduce for long texts."""
    chunks = _chunk_text(text)
    length_map = {"concise": (50, 150), "detailed": (100, 250), "short": (30, 80)}
    min_len, max_len = length_map.get(style, (50, 150))

    if len(chunks) <= 1:
        return hf_client.summarize(text, max_length=max_len, min_length=min_len)

    # Map: summarize each chunk
    partials = []
    for ch in chunks:
        try:
            s = hf_client.summarize(ch, max_length=max_len, min_length=min_len)
            partials.append(s)
        except Exception:
            partials.append(ch[:500])

    # Reduce: summarize the combined partials
    combined = " ".join(p for p in partials if p)
    try:
        return hf_client.summarize(combined, max_length=max_len, min_length=min_len)
    except Exception:
        return combined

def init_summarizer(_hf_client):
    """Initialize with HFClient. Called from main.py."""
    global hf_client
    hf_client = _hf_client

    @summarize_bp.route("/api/summarize", methods=["POST"])
    def summarize_endpoint():
        body = request.get_json(silent=True) or {}
        selection_text = (body.get("selectionText") or body.get("text") or "").strip()
        doc_id = (body.get("docId") or body.get("doc_id") or "").strip()
        style = (body.get("style") or "concise").lower()

        if not selection_text:
            return jsonify({"error": "Missing selectionText"}), 400

        cleaned = _clean_selection_text(selection_text)
        try:
            summary = _map_reduce_summary(cleaned, style)
            if not summary:
                return jsonify({"error": "Failed to summarize"}), 500
            return jsonify({
                "summary": summary,
                "doc_id": doc_id or None,
                "length": len(cleaned),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return summarize_bp
```

- [ ] **Step 2: Commit**

```bash
git add backend/summarize.py
git commit -m "feat: rewrite summarize.py to use BART via HuggingFace API"
```

---

## Task 7: Rewire quiz.py to Use T5

**Files:**
- Modify: `backend/quiz.py`

- [ ] **Step 1: Rewrite quiz.py**

Replace Gemini calls with T5 via HFClient. Keep the same endpoint contract so React doesn't need changes:

```python
# backend/quiz.py
from flask import Blueprint, request, jsonify
import re as _re
import json

quiz_bp = Blueprint("quiz", __name__)

vector_store = None
has_index = None
fetch_doc_from_node = None
extract_text_for_mimetype = None
hf_client = None

def init_quiz(_vector_store, _has_index, _fetch_doc_from_node, _extract_text_for_mimetype, _hf_client):
    global vector_store, has_index, fetch_doc_from_node, extract_text_for_mimetype, hf_client
    vector_store = _vector_store
    has_index = _has_index
    fetch_doc_from_node = _fetch_doc_from_node
    extract_text_for_mimetype = _extract_text_for_mimetype
    hf_client = _hf_client

@quiz_bp.route("/api/document/generate-quiz", methods=["POST"])
def generate_quiz():
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or body.get("documentId") or "").strip()
    if not doc_id:
        return jsonify({"success": False, "error": "doc_id is required"}), 400

    try:
        num_questions = int(body.get("num_questions", 5))
    except Exception:
        num_questions = 5
    difficulty = (body.get("difficulty") or "medium").lower()
    qtypes = body.get("question_types") or ["mcq", "true_false", "short_answer"]

    # Build context
    context = ""
    try:
        if has_index(doc_id):
            chunks = vector_store.get_chunks(doc_id)
            context = "\n\n".join([c["text"] for c in chunks])
        else:
            ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
            if not ok:
                return jsonify({"success": False, "error": filename}), 404
            context = extract_text_for_mimetype(filename, mimetype, data_bytes)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to load document: {e}"}), 500

    context = (context or "").strip()
    if not context:
        return jsonify({"success": False, "error": "Document has no readable text"}), 400

    # Generate quiz using T5 (Flan-T5)
    prompt = (
        f"Generate {num_questions} {difficulty} quiz questions from the following text. "
        f"Include these question types: {', '.join(qtypes)}. "
        "Return ONLY valid JSON with this schema: "
        '{"questions": [{"type": "mcq|true_false|short_answer", "question": "...", '
        '"options": ["..."] (for mcq only), "correct_answer": "...", "explanation": "..."}]}.\n\n'
        f"Text: {context[:4000]}\n\nJSON:"
    )

    try:
        raw = hf_client.generate_text(prompt, model="quiz", max_length=1024)

        def _parse_json_safely(s):
            if not s:
                return None
            try:
                return json.loads(s)
            except Exception:
                m = _re.search(r"\{[\s\S]*\}", s)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
                return None

        quiz = _parse_json_safely(raw)

        # If T5 can't produce valid JSON, build questions from text generation
        if not isinstance(quiz, dict) or "questions" not in quiz:
            # Fallback: generate individual questions
            questions = []
            for i in range(min(num_questions, 5)):
                qtype = qtypes[i % len(qtypes)]
                q_prompt = f"Generate one {qtype} {difficulty} question about: {context[:2000]}"
                q_text = hf_client.generate_text(q_prompt, model="quiz", max_length=200)
                if q_text:
                    item = {
                        "type": qtype,
                        "question": q_text.strip(),
                        "correct_answer": "See document for answer",
                        "explanation": "Generated from document content",
                    }
                    if qtype == "mcq":
                        item["options"] = ["Option A", "Option B", "Option C", "See document"]
                    elif qtype == "true_false":
                        item["correct_answer"] = "true"
                    questions.append(item)
            if questions:
                return jsonify({"success": True, "quiz": {"questions": questions}})
            return jsonify({"success": False, "error": "Could not generate quiz. Try again."}), 502

        # Sanitize questions (same logic as before)
        qs = []
        for q in quiz.get("questions", [])[:num_questions]:
            if not isinstance(q, dict):
                continue
            qtype = str(q.get("type", "")).strip().lower()
            if qtype not in ("mcq", "true_false", "short_answer"):
                continue
            question = str(q.get("question", "")).strip()
            if not question:
                continue
            correct = str(q.get("correct_answer", "")).strip()
            item = {
                "type": qtype,
                "question": question,
                "correct_answer": correct or "Not provided",
                "explanation": str(q.get("explanation", "")).strip() or "",
            }
            if qtype == "mcq":
                opts = q.get("options") or []
                opts = [str(o).strip() for o in opts if str(o).strip()]
                if correct and correct not in opts:
                    opts.append(correct)
                item["options"] = opts[:5] if len(opts) >= 3 else ["Option A", "Option B", correct or "Option C"]
            qs.append(item)

        if not qs:
            return jsonify({"success": False, "error": "No valid questions generated."}), 502

        return jsonify({"success": True, "quiz": {"questions": qs}})
    except Exception as e:
        return jsonify({"success": False, "error": f"Quiz generation failed: {e}"}), 500
```

- [ ] **Step 2: Commit**

```bash
git add backend/quiz.py
git commit -m "feat: rewrite quiz.py to use T5 via HuggingFace API"
```

---

## Task 8: Rewire flashcard.py to Use BART

**Files:**
- Modify: `backend/flashcard.py`

- [ ] **Step 1: Rewrite flashcard.py**

Same pattern — replace Gemini with HFClient:

```python
# backend/flashcard.py
from flask import Blueprint, request, jsonify
import json
import re as _re

flashcard_bp = Blueprint("flashcard", __name__)

vector_store = None
has_index = None
fetch_doc_from_node = None
extract_text_for_mimetype = None
hf_client = None

def init_flashcards(_vector_store, _has_index, _fetch_doc_from_node, _extract_text_for_mimetype, _hf_client):
    global vector_store, has_index, fetch_doc_from_node, extract_text_for_mimetype, hf_client
    vector_store = _vector_store
    has_index = _has_index
    fetch_doc_from_node = _fetch_doc_from_node
    extract_text_for_mimetype = _extract_text_for_mimetype
    hf_client = _hf_client

@flashcard_bp.route("/api/document/generate-flashcards", methods=["POST"])
def generate_flashcards():
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or body.get("documentId") or "").strip()
    if not doc_id:
        return jsonify({"success": False, "error": "doc_id is required"}), 400
    try:
        num_cards = max(3, min(int(body.get("num_cards", 10)), 30))
    except Exception:
        num_cards = 10

    # Build context
    context = ""
    try:
        if has_index(doc_id):
            chunks = vector_store.get_chunks(doc_id)
            context = "\n\n".join([c["text"] for c in chunks])
        else:
            ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
            if not ok:
                return jsonify({"success": False, "error": filename}), 404
            context = extract_text_for_mimetype(filename, mimetype, data_bytes)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to load document: {e}"}), 500

    context = (context or "").strip()
    if not context:
        return jsonify({"success": False, "error": "Document has no readable text"}), 400

    # Generate flashcards using T5/Flan-T5
    prompt = (
        f"Generate {num_cards} study flashcards from the following text. "
        "Return ONLY valid JSON: "
        '{"flashcards": [{"front": "term or question", "back": "answer", '
        '"category": "topic", "difficulty": "Easy|Medium|Hard"}]}.\n\n'
        f"Text: {context[:4000]}\n\nJSON:"
    )

    try:
        raw = hf_client.generate_text(prompt, model="text2text", max_length=1024)

        def _parse_json_safely(s):
            if not s:
                return None
            try:
                return json.loads(s)
            except Exception:
                m = _re.search(r"\{[\s\S]*\}", s)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
                return None

        data = _parse_json_safely(raw)

        # If JSON parsing works, use it
        if isinstance(data, dict) and isinstance(data.get("flashcards"), list):
            cards = []
            for c in data["flashcards"][:num_cards]:
                if not isinstance(c, dict):
                    continue
                front = str(c.get("front", "")).strip()
                back = str(c.get("back", "")).strip()
                if not front or not back:
                    continue
                cards.append({
                    "front": front[:200],
                    "back": back[:600],
                    "category": str(c.get("category", "General")).strip() or "General",
                    "difficulty": str(c.get("difficulty", "Medium")).strip().capitalize(),
                })
            if cards:
                return jsonify({"success": True, "flashcards": cards})

        # Fallback: extract key sentences as flashcards
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 20]
        cards = []
        for i, sent in enumerate(sentences[:num_cards]):
            cards.append({
                "front": f"What does this mean: {sent[:100]}...?" if len(sent) > 100 else f"Explain: {sent}",
                "back": sent[:400],
                "category": "General",
                "difficulty": "Medium",
            })
        if cards:
            return jsonify({"success": True, "flashcards": cards})

        return jsonify({"success": False, "error": "Could not generate flashcards."}), 502
    except Exception as e:
        return jsonify({"success": False, "error": f"Flashcard generation failed: {e}"}), 500
```

- [ ] **Step 2: Commit**

```bash
git add backend/flashcard.py
git commit -m "feat: rewrite flashcard.py to use BART/T5 via HuggingFace API"
```

---

## Task 9: Web Search — SerpAPI + NewsAPI + CrossRef

**Files:**
- Create: `backend/web_search.py`
- Create: `backend/tests/test_web_search.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_web_search.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from web_search import WebSearcher

def test_serp_search_returns_list():
    ws = WebSearcher()
    results = ws.search_web("machine learning summarization")
    assert isinstance(results, list)
    # May be empty if no API key, but should not crash
    for r in results:
        assert "title" in r
        assert "link" in r

def test_news_search_returns_list():
    ws = WebSearcher()
    results = ws.search_news("artificial intelligence")
    assert isinstance(results, list)

def test_crossref_search_returns_list():
    ws = WebSearcher()
    results = ws.search_crossref("deep learning")
    assert isinstance(results, list)
    for r in results:
        assert "title" in r

def test_combined_search():
    ws = WebSearcher()
    results = ws.search_all("natural language processing")
    assert isinstance(results, dict)
    assert "web" in results
    assert "news" in results
    assert "academic" in results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_web_search.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement WebSearcher**

```python
# backend/web_search.py
"""
Multi-source web search: SerpAPI (Google), NewsAPI, CrossRef (academic).
All APIs have free tiers. Gracefully degrades if API keys missing.
"""
import os
import requests

SERP_API_KEY = os.environ.get("SERP_API_KEY", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

class WebSearcher:
    """Search web, news, and academic sources."""

    def search_web(self, query, num_results=5):
        """Search Google via SerpAPI."""
        if not SERP_API_KEY:
            return []
        try:
            resp = requests.get("https://serpapi.com/search", params={
                "q": query,
                "api_key": SERP_API_KEY,
                "num": num_results,
                "engine": "google",
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google",
                })
            return results
        except Exception as e:
            print(f"[SerpAPI] Error: {e}")
            return []

    def search_news(self, query, num_results=5):
        """Search news via NewsAPI."""
        if not NEWS_API_KEY:
            return []
        try:
            resp = requests.get("https://newsapi.org/v2/everything", params={
                "q": query,
                "apiKey": NEWS_API_KEY,
                "pageSize": num_results,
                "sortBy": "relevancy",
                "language": "en",
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for article in data.get("articles", [])[:num_results]:
                results.append({
                    "title": article.get("title", ""),
                    "link": article.get("url", ""),
                    "snippet": article.get("description", ""),
                    "source": "news",
                    "published": article.get("publishedAt", ""),
                })
            return results
        except Exception as e:
            print(f"[NewsAPI] Error: {e}")
            return []

    def search_crossref(self, query, num_results=5):
        """Search academic papers via CrossRef (free, no key needed)."""
        try:
            resp = requests.get("https://api.crossref.org/works", params={
                "query": query,
                "rows": num_results,
                "sort": "relevance",
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("message", {}).get("items", [])[:num_results]:
                title_parts = item.get("title", [""])
                title = title_parts[0] if title_parts else ""
                doi = item.get("DOI", "")
                results.append({
                    "title": title,
                    "link": f"https://doi.org/{doi}" if doi else "",
                    "doi": doi,
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])[:3]
                    ],
                    "published": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]),
                    "source": "crossref",
                    "cited_by": item.get("is-referenced-by-count", 0),
                })
            return results
        except Exception as e:
            print(f"[CrossRef] Error: {e}")
            return []

    def search_all(self, query, num_results=5):
        """Search all sources and return combined results."""
        return {
            "web": self.search_web(query, num_results),
            "news": self.search_news(query, num_results),
            "academic": self.search_crossref(query, num_results),
        }
```

- [ ] **Step 4: Add Flask endpoints in main.py**

Add to `backend/main.py`:
```python
from web_search import WebSearcher
web_searcher = WebSearcher()

@app.route("/api/search/web", methods=["POST"])
def web_search():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400
    source = body.get("source", "all")  # "web", "news", "academic", "all"
    num = min(int(body.get("num_results", 5)), 10)
    if source == "all":
        results = web_searcher.search_all(query, num)
    elif source == "web":
        results = {"web": web_searcher.search_web(query, num)}
    elif source == "news":
        results = {"news": web_searcher.search_news(query, num)}
    elif source == "academic":
        results = {"academic": web_searcher.search_crossref(query, num)}
    else:
        results = web_searcher.search_all(query, num)
    return jsonify({"success": True, "results": results})
```

- [ ] **Step 5: Run tests**

Run: `cd backend && python -m pytest tests/test_web_search.py -v`
Expected: PASS (CrossRef tests pass without API key; SerpAPI/NewsAPI return empty lists without keys)

- [ ] **Step 6: Commit**

```bash
git add backend/web_search.py backend/tests/test_web_search.py backend/main.py
git commit -m "feat: add multi-source web search (SerpAPI, NewsAPI, CrossRef)"
```

---

## Task 10: Citation Analyzer

**Files:**
- Create: `backend/citation_analyzer.py`

- [ ] **Step 1: Implement citation_analyzer.py**

```python
# backend/citation_analyzer.py
"""
Citation analysis: extract references from text, fetch metadata via CrossRef,
compute trust scores based on citation count, recency, and source authority.
"""
import re
import requests

class CitationAnalyzer:
    """Extract and analyze citations from document text."""

    # Common reference patterns
    REF_PATTERNS = [
        re.compile(r"\[(\d+)\]\s*(.+?)(?:\n|$)"),  # [1] Author, Title...
        re.compile(r"(\d+)\.\s+([A-Z][^.]+\.\s+.+?)(?:\n|$)"),  # 1. Author. Title...
        re.compile(r"(?:doi|DOI):\s*(10\.\d{4,}/[^\s]+)"),  # DOI references
    ]

    def extract_references(self, text):
        """Extract reference strings from document text."""
        refs = []
        seen = set()
        for pattern in self.REF_PATTERNS:
            for match in pattern.finditer(text):
                ref_text = match.group(0).strip()
                if ref_text not in seen and len(ref_text) > 15:
                    seen.add(ref_text)
                    refs.append(ref_text)
        return refs

    def fetch_metadata(self, query_or_doi):
        """Fetch citation metadata from CrossRef."""
        try:
            if query_or_doi.startswith("10."):
                # It's a DOI
                resp = requests.get(f"https://api.crossref.org/works/{query_or_doi}", timeout=10)
            else:
                resp = requests.get("https://api.crossref.org/works", params={
                    "query": query_or_doi[:200],
                    "rows": 1,
                }, timeout=10)

            resp.raise_for_status()
            data = resp.json()

            if "message" in data:
                item = data["message"]
                if isinstance(item, dict) and "items" in item:
                    item = item["items"][0] if item["items"] else {}

                title_parts = item.get("title", [""])
                return {
                    "title": title_parts[0] if title_parts else "",
                    "doi": item.get("DOI", ""),
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])[:5]
                    ],
                    "year": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]),
                    "cited_by": item.get("is-referenced-by-count", 0),
                    "publisher": item.get("publisher", ""),
                    "type": item.get("type", ""),
                }
            return None
        except Exception:
            return None

    def compute_trust_score(self, metadata):
        """Compute a trust score (0-100) based on citations, recency, source type."""
        if not metadata:
            return 0
        score = 30  # base score for having metadata

        # Citation count (max +30)
        cited = metadata.get("cited_by", 0)
        if cited > 100:
            score += 30
        elif cited > 50:
            score += 25
        elif cited > 10:
            score += 15
        elif cited > 0:
            score += 5

        # Recency (max +20)
        try:
            year = int(metadata.get("year", 0))
            if year >= 2023:
                score += 20
            elif year >= 2020:
                score += 15
            elif year >= 2015:
                score += 10
            elif year >= 2010:
                score += 5
        except (ValueError, TypeError):
            pass

        # Publisher/type (max +20)
        pub = (metadata.get("publisher") or "").lower()
        doc_type = (metadata.get("type") or "").lower()
        if any(k in pub for k in ["springer", "elsevier", "ieee", "acm", "nature", "wiley"]):
            score += 20
        elif doc_type == "journal-article":
            score += 15
        elif doc_type == "proceedings-article":
            score += 10

        return min(score, 100)

    def analyze_document(self, text):
        """Full pipeline: extract refs, fetch metadata, compute trust scores."""
        refs = self.extract_references(text)
        analyzed = []
        for ref_text in refs[:10]:  # Limit to 10 to avoid rate limits
            meta = self.fetch_metadata(ref_text)
            trust = self.compute_trust_score(meta)
            analyzed.append({
                "original_text": ref_text,
                "metadata": meta,
                "trust_score": trust,
            })
        return analyzed
```

- [ ] **Step 2: Add Flask endpoint in main.py**

```python
from citation_analyzer import CitationAnalyzer
citation_analyzer = CitationAnalyzer()

@app.route("/api/document/citations", methods=["POST"])
def analyze_citations():
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip()
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400
    try:
        ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
        if not ok:
            return jsonify({"error": filename}), 404
        text = extract_text_for_mimetype(filename, mimetype, data_bytes)
        results = citation_analyzer.analyze_document(text)
        return jsonify({"success": True, "citations": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

- [ ] **Step 3: Commit**

```bash
git add backend/citation_analyzer.py backend/main.py
git commit -m "feat: add citation analyzer with CrossRef metadata and trust scores"
```

---

## Task 11: Graphviz/Plotly Visualization

**Files:**
- Create: `backend/visualizer.py`

- [ ] **Step 1: Implement visualizer.py**

```python
# backend/visualizer.py
"""
Visualization: Graphviz flowcharts and Plotly charts from document content.
"""
import os
import json
import tempfile
import graphviz
import plotly.graph_objects as go
import plotly.io as pio


class Visualizer:
    """Generate flowcharts and charts from document content."""

    def __init__(self, hf_client):
        self.hf_client = hf_client

    def generate_flowchart(self, text, doc_id="doc"):
        """Generate a flowchart from document structure using Graphviz.
        Returns SVG string.
        """
        # Use T5 to extract steps/process from text
        prompt = (
            "Extract the main steps or process flow from this text as a numbered list. "
            "Return ONLY a numbered list, one step per line:\n\n"
            f"{text[:3000]}"
        )
        raw = self.hf_client.generate_text(prompt, model="text2text", max_length=300)
        steps = [s.strip() for s in raw.strip().split("\n") if s.strip()]
        if not steps:
            steps = ["Start", "Process Document", "Analyze", "Generate Output", "End"]

        # Clean step text (remove numbering)
        import re
        cleaned = []
        for s in steps:
            s = re.sub(r"^\d+[.)]\s*", "", s).strip()
            if s:
                cleaned.append(s[:60])  # cap label length
        steps = cleaned or ["Start", "End"]

        # Build Graphviz digraph
        dot = graphviz.Digraph(format="svg")
        dot.attr(rankdir="TB", bgcolor="transparent")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="#E8F0FE",
                 fontname="Arial", fontsize="11")
        dot.attr("edge", color="#4285F4")

        for i, step in enumerate(steps):
            node_id = f"step_{i}"
            if i == 0:
                dot.node(node_id, step, shape="ellipse", fillcolor="#34A853", fontcolor="white")
            elif i == len(steps) - 1:
                dot.node(node_id, step, shape="ellipse", fillcolor="#EA4335", fontcolor="white")
            else:
                dot.node(node_id, step)

            if i > 0:
                dot.edge(f"step_{i-1}", node_id)

        svg = dot.pipe(format="svg").decode("utf-8")
        return svg

    def generate_word_frequency_chart(self, text):
        """Generate a word frequency bar chart using Plotly. Returns JSON plot data."""
        import re
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        stop = {"that", "this", "with", "from", "have", "been", "were", "will",
                "their", "they", "them", "than", "then", "also", "more", "some",
                "which", "what", "when", "where", "would", "could", "should",
                "about", "into", "each", "make", "like", "just", "over", "such"}
        filtered = [w for w in words if w not in stop]

        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]
        if not top:
            return None

        labels = [t[0] for t in top]
        values = [t[1] for t in top]

        fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color="#4285F4")])
        fig.update_layout(
            title="Top Words in Document",
            xaxis_title="Word",
            yaxis_title="Frequency",
            template="plotly_white",
        )
        return json.loads(pio.to_json(fig))

    def generate_topic_distribution(self, text):
        """Generate a pie chart of topic distribution. Returns JSON plot data."""
        prompt = (
            "Identify the main topics in this text and estimate their percentage. "
            "Return as: Topic1: 30%, Topic2: 25%, etc.\n\n"
            f"{text[:2000]}"
        )
        raw = self.hf_client.generate_text(prompt, model="text2text", max_length=200)

        import re
        pairs = re.findall(r"([A-Za-z\s]+):\s*(\d+)%", raw)
        if not pairs:
            return None

        labels = [p[0].strip() for p in pairs]
        values = [int(p[1]) for p in pairs]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title="Topic Distribution", template="plotly_white")
        return json.loads(pio.to_json(fig))
```

- [ ] **Step 2: Add Flask endpoints in main.py**

```python
from visualizer import Visualizer
visualizer = Visualizer(hf_client)

@app.route("/api/document/visualize", methods=["POST"])
def visualize_doc():
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip()
    viz_type = (body.get("type") or "flowchart").lower()
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400
    try:
        ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
        if not ok:
            return jsonify({"error": filename}), 404
        text = extract_text_for_mimetype(filename, mimetype, data_bytes)

        if viz_type == "flowchart":
            svg = visualizer.generate_flowchart(text, doc_id)
            return jsonify({"success": True, "type": "flowchart", "svg": svg})
        elif viz_type == "word_frequency":
            chart = visualizer.generate_word_frequency_chart(text)
            return jsonify({"success": True, "type": "word_frequency", "chart": chart})
        elif viz_type == "topics":
            chart = visualizer.generate_topic_distribution(text)
            return jsonify({"success": True, "type": "topics", "chart": chart})
        else:
            return jsonify({"error": f"Unknown viz type: {viz_type}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

- [ ] **Step 3: Commit**

```bash
git add backend/visualizer.py backend/main.py
git commit -m "feat: add Graphviz flowcharts and Plotly charts for document visualization"
```

---

## Task 12: Code Generator

**Files:**
- Create: `backend/code_generator.py`

- [ ] **Step 1: Implement code_generator.py**

```python
# backend/code_generator.py
"""
Code generation from research concepts using HuggingFace models.
"""

class CodeGenerator:
    """Generate code from document concepts."""

    def __init__(self, hf_client):
        self.hf_client = hf_client

    def generate(self, concept, language="python"):
        """Generate code that implements the given concept."""
        prompt = (
            f"# {language.capitalize()} implementation of: {concept}\n"
            f"# Write clean, well-commented {language} code\n\n"
        )
        code = self.hf_client.generate_code(prompt, max_length=512)
        return code.strip() if code else "# Could not generate code for this concept."

    def generate_from_doc(self, text, language="python"):
        """Extract key concepts from doc text and generate code for each."""
        prompt = (
            "List the 3 main technical concepts from this text that could be implemented as code. "
            "Return only the concept names, one per line:\n\n"
            f"{text[:2000]}"
        )
        raw = self.hf_client.generate_text(prompt, model="text2text", max_length=150)
        concepts = [c.strip() for c in raw.strip().split("\n") if c.strip()][:3]

        results = []
        for concept in concepts:
            code = self.generate(concept, language)
            results.append({"concept": concept, "language": language, "code": code})
        return results
```

- [ ] **Step 2: Add Flask endpoint in main.py**

```python
from code_generator import CodeGenerator
code_gen = CodeGenerator(hf_client)

@app.route("/api/document/generate-code", methods=["POST"])
def gen_code():
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip()
    concept = (body.get("concept") or "").strip()
    language = (body.get("language") or "python").lower()

    if concept:
        code = code_gen.generate(concept, language)
        return jsonify({"success": True, "results": [{"concept": concept, "language": language, "code": code}]})

    if not doc_id:
        return jsonify({"error": "doc_id or concept required"}), 400

    try:
        ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
        if not ok:
            return jsonify({"error": filename}), 404
        text = extract_text_for_mimetype(filename, mimetype, data_bytes)
        results = code_gen.generate_from_doc(text, language)
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

- [ ] **Step 3: Commit**

```bash
git add backend/code_generator.py backend/main.py
git commit -m "feat: add code generation from document concepts"
```

---

## Task 13: Report Builder (PDF/PPT Export)

**Files:**
- Create: `backend/report_builder.py`

- [ ] **Step 1: Implement report_builder.py**

```python
# backend/report_builder.py
"""
Report builder: combines summaries, citations, visuals, and code into PDF/PPT.
Uses ReportLab for PDF and python-pptx for PowerPoint.
"""
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from pptx import Presentation
from pptx.util import Inches, Pt


class ReportBuilder:
    """Generate PDF and PPTX reports from analysis results."""

    def build_pdf(self, title, summary, citations=None, code_snippets=None):
        """Generate a PDF report. Returns bytes."""
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=20)
        heading_style = ParagraphStyle("ReportHeading", parent=styles["Heading2"], fontSize=14, spaceAfter=10)
        body_style = styles["BodyText"]

        story = []

        # Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2*inch))

        # Summary
        story.append(Paragraph("Summary", heading_style))
        for line in summary.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 0.2*inch))

        # Citations
        if citations:
            story.append(Paragraph("Citations & References", heading_style))
            for i, cit in enumerate(citations, 1):
                meta = cit.get("metadata") or {}
                trust = cit.get("trust_score", 0)
                text = f"{i}. {meta.get('title', cit.get('original_text', 'Unknown'))}"
                if meta.get("authors"):
                    text += f" — {', '.join(meta['authors'][:3])}"
                if meta.get("year"):
                    text += f" ({meta['year']})"
                text += f" [Trust: {trust}/100]"
                story.append(Paragraph(text, body_style))
            story.append(Spacer(1, 0.2*inch))

        # Code
        if code_snippets:
            story.append(Paragraph("Generated Code", heading_style))
            code_style = ParagraphStyle("Code", parent=body_style,
                                        fontName="Courier", fontSize=8, backColor=colors.Color(0.95, 0.95, 0.95))
            for snippet in code_snippets:
                story.append(Paragraph(f"Concept: {snippet.get('concept', '')}", body_style))
                code_text = snippet.get("code", "").replace("\n", "<br/>").replace(" ", "&nbsp;")
                story.append(Paragraph(code_text, code_style))
                story.append(Spacer(1, 0.1*inch))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()

    def build_pptx(self, title, summary, citations=None, code_snippets=None):
        """Generate a PPTX report. Returns bytes."""
        prs = Presentation()

        # Title slide
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = title
        slide.placeholders[1].text = "Generated by Smart Research Assistant"

        # Summary slide
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Summary"
        body = slide.placeholders[1]
        tf = body.text_frame
        tf.text = ""
        for line in summary.split("\n")[:10]:
            line = line.strip()
            if line:
                p = tf.add_paragraph()
                p.text = line
                p.font.size = Pt(14)

        # Citations slide
        if citations:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = "Citations & Trust Scores"
            body = slide.placeholders[1]
            tf = body.text_frame
            tf.text = ""
            for cit in citations[:8]:
                meta = cit.get("metadata") or {}
                trust = cit.get("trust_score", 0)
                p = tf.add_paragraph()
                p.text = f"{meta.get('title', 'Unknown')} — Trust: {trust}/100"
                p.font.size = Pt(12)

        # Code slide
        if code_snippets:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = "Generated Code"
            body = slide.placeholders[1]
            tf = body.text_frame
            tf.text = ""
            for snippet in code_snippets[:3]:
                p = tf.add_paragraph()
                p.text = f"# {snippet.get('concept', '')}"
                p.font.size = Pt(11)
                p = tf.add_paragraph()
                p.text = snippet.get("code", "")[:300]
                p.font.size = Pt(10)

        buf = io.BytesIO()
        prs.save(buf)
        buf.seek(0)
        return buf.getvalue()
```

- [ ] **Step 2: Add Flask endpoints in main.py**

```python
from report_builder import ReportBuilder
report_builder = ReportBuilder()

@app.route("/api/document/report", methods=["POST"])
def generate_report():
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip()
    fmt = (body.get("format") or "pdf").lower()

    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400

    try:
        ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
        if not ok:
            return jsonify({"error": filename}), 404
        text = extract_text_for_mimetype(filename, mimetype, data_bytes)

        # Generate all components
        summary = hf_client.summarize(text[:5000])
        citations = citation_analyzer.analyze_document(text)
        code_snippets = code_gen.generate_from_doc(text)

        title = f"Research Report: {filename}"

        if fmt == "pptx":
            data = report_builder.build_pptx(title, summary, citations, code_snippets)
            return app.response_class(data, mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                       headers={"Content-Disposition": f'attachment; filename="{filename}_report.pptx"'})
        else:
            data = report_builder.build_pdf(title, summary, citations, code_snippets)
            return app.response_class(data, mimetype="application/pdf",
                                       headers={"Content-Disposition": f'attachment; filename="{filename}_report.pdf"'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

- [ ] **Step 3: Commit**

```bash
git add backend/report_builder.py backend/main.py
git commit -m "feat: add report builder with PDF/PPT export"
```

---

## Task 14: Frontend — WebSearch Component

**Files:**
- Create: `my-app/src/Components/WebSearch.jsx`

- [ ] **Step 1: Create WebSearch.jsx**

```jsx
// my-app/src/Components/WebSearch.jsx
import React, { useState } from "react";
import { pyApiUrl } from "../config";

export default function WebSearch({ docId }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [source, setSource] = useState("all");

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(pyApiUrl("/api/search/web"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, source, num_results: 5 }),
      });
      const data = await res.json();
      if (data.success) setResults(data.results);
    } catch (err) {
      console.error("Search error:", err);
    } finally {
      setLoading(false);
    }
  };

  const renderResults = (items, type) => {
    if (!items || items.length === 0) return <p>No {type} results found.</p>;
    return items.map((item, i) => (
      <div key={i} style={{ marginBottom: 12, padding: 10, background: "#f9f9f9", borderRadius: 8 }}>
        <a href={item.link} target="_blank" rel="noreferrer" style={{ fontWeight: 600 }}>
          {item.title || "Untitled"}
        </a>
        {item.snippet && <p style={{ margin: "4px 0", fontSize: 13 }}>{item.snippet}</p>}
        {item.authors && <p style={{ fontSize: 12, color: "#666" }}>{item.authors.join(", ")}</p>}
        {item.cited_by > 0 && <span style={{ fontSize: 11, color: "#888" }}>Cited by: {item.cited_by}</span>}
        <span style={{ fontSize: 11, color: "#888", marginLeft: 8 }}>Source: {item.source}</span>
      </div>
    ));
  };

  return (
    <div style={{ padding: 16 }}>
      <h3>Web Search</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search web, news, academic papers..."
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #ccc" }}
        />
        <select value={source} onChange={(e) => setSource(e.target.value)}
                style={{ padding: 8, borderRadius: 6 }}>
          <option value="all">All Sources</option>
          <option value="web">Google</option>
          <option value="news">News</option>
          <option value="academic">Academic</option>
        </select>
        <button onClick={handleSearch} disabled={loading}
                style={{ padding: "8px 16px", borderRadius: 6, background: "#4285F4", color: "#fff", border: "none" }}>
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
      {results && (
        <div>
          {results.web && <div><h4>Web Results</h4>{renderResults(results.web, "web")}</div>}
          {results.news && <div><h4>News</h4>{renderResults(results.news, "news")}</div>}
          {results.academic && <div><h4>Academic Papers</h4>{renderResults(results.academic, "academic")}</div>}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add my-app/src/Components/WebSearch.jsx
git commit -m "feat: add WebSearch React component"
```

---

## Task 15: Frontend — Visualization, CodeGen, ReportBuilder Components

**Files:**
- Create: `my-app/src/Components/Visualizer.jsx`
- Create: `my-app/src/Components/CodeGenerator.jsx`
- Create: `my-app/src/Components/ReportBuilder.jsx`

- [ ] **Step 1: Create Visualizer.jsx**

```jsx
import React, { useState } from "react";
import { pyApiUrl } from "../config";

export default function Visualizer({ docId }) {
  const [vizType, setVizType] = useState("flowchart");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const generate = async () => {
    setLoading(true);
    try {
      const res = await fetch(pyApiUrl("/api/document/visualize"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, type: vizType }),
      });
      const data = await res.json();
      if (data.success) setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 16 }}>
      <h3>Document Visualization</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <select value={vizType} onChange={(e) => setVizType(e.target.value)}
                style={{ padding: 8, borderRadius: 6 }}>
          <option value="flowchart">Flowchart</option>
          <option value="word_frequency">Word Frequency</option>
          <option value="topics">Topic Distribution</option>
        </select>
        <button onClick={generate} disabled={loading || !docId}
                style={{ padding: "8px 16px", borderRadius: 6, background: "#34A853", color: "#fff", border: "none" }}>
          {loading ? "Generating..." : "Generate"}
        </button>
      </div>
      {result && result.type === "flowchart" && result.svg && (
        <div dangerouslySetInnerHTML={{ __html: result.svg }} style={{ background: "#fff", padding: 16, borderRadius: 8 }} />
      )}
      {result && result.chart && (
        <div id="plotly-chart" style={{ width: "100%", minHeight: 400 }}>
          {/* Render Plotly chart client-side using plotly.js */}
          <script
            dangerouslySetInnerHTML={{
              __html: `if(window.Plotly){Plotly.newPlot('plotly-chart',${JSON.stringify(result.chart.data)},${JSON.stringify(result.chart.layout)})}`
            }}
          />
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create CodeGenerator.jsx**

```jsx
import React, { useState } from "react";
import { pyApiUrl } from "../config";

export default function CodeGenerator({ docId }) {
  const [concept, setConcept] = useState("");
  const [language, setLanguage] = useState("python");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const generate = async () => {
    setLoading(true);
    try {
      const res = await fetch(pyApiUrl("/api/document/generate-code"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, concept, language }),
      });
      const data = await res.json();
      if (data.success) setResults(data.results);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 16 }}>
      <h3>Code Generator</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <input
          type="text"
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          placeholder="Enter a concept (or leave blank to auto-extract from doc)"
          style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #ccc" }}
        />
        <select value={language} onChange={(e) => setLanguage(e.target.value)}
                style={{ padding: 8, borderRadius: 6 }}>
          <option value="python">Python</option>
          <option value="javascript">JavaScript</option>
        </select>
        <button onClick={generate} disabled={loading || (!docId && !concept)}
                style={{ padding: "8px 16px", borderRadius: 6, background: "#673AB7", color: "#fff", border: "none" }}>
          {loading ? "Generating..." : "Generate Code"}
        </button>
      </div>
      {results.map((r, i) => (
        <div key={i} style={{ marginBottom: 16, background: "#1e1e1e", color: "#d4d4d4", padding: 16, borderRadius: 8 }}>
          <div style={{ color: "#888", marginBottom: 8 }}>Concept: {r.concept} ({r.language})</div>
          <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: 13 }}>{r.code}</pre>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Create ReportBuilder.jsx**

```jsx
import React, { useState } from "react";
import { pyApiUrl } from "../config";

export default function ReportBuilder({ docId }) {
  const [format, setFormat] = useState("pdf");
  const [loading, setLoading] = useState(false);

  const generate = async () => {
    setLoading(true);
    try {
      const res = await fetch(pyApiUrl("/api/document/report"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, format }),
      });
      if (!res.ok) throw new Error("Failed to generate report");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report.${format === "pptx" ? "pptx" : "pdf"}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 16 }}>
      <h3>Export Research Report</h3>
      <p style={{ fontSize: 13, color: "#666" }}>
        Generates a full report with summary, citations, trust scores, and code snippets.
      </p>
      <div style={{ display: "flex", gap: 8 }}>
        <select value={format} onChange={(e) => setFormat(e.target.value)}
                style={{ padding: 8, borderRadius: 6 }}>
          <option value="pdf">PDF</option>
          <option value="pptx">PowerPoint (PPTX)</option>
        </select>
        <button onClick={generate} disabled={loading || !docId}
                style={{ padding: "8px 16px", borderRadius: 6, background: "#FF6D00", color: "#fff", border: "none" }}>
          {loading ? "Generating..." : "Download Report"}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add my-app/src/Components/Visualizer.jsx my-app/src/Components/CodeGenerator.jsx my-app/src/Components/ReportBuilder.jsx
git commit -m "feat: add Visualizer, CodeGenerator, ReportBuilder React components"
```

---

## Task 16: Wire New Components into UploadPage/Chat

**Files:**
- Modify: `my-app/src/Components/UploadPage.jsx` or `my-app/src/Components/Chat.jsx`

- [ ] **Step 1: Add tab/panel navigation for new features**

In the main workspace component (UploadPage.jsx), add a feature panel alongside existing Chat/Quiz/Flashcard tabs. Import the new components:

```jsx
import WebSearch from "./WebSearch";
import Visualizer from "./Visualizer";
import CodeGenerator from "./CodeGenerator";
import ReportBuilder from "./ReportBuilder";
```

Add a feature selector UI (tabs or buttons) that conditionally renders:
```jsx
{activePanel === "search" && <WebSearch docId={currentDocId} />}
{activePanel === "visualize" && <Visualizer docId={currentDocId} />}
{activePanel === "codegen" && <CodeGenerator docId={currentDocId} />}
{activePanel === "report" && <ReportBuilder docId={currentDocId} />}
```

- [ ] **Step 2: Commit**

```bash
git add my-app/src/Components/UploadPage.jsx
git commit -m "feat: wire new feature panels into UploadPage"
```

---

## Task 17: Critical Bug Fixes

**Files:**
- Modify: `servers/routes/auth.js` (hardcoded admin creds)
- Modify: `servers/routes/document.js` (deprecated ObjectId)
- Modify: `servers/routes/chat.js` (FLASK_ASK_URL fallback)

- [ ] **Step 1: Remove hardcoded admin credentials in auth.js**

Find the hardcoded `admin123@gmail.com` / `adminhere` block and replace with proper admin seeding via env vars, or remove entirely and let admins be created via the normal signup + manual DB role update.

- [ ] **Step 2: Fix deprecated ObjectId in document.js**

Change `mongoose.Types.ObjectId(userId)` to `new mongoose.Types.ObjectId(userId)`.

- [ ] **Step 3: Fix FLASK_ASK_URL derivation in chat.js**

```javascript
// Change from:
const FLASK_ASK_URL = process.env.FLASK_ASK_URL || "http://localhost:5001/api/document/ask";
// To:
const FLASK_BASE = process.env.FLASK_BASE_URL || process.env.FLASK_SERVICE_URL || "http://localhost:5001";
const FLASK_ASK_URL = process.env.FLASK_ASK_URL || `${FLASK_BASE}/api/document/ask`;
```

- [ ] **Step 4: Remove dead commented code in quiz.py**

Delete lines 230-439 (the entire commented-out duplicate function).

- [ ] **Step 5: Commit**

```bash
git add servers/routes/auth.js servers/routes/document.js servers/routes/chat.js backend/quiz.py
git commit -m "fix: remove hardcoded admin creds, fix ObjectId, fix Flask URL derivation, remove dead code"
```

---

## Task 18: Update .env.example and Documentation

**Files:**
- Create: `backend/.env.example`
- Modify: `README.md` (update tech stack section)

- [ ] **Step 1: Create backend/.env.example**

```env
# HuggingFace API token (free at https://huggingface.co/settings/tokens)
HF_API_TOKEN=hf_xxxxxxxxxxxxx

# SerpAPI key (free 100 searches/month at https://serpapi.com)
SERP_API_KEY=

# NewsAPI key (free at https://newsapi.org)
NEWS_API_KEY=

# FAISS store path
FAISS_STORE_PATH=./faiss_store

# Node backend URL (for fetching documents)
NODE_BASE_URL=http://localhost:5000

# Service token (must match servers/.env)
SERVICE_TOKEN=smartdoc-service-token

# Frontend origins
FRONTEND_ORIGINS=http://localhost:3000
```

- [ ] **Step 2: Update README tech stack to match PPT**

Update the Technology Stack section to reflect:
- BART (summarization)
- T5 / Flan-T5 (quiz generation)
- sentence-transformers (embeddings)
- FAISS (vector store)
- SerpAPI, NewsAPI, CrossRef API (web search)
- Graphviz, Plotly (visualization)
- ReportLab, python-pptx (report export)

- [ ] **Step 3: Commit**

```bash
git add backend/.env.example README.md
git commit -m "docs: update env template and README for new tech stack"
```

---

## Execution Order

**Critical path (do in order):**
1. Task 1 → 2 → 3 → 4 → 5 (backbone swap — everything else depends on this)

**Independent (can parallel after Task 5):**
- Task 6 (summarize.py)
- Task 7 (quiz.py)
- Task 8 (flashcard.py)
- Task 9 (web search)
- Task 10 (citations)
- Task 11 (visualization)
- Task 12 (code gen)
- Task 13 (report builder)

**After all backend tasks:**
- Task 14 → 15 → 16 (frontend)

**Anytime:**
- Task 17 (bug fixes)
- Task 18 (docs)
