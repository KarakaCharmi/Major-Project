import os, sys, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import FAISSStore


def test_add_and_search():
    tmp = tempfile.mkdtemp()
    try:
        store = FAISSStore(persist_dir=tmp, dimension=384)
        store.add_document("doc1", ["Hello world", "Python is great"], [[0.1]*384, [0.2]*384])
        results = store.search([0.1]*384, doc_id="doc1", top_k=2)
        assert len(results) == 2
        texts = [r["text"] for r in results]
        assert "Hello world" in texts
        assert "Python is great" in texts
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
        store2 = FAISSStore(persist_dir=tmp, dimension=384)
        assert store2.has_document("doc1")
        results = store2.search([0.4]*384, doc_id="doc1", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "persistent chunk"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
