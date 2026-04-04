from flask import Flask, request, jsonify, send_file
from werkzeug.exceptions import HTTPException
from flask_cors import CORS
import os, io, PyPDF2
from docx import Document as DocxDocument
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ====== NEW AI BACKBONE: BART/T5/FAISS ======
from models.hf_client import HFClient
from models.embedder import Embedder
from vector_store import FAISSStore

hf_client = HFClient()
embedder = Embedder()

FAISS_STORE_PATH = os.environ.get("FAISS_STORE_PATH", os.path.join(os.getcwd(), "faiss_store"))
vector_store = FAISSStore(persist_dir=FAISS_STORE_PATH, dimension=embedder.dimension)
# =============================================

from quiz import quiz_bp, init_quiz
from flashcard import flashcard_bp, init_flashcards
from summarize import init_summarizer, summarize_bp
from web_search import WebSearcher
from citation_analyzer import CitationAnalyzer
from visualizer import Visualizer
from code_generator import CodeGenerator
from report_builder import ReportBuilder
import requests
import tempfile, importlib
import hashlib
from better_profanity import profanity
import threading
import re

# Initialize new feature modules
web_searcher = WebSearcher()
citation_analyzer = CitationAnalyzer()
visualizer = Visualizer(hf_client)
code_gen = CodeGenerator(hf_client)
report_builder = ReportBuilder()

profanity.load_censor_words()

pdf_cache = {}

# ====== CONFIG ======
app = Flask(__name__)

_origins = os.environ.get("FRONTEND_ORIGINS", "http://localhost:3000")
try:
    _raw_allow = [o.strip() for o in _origins.split(",") if o.strip()]
except Exception:
    _raw_allow = ["http://localhost:3000"]

_allow_processed = []
for entry in _raw_allow:
    if entry.startswith("*."):

        import re as _re
        domain = _re.escape(entry[2:]) 
        _allow_processed.append(fr"https?://.*\.{domain}$")
    elif entry.startswith("http://*.") or entry.startswith("https://*."):

        import re as _re
        scheme, rest = entry.split("://", 1)
        domain = rest[2:]  
        domain_escaped = _re.escape(domain)
        _allow_processed.append(fr"{scheme}://.*\.{domain_escaped}$")
    else:
        _allow_processed.append(entry)

CORS(app, resources={r"/*": {"origins": _allow_processed}})

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

URL_REGEX = re.compile(r"(https?://[^\s]+|www\.[^\s]+|ftp://[^\s]+|mailto:[^\s]+|t\.me/[^\s]+|discord\.gg/[^\s]+)", re.IGNORECASE)

NODE_BASE_URL = os.environ.get("NODE_BASE_URL", "http://localhost:5000")
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "smartdoc-service-token")
NODE_FETCH_TIMEOUT = int(os.environ.get("NODE_FETCH_TIMEOUT", "45"))
CHUNK_UPSERT_URL = os.environ.get("CHUNK_UPSERT_URL", f"{NODE_BASE_URL}/api/search/internal/chunks/upsert")

def contains_link(text):
    return bool(URL_REGEX.search(text))


NOISE_DISTANCE_THRESHOLD = 0.6

# ====== GREETING/SMALL-TALK DETECTION & TOPIC SUGGESTIONS ======
GREET_WORDS = {
    "hi", "hello", "hey", "yo", "hola", "namaste",
    "good morning", "good afternoon", "good evening",
    "gm", "ge", "gn"
}
SMALL_TALK = {"how are you", "what's up", "sup", "howdy"}
WISHES = {"have a nice day", "good day", "good night"}

GENERIC_TOPICS = [
    "Introduction", "Overview", "Summary", "Background", "Objectives",
    "Methodology", "Approach", "Results", "Discussion", "Conclusion",
    "Features", "Requirements", "Limitations", "Future Work"
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def is_greeting_or_smalltalk(text: str) -> bool:
    s = _norm(text)
    if not s:
        return False
    if len(s) <= 40 and "?" not in s:
        for kw in list(GREET_WORDS) + list(SMALL_TALK) + list(WISHES):
            if s == kw or re.search(rf"(^|\b){re.escape(kw)}(\b|$)", s):
                return True
    for kw in GREET_WORDS:
        if re.search(rf"(^|\b){re.escape(kw)}(\b|$)", s):
            return True
    return False

def extract_headings_from_text(text: str, limit: int = 6) -> list:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    candidates = []
    seen = set()
    num_pat = re.compile(r"^\d+(?:\.\d+){0,3}\s+.{3,80}$")
    keyword_set = {k.lower() for k in GENERIC_TOPICS}
    for ln in lines:
        if not ln or len(ln) > 100:
            continue
        low = ln.lower()
        is_upperish = (ln == ln.upper() and 3 <= len(ln) <= 80)
        ends_colon = ln.endswith(":") and 3 <= len(ln) <= 80
        looks_numbered = bool(num_pat.match(ln))
        has_keyword = any(k in low for k in keyword_set)
        words = ln.split()
        short_title = 1 <= len(words) <= 8 and ln[0].isupper()
        if looks_numbered or is_upperish or ends_colon or has_keyword or short_title:
            key = low.strip(":")
            if key not in seen:
                seen.add(key)
                clean = ln.strip().rstrip(": .")
                candidates.append(clean)
                if len(candidates) >= limit:
                    break
    return candidates

def suggest_topics_for_doc(doc_id: str) -> list:
    st = consent_state.get(doc_id) or {}
    if st.get("sensitive") and not st.get("confirmed"):
        return GENERIC_TOPICS[:6]
    try:
        ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
        if not ok:
            return GENERIC_TOPICS[:6]
        text = extract_text_for_mimetype(filename or "document", mimetype or "", data_bytes or b"")
        heads = extract_headings_from_text(text, limit=6)
        return heads if heads else GENERIC_TOPICS[:6]
    except Exception:
        return GENERIC_TOPICS[:6]

# ====== SENSITIVE DATA DETECTION STATE & PATTERNS ======
# In-memory consent state per document. Persist in DB/Redis in production.
# { doc_id: { "sensitive": bool, "confirmed": bool, "awaiting": bool, "last_scan": str, "summary": dict } }
consent_state = {}
general_fallback = {}

SENSITIVE_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d{3}[\s-]?){2}\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

def detect_sensitive(text: str) -> dict:
    summary = {"found": False, "matches": {}}
    if not text:
        return summary
    any_found = False
    for name, pattern in SENSITIVE_PATTERNS.items():
        try:
            hits = pattern.findall(text)
            if hits:
                any_found = True
                summary["matches"][name] = len(hits)
        except Exception:
            continue
    summary["found"] = any_found
    print("[Sensitive Check] Summary:", {"found": summary["found"], "matches": summary["matches"]})
    return summary

# ====== HELPERS ======
def extract_text_from_pdf_bytes(data: bytes) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        for page in reader.pages:
            content = page.extract_text() or ""
            text += content + "\n"
    except Exception as e:
        print("PDF extraction error:", e)
    return text

def extract_text_from_docx_bytes(data: bytes) -> str:
    text = ""
    try:
        with io.BytesIO(data) as f:
            doc = DocxDocument(f)
            for p in doc.paragraphs:
                text += p.text + "\n"
    except Exception as e:
        print("DOCX extraction error:", e)
    return text

def extract_text_from_txt_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception as e:
        print("TXT extraction error:", e)
        return ""

def extract_text_for_mimetype(filename: str, mimetype: str, data: bytes) -> str:
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    if mimetype == "application/pdf" or ext == "pdf":
        return extract_text_from_pdf_bytes(data)
    elif mimetype in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword") or ext in ("docx", "doc"):
        return extract_text_from_docx_bytes(data)
    elif mimetype == "text/plain" or ext == "txt":
        return extract_text_from_txt_bytes(data)
    return ""

def chunk_text(text, size=1000, overlap=200):
    """Paragraph-aware chunking with overlap.
    - Prefer splitting on double newlines (paragraphs) to preserve context boundaries.
    - Then pack paragraphs into windows up to ~size characters with overlap between windows.
    """
    text = (text or "").strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text]

    windows = []
    buf = []
    cur_len = 0
    for p in paras:
        p_len = len(p) + 2  
        if cur_len + p_len <= size or not buf:
            buf.append(p)
            cur_len += p_len
        else:
            windows.append("\n\n".join(buf))
            join = "\n\n".join(buf)
            if overlap > 0 and len(join) > overlap:
                tail = join[-overlap:]
                buf = [tail, p]
                cur_len = len(tail) + p_len
            else:
                buf = [p]
                cur_len = p_len
    if buf:
        windows.append("\n\n".join(buf))
    return windows

def split_sheet_sections(text: str):
    """Split text into sections by lines that start with '# Sheet: <name>'.
    Returns list of tuples (sheet_name, content_str). If no markers found, returns [(None, text)].
    """
    lines = (text or "").splitlines()
    sections = []
    current_name = None
    current_lines = []
    found = False
    for ln in lines:
        if ln.startswith("# Sheet: "):
            found = True
            if current_lines:
                sections.append((current_name, "\n".join(current_lines).strip()))
                current_lines = []
            current_name = ln[len("# Sheet: "):].strip() or None
        else:
            current_lines.append(ln)
    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))
    if not found:
        return [(None, text or "")]
    return [(name, body) for (name, body) in sections if (body or "").strip()]

def generate_embeddings(text, timeout_sec: int = 20):
    """Generate embeddings using local sentence-transformers."""
    try:
        return embedder.embed(text)
    except Exception as e:
        print("Embedding error:", e)
        return None

# ====== ENDPOINTS ======

# ---- HEALTHCHECK ----
@app.route("/healthz", methods=["GET"]) 
def healthz():
    return jsonify({"status": "ok"})

# ---- ROOT ----
@app.route("/", methods=["GET", "HEAD"]) 
def root():
    return jsonify({"service": "SmartDocQ Flask", "status": "ok"})

# ---- INDEX FROM ATLAS (optional manual trigger) ----
@app.route("/api/index-from-atlas", methods=["POST"])
def index_from_atlas():
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("documentId") or body.get("doc_id") or "").strip()
    if not doc_id:
        return jsonify({"error": "Missing documentId"}), 400
    try:
        ok, filename, mimetype, data = fetch_doc_from_node(doc_id)
        if not ok:
            return jsonify({"error": filename}), 404

        text = extract_text_for_mimetype(filename, mimetype, data)
        if not text:
            return jsonify({"error": "Unsupported or empty document"}), 400

        # Pull persisted consent meta from Node
        meta = fetch_doc_meta_from_node(doc_id) or {}
        scan = detect_sensitive(text)
        prev = consent_state.get(doc_id) or {}
        consent_state[doc_id] = {
            "sensitive": bool(scan.get("found")),
            "confirmed": bool(meta.get("consentConfirmed") or prev.get("confirmed", False)),
            "awaiting": False,
            "last_scan": "ok",
            "summary": scan,
        }
        if scan.get("found") and not (meta.get("consentConfirmed") or prev.get("confirmed", False)):
            return jsonify({
                "message": "Sensitive data detected; indexing deferred until consent.",
                "requireConfirmation": True,
                "sensitiveSummary": scan,
                "doc_id": doc_id,
            }), 200

        indexed, added = index_bytes(doc_id, filename, mimetype, data)
        if not indexed:
            return jsonify({"error": "Unsupported or empty document"}), 400
        return jsonify({"message": f"Indexed {added} chunks", "doc_id": doc_id, "requireConfirmation": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Word to PDF Conversion Endpoint ----
@app.route("/api/convert/word-to-pdf", methods=["POST"])
def convert_word_to_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        data = file.read()
        filename = file.filename or "document"
        
        ext = (filename.rsplit('.',1)[-1].lower() if '.' in filename else '')
        content_type = file.content_type or ''
        
        if content_type not in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") and ext not in ("doc", "docx"):
            return jsonify({"error": "Not a Word document"}), 415
        

        try:
            docx2pdf = importlib.import_module("docx2pdf")
        except Exception:
            return jsonify({"error": "docx2pdf not installed on server"}), 501
        

        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, filename)

            if not in_path.lower().endswith((".docx", ".doc")):
                in_path += ".docx"
            

            with open(in_path, 'wb') as f:
                f.write(data)
            
            temp_pdf = os.path.join(td, "converted.pdf")
            

            docx2pdf.convert(in_path, temp_pdf)
          
            with open(temp_pdf, 'rb') as f:
                pdf_data = f.read()
            
  
            response = app.response_class(
                pdf_data,
                mimetype='application/pdf',
                headers={'Content-Disposition': f'attachment; filename="{filename.rsplit(".", 1)[0]}.pdf"'}
            )
            return response
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Word Preview as PDF ----
@app.route("/api/document/preview/<doc_id>.pdf", methods=["GET"])
def preview_word_as_pdf(doc_id):
    try:
      
        cache_key = f"pdf_preview_{doc_id}"
        if cache_key in pdf_cache:
            cached_path = pdf_cache[cache_key]
            if os.path.exists(cached_path):
                return send_file(cached_path, mimetype="application/pdf", as_attachment=False, download_name="preview.pdf")
            else:
            
                del pdf_cache[cache_key]
        
        ok, filename, mimetype, data = fetch_doc_from_node(doc_id)
        if not ok:
            return jsonify({"error": filename}), 404
  
        ext = (filename.rsplit('.',1)[-1].lower() if '.' in filename else '')
        if mimetype not in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") and ext not in ("doc", "docx"):
            return jsonify({"error": "Not a Word document"}), 415
        
   
        try:
            docx2pdf = importlib.import_module("docx2pdf")
        except Exception:
            return jsonify({"error": "docx2pdf not installed on server"}), 501
        
     
        cache_dir = os.path.join(os.getcwd(), "pdf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
       
        content_hash = hashlib.md5(data).hexdigest()[:8]
        cached_pdf = os.path.join(cache_dir, f"{doc_id}_{content_hash}.pdf")
      
        if os.path.exists(cached_pdf):
            pdf_cache[cache_key] = cached_pdf
            return send_file(cached_pdf, mimetype="application/pdf", as_attachment=False, download_name="preview.pdf")
        
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, filename)

            if not in_path.lower().endswith((".docx", ".doc")):
                in_path += ".docx"
            with open(in_path, 'wb') as f:
                f.write(data)
            temp_pdf = os.path.join(td, "preview.pdf")

            docx2pdf.convert(in_path, temp_pdf)
            
            import shutil
            shutil.copy2(temp_pdf, cached_pdf)
            pdf_cache[cache_key] = cached_pdf
            
            return send_file(cached_pdf, mimetype="application/pdf", as_attachment=False, download_name="preview.pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- MY DOCS ----
@app.route("/api/document/my", methods=["GET"])
def list_docs():
    docs = {}
    for doc_id in vector_store._docs:
        chunks = vector_store.get_chunks(doc_id)
        if chunks:
            docs[doc_id] = {"_id": doc_id, "name": doc_id, "type": "document", "size": len(chunks)}
    return jsonify(list(docs.values()))

# ---- RENAME ----
@app.route("/api/document/<doc_id>", methods=["PUT"])
def rename_doc(doc_id):
    data = request.get_json(silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "Missing new name"}), 400
    # Rename is handled by Node; FAISS doesn't track filenames
    return jsonify({"message": "Renamed successfully"})

# ---- DELETE ----
@app.route("/api/document/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    vector_store.delete_document(doc_id)
    return jsonify({"message": "Deleted successfully"})

# ---- ASK ----
@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), getattr(e, "code", 500)
    print("Unhandled Exception:", e)
    return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def handle_request_entity_too_large(e):
    return jsonify({"error": "File too large. Max 25 MB."}), 413

# ---- ASK (Chat-ready) ----
@app.route("/api/document/ask", methods=["POST"])
def ask_doc():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    doc_id = data.get("doc_id", "").strip()

    if not question:
        return jsonify({"error": "Missing question"}), 400

    if is_greeting_or_smalltalk(question):
        topics = suggest_topics_for_doc(doc_id) if doc_id else GENERIC_TOPICS[:6]

        bullet = "\n".join(f"- {t}" for t in topics)
        msg = (
            "Hello! I'm here to help you with your document. You can ask questions about the following sections/topics in your document:\n"
            f"{bullet}\n\n"
            "Please type a question related to one of these topics."
        )
        return jsonify({"answer": msg, "requireConfirmation": False})
    
    if URL_REGEX.search(question):
        return jsonify({"answer": "Warning: No links allowed. Please ask using text only."}), 422
    if profanity.contains_profanity(question):
        return jsonify({"answer": "Warning: Please avoid using offensive words."}), 422
    
    if not doc_id:
        return jsonify({"error": "Missing doc_id"}), 400

    try:

        state = consent_state.get(doc_id) or {"sensitive": False, "confirmed": False, "awaiting": False}
        if state.get("sensitive") and not state.get("confirmed"):
            q_lower = question.lower().strip()
            if q_lower in ("y", "yes"):
                state["confirmed"] = True
                state["awaiting"] = False
                consent_state[doc_id] = state
                return jsonify({
                    "answer": "Proceeding. You can now ask questions about this document.",
                    "requireConfirmation": False
                })
            if q_lower in ("n", "no"):
                state["awaiting"] = False
                consent_state[doc_id] = state
                return jsonify({
                    "answer": "Chat cancelled. Please re-upload a cleaned version of the document without sensitive data.",
                    "requireConfirmation": False
                })

            state["awaiting"] = True
            consent_state[doc_id] = state
            return jsonify({
                "answer": "Warning: Sensitive or private information detected in this document (e.g., personal IDs, contact info, or financial data).\nDo you still want to proceed with chatting about it? (y/n)",
                "requireConfirmation": True,
                "sensitiveSummary": state.get("summary", {})
            })


        gf = general_fallback.get(doc_id) or {"awaiting": False}
        if gf.get("awaiting"):
            q_lower = question.lower().strip()
            if q_lower in ("y", "yes"):
         
                orig_q = gf.get("pending_question") or ""
              
                general_fallback[doc_id] = {"awaiting": False}

                if not orig_q:
                    return jsonify({
                        "answer": "Okay, please ask your question again.",
                    })
       
                try:
                    answer = hf_client.generate_text(f"Answer this question clearly and accurately: {orig_q}", max_length=300)
                    if answer:
                        return jsonify({"answer": format_response(answer.strip())})
                    else:
                        return jsonify({"answer": "Could not generate a general answer."})
                except Exception as e:
                    print("General fallback error:", e)
                    return jsonify({"answer": "Error generating a general answer. Please try again."})

            if q_lower in ("n", "no"):
      
                general_fallback[doc_id] = {"awaiting": False}
                return jsonify({
                    "answer": "Okay, I won't answer that. Please ask a question based on the uploaded document.",
                })

      
            return jsonify({
                "answer": "I couldn't find relevant information about your question in the uploaded document.\nDo you want me to answer using general knowledge instead? Reply \"y\" for yes or \"n\" for no.",
            })


        if not has_index(doc_id):

            _start_background_indexing(doc_id)
            return jsonify({
                "answer": "Indexing this document in the background. Please try your question again in ~30-60 seconds.",
                "requireConfirmation": False
            })


        q_emb = generate_embeddings(question)
        if not q_emb:
            return jsonify({"error": "Failed to generate embedding"}), 500

        # Search FAISS for relevant chunks
        results = vector_store.search(q_emb, doc_id=doc_id, top_k=5)

        if not results or all(r["score"] < 0.2 for r in results):
            general_fallback[doc_id] = {
                "awaiting": True,
                "pending_question": question,
            }
            return jsonify({
                "answer": (
                    "I couldn't find relevant information about your question in the uploaded document.\n"
                    "Do you want me to answer using general knowledge instead? Reply \"y\" for yes or \"n\" for no."
                )
            })

        context = "\n\n".join([r["text"] for r in results])

        # Use HF client (extractive QA + Llama fallback) to answer
        raw_text = hf_client.answer_question(question, context)

        if not raw_text or not raw_text.strip():
            return jsonify({"answer": "Could not generate an answer. Please try rephrasing."})

        raw_text = raw_text.strip()

        if is_out_of_doc_answer(raw_text):
            general_fallback[doc_id] = {
                "awaiting": True,
                "pending_question": question,
            }
            appended = (
                raw_text
                + "\n\nDo you want me to answer using general knowledge instead? Reply \"y\" for yes or \"n\" for no."
            )
            return jsonify({"answer": format_response(appended), "requireConfirmation": False})

        answer_text = format_response(raw_text)
        return jsonify({"answer": answer_text, "requireConfirmation": False})

    except Exception as e:
        print("Ask error:", e)
        return jsonify({"error": str(e)}), 500

def format_response(text):
    """
    Improve the formatting of AI responses for better readability
    """
    import re
    
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  
    text = re.sub(r'[ \t]+', ' ', text)  
    

    text = re.sub(r'(\w)\. ([A-Z])', r'\1.\n\n\2', text)
    

    text = re.sub(r'^\s*[-•*]\s*', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(\d+)\.\s*', r'\1. ', text, flags=re.MULTILINE)
    

    text = re.sub(r'([.!?])\s*(•|\d+\.)', r'\1\n\n\2', text)
    

    text = re.sub(r':(\s*)(•|\d+\.)', r':\n\n\1\2', text)
    

    text = re.sub(r'([^:\n]):\s*\n', r'\1:\n\n', text)
    

    text = text.strip()
    return text

def is_out_of_doc_answer(text: str) -> bool:
    """Heuristically detect when the LLM indicates the answer isn't in the provided context/document.
    This helps us surface the y/n general-knowledge prompt even when reranked context existed but lacked the answer.
    """
    low = (text or "").strip().lower()
    if not low:
        return False
    patterns = [
        "not in the context",
        "context provided does not contain",
        "provided context does not contain",
        "does not contain information",
        "doesn't contain information",
        "i couldn't find",
        "i could not find",
        "couldn't find in your document",
        "could not find in your document",
        "not found in your document",
        "not found in the document",
        "no relevant information",
        "not available in the document",
        "outside the document",
        "outside of the document",
        "not present in the document",
    ]
    return any(p in low for p in patterns)

def fetch_doc_from_node(doc_id: str):
    """Fetch binary document from Node API /api/document/:id/download (requires user token in frontend).
    For server-side, assume Node allows local trusted call without auth or you can add a service token.
    """
    try:
        url = f"{NODE_BASE_URL}/api/document/{doc_id}/download"

        headers = {"x-service-token": SERVICE_TOKEN}
        r = requests.get(url, headers=headers, timeout=NODE_FETCH_TIMEOUT)
        if r.status_code != 200:
            return False, f"Node returned {r.status_code}", None, None
   
        disp = r.headers.get("Content-Disposition", "")
        filename = "document"
        if "filename=" in disp:
            filename = disp.split("filename=")[-1].strip('"')
        mimetype = r.headers.get("Content-Type", "application/octet-stream")
        return True, filename, mimetype, r.content
    except Exception as e:
        return False, str(e), None, None

def fetch_doc_meta_from_node(doc_id: str):
    """Fetch document metadata (sensitiveFound/consentConfirmed) from Node for consent persistence."""
    try:
        url = f"{NODE_BASE_URL}/api/document/{doc_id}/_meta"
        headers = {"x-service-token": SERVICE_TOKEN}
        r = requests.get(url, headers=headers, timeout=NODE_FETCH_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# --- Background indexing support ---
_indexing_in_progress = set()
_indexing_lock = threading.Lock()

def _background_index(doc_id: str):
    try:
        ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
        if not ok:
            return

        text_for_scan = extract_text_for_mimetype(filename, mimetype, data_bytes)
        if not text_for_scan:
            return
        scan = detect_sensitive(text_for_scan)
        prev = consent_state.get(doc_id) or {}
        consent_state[doc_id] = {
            "sensitive": bool(scan.get("found")),
            "confirmed": bool(prev.get("confirmed", False)),
            "awaiting": False,
            "last_scan": "ok",
            "summary": scan,
        }
        if scan.get("found") and not prev.get("confirmed", False):
            return
        index_bytes(doc_id, filename, mimetype, data_bytes)
    finally:
        with _indexing_lock:
            _indexing_in_progress.discard(doc_id)

def _start_background_indexing(doc_id: str):
    with _indexing_lock:
        if doc_id in _indexing_in_progress:
            return
        _indexing_in_progress.add(doc_id)
    th = threading.Thread(target=_background_index, args=(doc_id,), daemon=True)
    th.start()

def has_index(doc_id: str) -> bool:
    return vector_store.has_document(doc_id)

def index_bytes(doc_id: str, filename: str, mimetype: str, data: bytes):
    """Extract text, chunk, embed, store in FAISS."""
    text = extract_text_for_mimetype(filename, mimetype, data)
    text = (text or "").strip()
    if not text:
        return False, 0

    sections = split_sheet_sections(text)
    all_chunks = []
    chunk_records = []
    chunk_index = 0

    for (sheet_name, body) in sections:
        chunks = chunk_text(body)
        for chunk in chunks:
            c = (chunk or "").strip()
            if not c:
                continue
            all_chunks.append(c)
            chunk_records.append({"chunk": chunk_index, "sheet": sheet_name or None, "text": c})
            chunk_index += 1

    if not all_chunks:
        return False, 0

    embeddings = embedder.embed_batch(all_chunks)
    vector_store.add_document(doc_id, all_chunks, embeddings)
    _push_chunks_to_node(doc_id, filename, chunk_records)
    return True, len(all_chunks)

def index_text(doc_id: str, filename: str, text: str):
    """Index plain text content for a document, replacing existing chunks."""
    text = (text or "").strip()
    if not text:
        return False, 0

    sections = split_sheet_sections(text)
    all_chunks = []
    chunk_records = []
    chunk_index = 0

    for (sheet_name, body) in sections:
        chunks = chunk_text(body)
        for chunk in chunks:
            c = (chunk or "").strip()
            if not c:
                continue
            all_chunks.append(c)
            chunk_records.append({"chunk": chunk_index, "sheet": sheet_name or None, "text": c})
            chunk_index += 1

    if not all_chunks:
        return False, 0

    embeddings = embedder.embed_batch(all_chunks)
    vector_store.add_document(doc_id, all_chunks, embeddings)
    _push_chunks_to_node(doc_id, filename or "document.txt", chunk_records)
    return True, len(all_chunks)

def _push_chunks_to_node(doc_id: str, filename: str, chunk_records: list):
    """Best-effort push of chunk texts to Node for keyword/metadata search. Non-fatal on error."""
    try:
        if not chunk_records:
            return
        payload = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": chunk_records,
        }
        headers = {"Content-Type": "application/json", "x-service-token": SERVICE_TOKEN}
        r = requests.post(CHUNK_UPSERT_URL, json=payload, headers=headers, timeout=NODE_FETCH_TIMEOUT)
        if r.status_code >= 300:
            print("[Chunks Upsert] Node returned", r.status_code, r.text[:200])
    except Exception as e:
        print("[Chunks Upsert] Error:", e)

# Endpoint to record user consent and optionally trigger indexing
@app.route("/api/document/consent", methods=["POST"])
def set_consent():
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or body.get("documentId") or "").strip()
    consent = bool(body.get("consent", False))
    if not doc_id:
        return jsonify({"error": "Missing doc_id"}), 400
    st = consent_state.get(doc_id) or {"sensitive": False, "confirmed": False}
    st["confirmed"] = consent
    st["awaiting"] = False
    consent_state[doc_id] = st
    # Persist consent to Node for durability (best-effort)
    try:
        requests.post(f"{NODE_BASE_URL}/api/document/{doc_id}/consent",
                      json={"consent": consent},
                      headers={"x-service-token": SERVICE_TOKEN},
                      timeout=NODE_FETCH_TIMEOUT)
    except Exception:
        pass

    if consent and not has_index(doc_id):
        ok, filename, mimetype, data_bytes = fetch_doc_from_node(doc_id)
        if not ok:
            return jsonify({"error": filename}), 404
        indexed, added = index_bytes(doc_id, filename, mimetype, data_bytes)
        if not indexed:
            return jsonify({"error": "Unsupported or empty document"}), 400
        return jsonify({"message": f"Consent recorded. Indexed {added} chunks.", "requireConfirmation": False})

    if not consent:
        return jsonify({"message": "Consent declined. Please upload a cleaned document.", "requireConfirmation": False})

    return jsonify({"message": "Consent recorded.", "requireConfirmation": False})

@app.route("/api/index/replace-text", methods=["POST"])
def replace_text_index():
    """
    Replace the indexed content for a document with the provided plain text, without re-uploading the file.
    Request JSON:
      { "doc_id"|"documentId": str, "text": str, "filename"?: str }
    Response JSON mirrors /api/index-from-atlas with requireConfirmation handling.
    """
    body = request.get_json(silent=True) or {}
    doc_id = (body.get("documentId") or body.get("doc_id") or "").strip()
    text = body.get("text")
    filename = (body.get("filename") or "document.txt").strip()

    if not doc_id:
        return jsonify({"error": "Missing documentId"}), 400
    if text is None:
        return jsonify({"error": "Missing text"}), 400

    try:

        scan = detect_sensitive(text or "")
        prev = consent_state.get(doc_id) or {}
        consent_state[doc_id] = {
            "sensitive": bool(scan.get("found")),
            "confirmed": bool(prev.get("confirmed", False)),
            "awaiting": False,
            "last_scan": "ok",
            "summary": scan,
        }
        if scan.get("found") and not prev.get("confirmed", False):
            return jsonify({
                "message": "Sensitive data detected; indexing deferred until consent.",
                "requireConfirmation": True,
                "sensitiveSummary": scan,
                "doc_id": doc_id,
            }), 200

        indexed, added = index_text(doc_id, filename, text)
        if not indexed:
            return jsonify({"error": "Empty text or indexing failed"}), 400
        return jsonify({"message": f"Indexed {added} chunks", "doc_id": doc_id, "requireConfirmation": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====== NEW FEATURE ENDPOINTS ======

@app.route("/api/search/web", methods=["POST"])
def web_search():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400
    source = body.get("source", "all")
    num = min(int(body.get("num_results", 5)), 10)
    if source == "web":
        results = {"web": web_searcher.search_web(query, num)}
    elif source == "academic":
        results = {"academic": web_searcher.search_crossref(query, num)}
    else:
        results = web_searcher.search_all(query, num)
    return jsonify({"success": True, "results": results})


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
        summary = hf_client.summarize(text[:5000])
        citations = citation_analyzer.analyze_document(text)
        code_snippets = code_gen.generate_from_doc(text)
        title = f"Research Report: {filename}"
        if fmt == "pptx":
            data = report_builder.build_pptx(title, summary, citations, code_snippets)
            return app.response_class(data,
                mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                headers={"Content-Disposition": f'attachment; filename="{filename}_report.pptx"'})
        else:
            data = report_builder.build_pdf(title, summary, citations, code_snippets)
            return app.response_class(data, mimetype="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}_report.pdf"'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====== REGISTER BLUEPRINTS ======

try:
    init_quiz(vector_store, has_index, fetch_doc_from_node, extract_text_for_mimetype, hf_client)
    app.register_blueprint(quiz_bp)
except Exception as _e:
    print(f"[Init] Quiz blueprint failed: {_e}")

try:
    init_flashcards(vector_store, has_index, fetch_doc_from_node, extract_text_for_mimetype, hf_client)
    app.register_blueprint(flashcard_bp)
except Exception as _e:
    print(f"[Init] Flashcard blueprint failed: {_e}")

try:
    init_summarizer(hf_client)
    app.register_blueprint(summarize_bp)
except Exception as _e:
    print(f"[Init] Summarizer blueprint failed: {_e}")

# ====== RUN SERVER ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)