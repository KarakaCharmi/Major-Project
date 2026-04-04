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

    # Generate flashcards using Llama
    prompt = (
        f"Generate exactly {num_cards} study flashcards from the following text. "
        "Return ONLY valid JSON (no markdown, no explanation) with this schema:\n"
        '{"flashcards": [{"front": "term or question", "back": "answer or explanation", '
        '"category": "topic name", "difficulty": "Easy|Medium|Hard"}]}\n\n'
        f"Text:\n{context[:6000]}"
    )

    try:
        raw = hf_client.generate_flashcard_json(prompt, max_length=2048)
        data = _parse_json_safely(raw)

        if isinstance(data, dict) and isinstance(data.get("flashcards"), list):
            cards = _sanitize_cards(data["flashcards"], num_cards)
            if cards:
                return jsonify({"success": True, "flashcards": cards})

        # Fallback: extract key sentences as flashcards
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 20]
        cards = []
        for sent in sentences[:num_cards]:
            cards.append({
                "front": f"Explain: {sent[:100]}..." if len(sent) > 100 else f"Explain: {sent}",
                "back": sent[:400],
                "category": "General",
                "difficulty": "Medium",
            })
        if cards:
            return jsonify({"success": True, "flashcards": cards})

        return jsonify({"success": False, "error": "Could not generate flashcards."}), 502
    except Exception as e:
        return jsonify({"success": False, "error": f"Flashcard generation failed: {e}"}), 500


def _parse_json_safely(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", s, _re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass
        m2 = _re.search(r"\{[\s\S]*\}", s)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass
        return None


def _sanitize_cards(raw_cards, num_cards):
    cards = []
    for c in raw_cards[:num_cards]:
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
            "difficulty": str(c.get("difficulty", "Medium")).strip().capitalize()
                if str(c.get("difficulty", "Medium")).strip().capitalize() in ("Easy", "Medium", "Hard")
                else "Medium",
        })
    return cards
