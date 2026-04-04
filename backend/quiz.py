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

    # Generate quiz using Llama via HF client
    prompt = (
        f"Generate exactly {num_questions} {difficulty} quiz questions from the following text. "
        f"Include these question types: {', '.join(qtypes)}. "
        "Return ONLY valid JSON with this exact schema (no markdown, no explanation):\n"
        '{"questions": [{"type": "mcq|true_false|short_answer", "question": "...", '
        '"options": ["..."] (for mcq only, 4 options), "correct_answer": "...", "explanation": "..."}]}\n\n'
        f"Text:\n{context[:6000]}"
    )

    try:
        raw = hf_client.generate_quiz_json(prompt, max_length=2048)
        quiz = _parse_json_safely(raw)

        if isinstance(quiz, dict) and isinstance(quiz.get("questions"), list):
            qs = _sanitize_questions(quiz["questions"], num_questions, qtypes)
            if qs:
                return jsonify({"success": True, "quiz": {"questions": qs}})

        # Fallback: try once more with a simpler prompt
        simple_prompt = (
            f"Create {num_questions} quiz questions about this text. Mix of multiple choice and true/false. "
            "Return valid JSON: {\"questions\": [{\"type\": \"mcq\", \"question\": \"...\", "
            "\"options\": [\"A\",\"B\",\"C\",\"D\"], \"correct_answer\": \"...\", \"explanation\": \"...\"}]}\n\n"
            f"Text:\n{context[:4000]}"
        )
        raw2 = hf_client.generate_quiz_json(simple_prompt, max_length=2048)
        quiz2 = _parse_json_safely(raw2)

        if isinstance(quiz2, dict) and isinstance(quiz2.get("questions"), list):
            qs2 = _sanitize_questions(quiz2["questions"], num_questions, qtypes)
            if qs2:
                return jsonify({"success": True, "quiz": {"questions": qs2}})

        return jsonify({"success": False, "error": "Could not generate quiz. Please try again."}), 502
    except Exception as e:
        return jsonify({"success": False, "error": f"Quiz generation failed: {e}"}), 500


def _parse_json_safely(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        # Try extracting JSON from markdown fences
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", s, _re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass
        # Try finding a JSON object
        m2 = _re.search(r"\{[\s\S]*\}", s)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass
        return None


def _sanitize_questions(raw_questions, num_questions, qtypes):
    qs = []
    for q in raw_questions[:num_questions]:
        if not isinstance(q, dict):
            continue
        qtype = str(q.get("type", "")).strip().lower()
        if qtype not in ("mcq", "true_false", "short_answer"):
            qtype = "mcq" if q.get("options") else "short_answer"
        question = str(q.get("question", "")).strip()
        if not question:
            continue
        correct = str(q.get("correct_answer", "")).strip()

        if qtype == "true_false":
            correct = correct.lower()
            if correct not in ("true", "false"):
                correct = "true" if correct in ("t", "yes", "y", "1") else "false"

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
            if len(opts) < 3:
                opts = ["Option A", "Option B", correct or "Option C"]
            seen = set()
            dedup = []
            for o in opts:
                if o not in seen:
                    dedup.append(o)
                    seen.add(o)
            item["options"] = dedup[:5]
        qs.append(item)
    return qs
