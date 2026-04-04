from flask import Blueprint, request, jsonify
import re

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
