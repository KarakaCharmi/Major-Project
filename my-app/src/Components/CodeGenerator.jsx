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
        body: JSON.stringify({ doc_id: docId, concept: concept.trim() || undefined, language }),
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
    <div>
      <h4 style={{ color: "#e0e0e0", margin: "0 0 10px" }}>Code Generator</h4>
      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        <input type="text" value={concept} onChange={(e) => setConcept(e.target.value)}
          placeholder="Concept (blank = auto-extract from doc)"
          style={{ flex: 1, padding: "7px 10px", borderRadius: 6, border: "1px solid #444", background: "#0f0f23", color: "#e0e0e0", fontSize: 13 }} />
        <select value={language} onChange={(e) => setLanguage(e.target.value)}
          style={{ padding: 7, borderRadius: 6, border: "1px solid #444", background: "#0f0f23", color: "#ccc", fontSize: 12 }}>
          <option value="python">Python</option>
          <option value="javascript">JavaScript</option>
        </select>
        <button onClick={generate} disabled={loading || (!docId && !concept.trim())}
          style={{ padding: "7px 16px", borderRadius: 6, background: "#673AB7", color: "#fff", border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600 }}>
          {loading ? "..." : "Generate"}
        </button>
      </div>
      {results.map((r, i) => (
        <div key={i} style={{ marginBottom: 12, background: "#0d1117", border: "1px solid #333", borderRadius: 8, overflow: "hidden" }}>
          <div style={{ padding: "6px 10px", background: "#161b22", color: "#8b949e", fontSize: 12, borderBottom: "1px solid #333" }}>
            {r.concept} ({r.language})
          </div>
          <pre style={{ margin: 0, padding: 10, whiteSpace: "pre-wrap", fontFamily: "'Fira Code', monospace", fontSize: 12, color: "#c9d1d9", maxHeight: 300, overflow: "auto" }}>{r.code}</pre>
        </div>
      ))}
    </div>
  );
}
