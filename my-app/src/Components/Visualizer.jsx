import React, { useState } from "react";
import { pyApiUrl } from "../config";

export default function Visualizer({ docId }) {
  const [vizType, setVizType] = useState("flowchart");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generate = async () => {
    if (!docId) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(pyApiUrl("/api/document/visualize"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, type: vizType }),
      });
      const data = await res.json();
      if (data.success) setResult(data);
      else setError(data.error || "Failed to generate visualization");
    } catch (err) {
      setError("Connection error. Is the Flask server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h4 style={{ color: "#e0e0e0", margin: "0 0 10px" }}>Document Visualization</h4>
      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        <select value={vizType} onChange={(e) => setVizType(e.target.value)}
          style={{ padding: 7, borderRadius: 6, border: "1px solid #444", background: "#0f0f23", color: "#ccc", fontSize: 12 }}>
          <option value="flowchart">Flowchart</option>
          <option value="word_frequency">Word Frequency</option>
          <option value="topics">Topic Distribution</option>
        </select>
        <button onClick={generate} disabled={loading || !docId}
          style={{ padding: "7px 16px", borderRadius: 6, background: "#34A853", color: "#fff", border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600 }}>
          {loading ? "Generating..." : "Generate"}
        </button>
      </div>
      {error && <p style={{ color: "#ff6b6b", fontSize: 13 }}>{error}</p>}
      {result && result.type === "flowchart" && result.svg && (
        <div dangerouslySetInnerHTML={{ __html: result.svg }}
          style={{ background: "#fff", padding: 12, borderRadius: 8, overflow: "auto", maxHeight: "40vh" }} />
      )}
      {result && result.chart && (
        <div style={{ background: "#16213e", padding: 12, borderRadius: 8 }}>
          <pre style={{ fontSize: 11, color: "#aaa", maxHeight: 250, overflow: "auto", margin: 0 }}>
            {JSON.stringify(result.chart.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
