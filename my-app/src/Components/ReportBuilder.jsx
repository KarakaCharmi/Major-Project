import React, { useState } from "react";
import { pyApiUrl } from "../config";

export default function ReportBuilder({ docId }) {
  const [format, setFormat] = useState("pdf");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generate = async () => {
    if (!docId) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(pyApiUrl("/api/document/report"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, format }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Failed to generate report");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report.${format === "pptx" ? "pptx" : "pdf"}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message || "Failed to generate report");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h4 style={{ color: "#e0e0e0", margin: "0 0 8px" }}>Export Research Report</h4>
      <p style={{ fontSize: 12, color: "#888", margin: "0 0 12px" }}>
        Generates a full report with summary, citations, trust scores, and code snippets.
      </p>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <select value={format} onChange={(e) => setFormat(e.target.value)}
          style={{ padding: 7, borderRadius: 6, border: "1px solid #444", background: "#0f0f23", color: "#ccc", fontSize: 12 }}>
          <option value="pdf">PDF</option>
          <option value="pptx">PowerPoint (PPTX)</option>
        </select>
        <button onClick={generate} disabled={loading || !docId}
          style={{ padding: "7px 16px", borderRadius: 6, background: "#FF6D00", color: "#fff", border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600 }}>
          {loading ? "Generating..." : "Download Report"}
        </button>
      </div>
      {error && <p style={{ color: "#ff6b6b", marginTop: 8, fontSize: 13 }}>{error}</p>}
    </div>
  );
}
