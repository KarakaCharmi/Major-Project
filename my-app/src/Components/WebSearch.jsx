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
    if (!items || items.length === 0) return <p style={{ color: "#888", fontSize: 13 }}>No {type} results found.</p>;
    return items.map((item, i) => (
      <div key={i} style={{ marginBottom: 10, padding: 10, background: "#16213e", borderRadius: 8, border: "1px solid #333" }}>
        <a href={item.link} target="_blank" rel="noreferrer" style={{ fontWeight: 600, color: "#64b5f6", textDecoration: "none", fontSize: 14 }}>
          {item.title || "Untitled"}
        </a>
        {item.snippet && <p style={{ margin: "4px 0 0", fontSize: 12, color: "#aaa" }}>{item.snippet}</p>}
        {item.authors && item.authors.length > 0 && <p style={{ fontSize: 11, color: "#888", margin: "2px 0 0" }}>{item.authors.join(", ")}</p>}
        <div style={{ display: "flex", gap: 12, fontSize: 11, color: "#666", marginTop: 4 }}>
          {item.cited_by > 0 && <span>Cited by: {item.cited_by}</span>}
          <span>Source: {item.source}</span>
          {item.published && <span>Year: {item.published}</span>}
        </div>
      </div>
    ));
  };

  return (
    <div>
      <h4 style={{ color: "#e0e0e0", margin: "0 0 10px" }}>Web Search</h4>
      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        <input type="text" value={query} onChange={(e) => setQuery(e.target.value)}
          placeholder="Search web, academic papers..."
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          style={{ flex: 1, padding: "7px 10px", borderRadius: 6, border: "1px solid #444", background: "#0f0f23", color: "#e0e0e0", fontSize: 13 }} />
        <select value={source} onChange={(e) => setSource(e.target.value)}
          style={{ padding: 7, borderRadius: 6, border: "1px solid #444", background: "#0f0f23", color: "#ccc", fontSize: 12 }}>
          <option value="all">All</option>
          <option value="web">Google</option>
          <option value="academic">Academic</option>
        </select>
        <button onClick={handleSearch} disabled={loading}
          style={{ padding: "7px 16px", borderRadius: 6, background: "#4285F4", color: "#fff", border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600 }}>
          {loading ? "..." : "Search"}
        </button>
      </div>
      {results && (
        <div>
          {results.web && results.web.length > 0 && <div><h5 style={{ color: "#aaa", margin: "8px 0 6px" }}>Web Results</h5>{renderResults(results.web, "web")}</div>}
          {results.academic && results.academic.length > 0 && <div><h5 style={{ color: "#aaa", margin: "8px 0 6px" }}>Academic Papers</h5>{renderResults(results.academic, "academic")}</div>}
        </div>
      )}
    </div>
  );
}
