"""
Citation analysis: extract references from text, fetch metadata via CrossRef,
compute trust scores based on citation count, recency, and source authority.
"""
import re
import requests


class CitationAnalyzer:
    """Extract and analyze citations from document text."""

    REF_PATTERNS = [
        re.compile(r"\[(\d+)\]\s*(.+?)(?:\n|$)"),
        re.compile(r"(\d+)\.\s+([A-Z][^.]+\.\s+.+?)(?:\n|$)"),
        re.compile(r"(?:doi|DOI):\s*(10\.\d{4,}/[^\s]+)"),
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
        score = 30  # base

        cited = metadata.get("cited_by", 0)
        if cited > 100:
            score += 30
        elif cited > 50:
            score += 25
        elif cited > 10:
            score += 15
        elif cited > 0:
            score += 5

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
        for ref_text in refs[:10]:
            meta = self.fetch_metadata(ref_text)
            trust = self.compute_trust_score(meta)
            analyzed.append({
                "original_text": ref_text,
                "metadata": meta,
                "trust_score": trust,
            })
        return analyzed
