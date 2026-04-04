"""
Multi-source web search: SerpAPI (Google), CrossRef (academic).
SerpAPI requires API key. CrossRef is free (no key needed).
"""
import os
import requests

SERP_API_KEY = os.environ.get("SERP_API_KEY", "")


class WebSearcher:
    """Search web and academic sources."""

    def search_web(self, query, num_results=5):
        """Search Google via SerpAPI."""
        if not SERP_API_KEY:
            return []
        try:
            resp = requests.get("https://serpapi.com/search", params={
                "q": query,
                "api_key": SERP_API_KEY,
                "num": num_results,
                "engine": "google",
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google",
                })
            return results
        except Exception as e:
            print(f"[SerpAPI] Error: {e}")
            return []

    def search_crossref(self, query, num_results=5):
        """Search academic papers via CrossRef (free, no key needed)."""
        try:
            resp = requests.get("https://api.crossref.org/works", params={
                "query": query,
                "rows": num_results,
                "sort": "relevance",
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("message", {}).get("items", [])[:num_results]:
                title_parts = item.get("title", [""])
                title = title_parts[0] if title_parts else ""
                doi = item.get("DOI", "")
                results.append({
                    "title": title,
                    "link": f"https://doi.org/{doi}" if doi else "",
                    "doi": doi,
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])[:3]
                    ],
                    "published": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]),
                    "source": "crossref",
                    "cited_by": item.get("is-referenced-by-count", 0),
                })
            return results
        except Exception as e:
            print(f"[CrossRef] Error: {e}")
            return []

    def search_all(self, query, num_results=5):
        """Search all sources and return combined results."""
        return {
            "web": self.search_web(query, num_results),
            "academic": self.search_crossref(query, num_results),
        }
