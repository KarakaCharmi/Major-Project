"""
Visualization: Graphviz flowcharts and Plotly charts from document content.
"""
import json
import re
import graphviz
import plotly.graph_objects as go
import plotly.io as pio


class Visualizer:
    """Generate flowcharts and charts from document content."""

    def __init__(self, hf_client):
        self.hf_client = hf_client

    def generate_flowchart(self, text, doc_id="doc"):
        """Generate a flowchart from document structure using Graphviz. Returns SVG string."""
        prompt = (
            "Extract the main steps or process flow from this text as a numbered list. "
            "Return ONLY a numbered list, one step per line, no other text:\n\n"
            f"{text[:3000]}"
        )
        raw = self.hf_client.generate_text(prompt, max_length=300)
        steps = [s.strip() for s in raw.strip().split("\n") if s.strip()]
        if not steps:
            steps = ["Start", "Process Document", "Analyze", "Generate Output", "End"]

        cleaned = []
        for s in steps:
            s = re.sub(r"^\d+[.)]\s*", "", s).strip()
            if s:
                cleaned.append(s[:60])
        steps = cleaned if cleaned else ["Start", "End"]

        dot = graphviz.Digraph(format="svg")
        dot.attr(rankdir="TB", bgcolor="transparent")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="#E8F0FE",
                 fontname="Arial", fontsize="11")
        dot.attr("edge", color="#4285F4")

        for i, step in enumerate(steps):
            node_id = f"step_{i}"
            if i == 0:
                dot.node(node_id, step, shape="ellipse", fillcolor="#34A853", fontcolor="white")
            elif i == len(steps) - 1:
                dot.node(node_id, step, shape="ellipse", fillcolor="#EA4335", fontcolor="white")
            else:
                dot.node(node_id, step)
            if i > 0:
                dot.edge(f"step_{i-1}", node_id)

        svg = dot.pipe(format="svg").decode("utf-8")
        return svg

    def generate_word_frequency_chart(self, text):
        """Generate a word frequency bar chart using Plotly. Returns JSON plot data."""
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        stop = {"that", "this", "with", "from", "have", "been", "were", "will",
                "their", "they", "them", "than", "then", "also", "more", "some",
                "which", "what", "when", "where", "would", "could", "should",
                "about", "into", "each", "make", "like", "just", "over", "such"}
        filtered = [w for w in words if w not in stop]

        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]
        if not top:
            return None

        labels = [t[0] for t in top]
        values = [t[1] for t in top]

        fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color="#4285F4")])
        fig.update_layout(
            title="Top Words in Document",
            xaxis_title="Word",
            yaxis_title="Frequency",
            template="plotly_white",
        )
        return json.loads(pio.to_json(fig))

    def generate_topic_distribution(self, text):
        """Generate a pie chart of topic distribution. Returns JSON plot data."""
        prompt = (
            "Identify the main topics in this text and estimate their percentage. "
            "Return ONLY in this format, one per line: TopicName: XX%\n\n"
            f"{text[:2000]}"
        )
        raw = self.hf_client.generate_text(prompt, max_length=200)

        pairs = re.findall(r"([A-Za-z\s]+):\s*(\d+)%", raw)
        if not pairs:
            return None

        labels = [p[0].strip() for p in pairs]
        values = [int(p[1]) for p in pairs]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title="Topic Distribution", template="plotly_white")
        return json.loads(pio.to_json(fig))
