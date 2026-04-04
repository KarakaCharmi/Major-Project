"""
Code generation from research concepts using HuggingFace models.
"""


class CodeGenerator:
    """Generate code from document concepts."""

    def __init__(self, hf_client):
        self.hf_client = hf_client

    def generate(self, concept, language="python"):
        """Generate code that implements the given concept."""
        prompt = (
            f"Write a clean, working {language} implementation of: {concept}\n"
            f"Include comments explaining the code."
        )
        code = self.hf_client.generate_code(prompt, max_length=512)
        return code.strip() if code else f"# Could not generate {language} code for: {concept}"

    def generate_from_doc(self, text, language="python"):
        """Extract key concepts from doc text and generate code for each."""
        prompt = (
            "List the 3 main technical concepts from this text that could be implemented as code. "
            "Return ONLY the concept names, one per line, nothing else:\n\n"
            f"{text[:2000]}"
        )
        raw = self.hf_client.generate_text(prompt, max_length=150)
        concepts = [c.strip() for c in raw.strip().split("\n") if c.strip()][:3]

        if not concepts:
            concepts = ["Data processing pipeline"]

        results = []
        for concept in concepts:
            code = self.generate(concept, language)
            results.append({"concept": concept, "language": language, "code": code})
        return results
