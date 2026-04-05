"""
HuggingFace Inference API client for BART, Llama, and code generation.
Uses huggingface_hub InferenceClient with multiple providers:
  - BART (summarization) via default hf-inference provider
  - Llama 3.1 (text gen, quiz, code, QA) via SambaNova provider (free)
  - RoBERTa (extractive QA) via default hf-inference provider
No GPU required locally.
"""
import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_API_TOKEN", "")

# Model IDs
BART_MODEL = "facebook/bart-large-cnn"
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# LLAMA_MODEL = "HuggingFaceH4/zephyr-7b-beta"

QA_MODEL = "deepset/roberta-base-squad2"

# print("HF TOKEN:", HF_TOKEN)
class HFClient:
    """Wrapper around HuggingFace Inference API using official SDK."""

    def __init__(self, token=None):
        self.token = token or HF_TOKEN
        # Default provider client (for BART summarization, RoBERTa QA)
        self.default_client = InferenceClient(token=self.token)
        # SambaNova provider client (for Llama text generation)
        self.llama_client = InferenceClient(token=self.token, provider="sambanova")

    def summarize(self, text, max_length=200, min_length=50):
        """Summarize text using BART-large-CNN."""
        truncated = text[:3000]
        try:
            result = self.default_client.summarization(
                truncated,
                model=BART_MODEL,
            )
            if hasattr(result, "summary_text"):
                return result.summary_text
            return str(result)
        except Exception as e:
            print(f"[HFClient] BART summarization error: {e}")
            # Fallback: use Llama for summarization
            return self._llama_chat(
                f"Summarize the following text concisely:\n\n{truncated[:2000]}",
                max_tokens=max_length
            )

    def generate_text(self, prompt, max_length=512):
        """Generate text using Llama 3.1 via SambaNova."""
        return self._llama_chat(prompt, max_tokens=max_length)

    def generate_quiz_json(self, prompt, max_length=1024):
        """Generate quiz JSON using Llama 3.1. Returns raw string."""
        system = (
            "You are a quiz generator. You MUST return ONLY valid JSON, no other text. "
            "Do not include markdown code fences or any explanation."
        )
        return self._llama_chat(prompt, system=system, max_tokens=max_length, temperature=0.4)

    def generate_flashcard_json(self, prompt, max_length=1024):
        """Generate flashcard JSON using Llama 3.1. Returns raw string."""
        system = (
            "You are a flashcard generator. You MUST return ONLY valid JSON, no other text. "
            "Do not include markdown code fences or any explanation."
        )
        return self._llama_chat(prompt, system=system, max_tokens=max_length, temperature=0.4)

    def generate_code(self, prompt, max_length=256):
        """Generate code using Llama 3.1."""
        system = (
            "You are a code generator. Write clean, working code. "
            "Return ONLY the code, no explanations or markdown fences."
        )
        return self._llama_chat(prompt, system=system, max_tokens=max_length, temperature=0.2)

    def answer_question(self, question, context, max_length=300):
        """Answer a question given context. Uses extractive QA first, falls back to Llama."""
        # Try extractive QA first (fast, precise)
        try:
            result = self.default_client.question_answering(
                question=question,
                context=context[:4000],
                model=QA_MODEL,
            )
            if hasattr(result, "answer") and result.answer:
                return result.answer
        except Exception as e:
            print(f"[HFClient] Extractive QA error: {e}")

        # Fallback: generative QA via Llama
        prompt = (
            f"Answer the following question based ONLY on the provided context.\n\n"
            f"Context: {context[:3000]}\n\n"
            f"Question: {question}\n\n"
            f"Provide a detailed, well-formatted answer:"
        )
        return self._llama_chat(prompt, max_tokens=max_length)

    def _llama_chat(self, prompt, system=None, max_tokens=512, temperature=0.7):
        """Internal: call Llama 3.1 via SambaNova chat_completion."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            result = self.llama_client.chat_completion(
                messages=messages,
                model=LLAMA_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            print(f"[HFClient] Llama chat error: {e}")
            return ""
