"""
RAG analyzer module.
Uses retrieved log context + LLM to answer questions about logs.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

try:
    from log_embeddings import LogEmbedder
    from retrievalvectorsdb import VectorStore
except ModuleNotFoundError:
    # Support direct execution: python analyzevectors/analyzer.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from log_embeddings import LogEmbedder
    from retrievalvectorsdb import VectorStore

def _load_environment() -> None:
    """Load environment variables from project root dotenv files."""
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / ".env"

    # Prefer .env; fallback to .env.example for local template-driven setups.
    load_dotenv(dotenv_path=env_file, override=False)

_load_environment()

SYSTEM_PROMPT = """You are an expert log analysis assistant. You analyze system logs
to help engineers diagnose issues, understand system behavior, and identify anomalies.

When given log context and a question:
1. Carefully examine the provided log entries
2. Identify relevant patterns, errors, warnings, and anomalies
3. Provide a clear, structured analysis
4. If the logs don't contain enough information to answer, say so
5. Detailed solution for the patterns, errors, warnings, and anomalies
6. Only use provided logs. Do not assume missing information.

Always reference specific log entries in your analysis when possible."""


class LogAnalyzer:
    """RAG-based log analyzer combining retrieval with LLM generation."""

    DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: LogEmbedder,
        model: str | None = None,
        top_k: int = 10,
        alpha: float = 0.7,
        reranker_model: str | None = None,
        rerank_multiplier: int = 4,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.alpha = alpha
        self.reranker_model = reranker_model or self.DEFAULT_RERANKER_MODEL
        self.rerank_multiplier = max(1, rerank_multiplier)
        self._reranker: CrossEncoder | None = None
        self._reranker_unavailable = False

        # Groq exposes an OpenAI-compatible API.
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.client: OpenAI | None = None
        self.model = model or "llama-3.1-8b-instant"

        if groq_api_key:
            self.client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

    def _get_reranker(self) -> CrossEncoder:
        """Load the cross-encoder reranker lazily."""
        if self._reranker is None:
            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank retrieved chunks with a cross-encoder model.

        Falls back to the original hybrid scores if the reranker cannot be loaded.
        """
        if not candidates:
            return []

        limit = top_k or self.top_k

        if self._reranker_unavailable:
            return candidates[:limit]

        try:
            reranker = self._get_reranker()
            paired_inputs = [(query, chunk) for chunk, _ in candidates]
            rerank_scores = reranker.predict(paired_inputs)
        except Exception as exc:
            self._reranker_unavailable = True
            print(f"Reranker unavailable, falling back to hybrid scores: {exc}")
            return candidates[:limit]

        reranked = sorted(
            ((chunk, float(score)) for (chunk, _), score in zip(candidates, rerank_scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        return reranked[:limit]

    def retrieve(self, query: str, rerank_results: bool = True) -> list[tuple[str, float]]:
        """Retrieve relevant log chunks using hybrid search, then rerank them."""
        query_embedding = self.embedder.embed([query], show_progress=False)
        candidate_k = max(self.top_k * self.rerank_multiplier, self.top_k)
        results = self.vector_store.hybrid_search(
            query=query,
            query_embedding=query_embedding[0],
            top_k=candidate_k,
            alpha=self.alpha,
        )
        if rerank_results:
            return self.rerank(query, results, top_k=self.top_k)
        return results[:self.top_k]

    def analyze(self, query: str) -> dict:
        """Full RAG pipeline: retrieve context, then generate analysis.

        Returns dict with 'answer', 'context', and 'sources'.
        """
        if self.client is None:
            raise ValueError("No API key found. Set GROQ_API_KEY in .env.")

        # Retrieve relevant log chunks
        results = self.retrieve(query)
        context_chunks = [chunk for chunk, _ in results]
        context_text = "\n\n---\n\n".join(context_chunks)

        user_message = (
            f"Log Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Please analyze the log entries above and answer the question."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "context": context_chunks,
            "scores": [score for _, score in results],
            "model": self.model,
        }
