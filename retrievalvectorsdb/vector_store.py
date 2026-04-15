"""
Vector store module using FAISS for similarity search
and BM25 for keyword-based retrieval (hybrid search).
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


class VectorStore:
    """Hybrid vector store combining FAISS (dense) and BM25 (sparse) retrieval."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[str], embeddings: np.ndarray) -> None:
        """Build the FAISS index and BM25 index from chunks and embeddings."""
        self.chunks = chunks
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.embedding_dim = embeddings.shape[1]
        
        #BM25 index
        tokenized = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"Vector store built: {len(chunks)} chunks, dim={self.embedding_dim}")

    def search_dense(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search using FAISS (dense vectors) + Returns (chunk, score) pairs."""
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results

    def search_sparse(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search using BM25-keyword-based. Returns chunks, score pairs."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build() first.")
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]
        return results

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        alpha: float = 0.7,
    ) -> list[tuple[str, float]]:
        """Hybrid search combining dense (FAISS) and sparse (BM25) results.

        alpha: weight for dense scores (1-alpha for sparse). Default 0.7.
        """
        dense_results = self.search_dense(query_embedding, top_k=top_k * 2)
        sparse_results = self.search_sparse(query, top_k=top_k * 2)

        # Normalize scores
        score_map: dict[str, float] = {}

        if dense_results:
            max_d = max(s for _, s in dense_results)
            min_d = min(s for _, s in dense_results)
            range_d = max_d - min_d if max_d != min_d else 1.0
            for chunk, score in dense_results:
                norm_score = (score - min_d) / range_d
                score_map[chunk] = alpha * norm_score

        if sparse_results:
            max_s = max(s for _, s in sparse_results)
            min_s = min(s for _, s in sparse_results)
            range_s = max_s - min_s if max_s != min_s else 1.0
            for chunk, score in sparse_results:
                norm_score = (score - min_s) / range_s
                score_map[chunk] = score_map.get(chunk, 0.0) + (1 - alpha) * norm_score

        sorted_results = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def save(self, directory: str) -> None:
        """Save the vector store to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)

        if self.bm25 is not None:
            with open(path / "bm25.pkl", "wb") as f:
                pickle.dump(self.bm25, f)
        # Save metadata
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"embedding_dim": self.embedding_dim, "num_chunks": len(self.chunks)}, f)

        print(f"Vector store saved to {path}")

    def load(self, directory: str) -> None:
        """Load the vector store from disk."""
        path = Path(directory)

        # Load metadata
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.embedding_dim = metadata["embedding_dim"]

        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))

        # Load chunks
        with open(path / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Load BM25
        with open(path / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        print(f"Vector store loaded: {len(self.chunks)} chunks, dim={self.embedding_dim}")
