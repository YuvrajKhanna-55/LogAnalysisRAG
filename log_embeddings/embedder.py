"""
Log embedding module using sentence-transformers and Model_Name=all-MiniLM-L6-v2.
Converts parsed log text into dense vector embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LogEmbedder:
    """Generate embeddings for log documents using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        batch_size: int = 256,
    ):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk_documents(self, documents: list[str]) -> list[str]:
        """Split documents into smaller chunks for embedding."""
        chunks = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc)
            chunks.extend(splits)
        return chunks

    def embed(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Returns numpy array of shape (num_texts, embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_documents(
        self, documents: list[str], show_progress: bool = True
    ) -> tuple[list[str], np.ndarray]:
        """Chunk documents and generate their embeddings.
        """
        print(f"Chunking {len(documents)} documents...")
        chunks = self.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks. Generating embeddings...")
        embeddings = self.embed(chunks, show_progress=show_progress)
        return chunks, embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()
