"""
Embedding and FAISS index management module.

Handles sentence embedding generation and FAISS index construction
for efficient similarity-based document retrieval.
"""

from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


class EmbeddingManager:
    """Manages embeddings and FAISS indices for different strategies."""

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding manager.

        Args:
            model_name: HuggingFace model name. If None, uses config.EMBEDDING_MODEL_NAME.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.model = None
        self.indices = {}  # Strategy -> FAISS index
        self.embeddings = {}  # Strategy -> embeddings array

    def load_model(self) -> None:
        """Load the sentence transformer model."""
        if config.VERBOSE:
            print(f"Loading embedding model: {self.model_name}...")

        self.model = SentenceTransformer(self.model_name)

        if config.VERBOSE:
            print(f"✓ Model loaded successfully!")

    def encode(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding. If None, uses config.EMBEDDING_BATCH_SIZE.
            show_progress: Whether to show progress bar.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings

    def build_index(
        self,
        embeddings: np.ndarray,
        strategy: str = "dense",
        metric: str = "ip",  # inner product (cosine similarity for normalized)
    ) -> faiss.Index:
        """
        Build FAISS index for embeddings.

        Args:
            embeddings: Embedding matrix of shape (n_docs, embedding_dim).
            strategy: Name of retrieval strategy (for storage).
            metric: Distance metric ("ip" for inner product, "l2" for L2).

        Returns:
            FAISS index.
        """
        if config.VERBOSE:
            print(
                f"Building FAISS index for '{strategy}' strategy "
                f"({embeddings.shape[0]} documents)..."
            )

        dimension = embeddings.shape[1]

        if metric == "ip":
            index = faiss.IndexFlatIP(dimension)
        elif metric == "l2":
            index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        index.add(embeddings.astype(np.float32))

        self.indices[strategy] = index
        self.embeddings[strategy] = embeddings

        if config.VERBOSE:
            print(f"✓ FAISS index built with {index.ntotal} documents")

        return index

    def search(
        self,
        query_embedding: np.ndarray,
        strategy: str,
        k: int = 10,
    ) -> Tuple[List[int], List[float]]:
        """
        Search for nearest neighbors in FAISS index.

        Args:
            query_embedding: Single query embedding of shape (1, embedding_dim).
            strategy: Retrieval strategy name.
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (indices, scores).
        """
        if strategy not in self.indices:
            raise ValueError(f"No index for strategy '{strategy}'")

        index = self.indices[strategy]
        scores, indices = index.search(query_embedding.astype(np.float32), k)

        return indices[0].tolist(), scores[0].tolist()

    def get_embeddings(self, strategy: str) -> np.ndarray:
        """Get embeddings for a strategy."""
        if strategy not in self.embeddings:
            raise ValueError(f"No embeddings for strategy '{strategy}'")
        return self.embeddings[strategy]

    def get_index(self, strategy: str) -> faiss.Index:
        """Get FAISS index for a strategy."""
        if strategy not in self.indices:
            raise ValueError(f"No index for strategy '{strategy}'")
        return self.indices[strategy]


def combine_acl_fields(doc) -> str:
    """
    Combine relevant ACL document fields into a single text.

    Used for dense document-level retrieval (title + abstract + authors).

    Args:
        doc: Document dictionary with fields.

    Returns:
        Combined text string.
    """
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    authors = (
        " ".join(doc.get("authors", []))
        if isinstance(doc.get("authors"), list)
        else ""
    )

    return f"{title}. {abstract}. Authors: {authors}".strip()


def combine_summary(doc) -> str:
    """
    Create summary representation of document (title + abstract).

    Used for summary-based retrieval.

    Args:
        doc: Document dictionary.

    Returns:
        Summary text string.
    """
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    return f"{title}. {abstract}".strip()
