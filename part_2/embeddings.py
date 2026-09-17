"""
Embedding generation module.

Handles sentence embedding generation for the retrieval strategies.
Index storage/search lives in vector_db.py (Qdrant-backed).
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from vector_db import acl_id_to_point_id


class Embedder:
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding manager.

        Args:
            model_name: HuggingFace model name. If None, uses config.EMBEDDING_MODEL_NAME.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.model = None

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


def build_dense_index(anthology_sample, embedder: "Embedder", vector_store, collection_name: str = "dense") -> None:
    """
    Encode each document's combined fields and upload them to `collection_name`,
    one point per document. Shared by main.py and RAGPipeline so the id/payload
    scheme can't drift between the two places that build this collection.

    Args:
        anthology_sample: HuggingFace dataset of documents.
        embedder: Embedder instance for encoding text.
        vector_store: VectorStore instance for indexing.
        collection_name: Name of the collection to build.
    """
    docs = [combine_acl_fields(doc) for doc in anthology_sample]
    embeddings = embedder.encode(docs, show_progress=True)

    ids = [acl_id_to_point_id(doc["acl_id"]) for doc in anthology_sample]
    payloads = [{"acl_id": doc["acl_id"]} for doc in anthology_sample]

    vector_store.build_index(collection_name, embeddings, payloads=payloads, ids=ids)


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