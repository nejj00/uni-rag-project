"""
Long document handling strategies.

Implements chunking and summarization strategies for handling long documents
in dense retrieval pipelines.
"""

from typing import List, Tuple

import config


class DocumentChunker:
    """Chunks documents into overlapping passages."""

    def __init__(
        self, chunk_size: int = None, overlap: int = None
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Size of chunks in words. If None, uses config.CHUNK_SIZE.
            overlap: Overlap between chunks in words. If None, uses config.CHUNK_OVERLAP.
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP

    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        words = text.split()
        if len(words) == 0:
            return []

        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks


class ChunkRetrieval:
    """Manages chunked document retrieval with FAISS indices."""

    def __init__(self, embedding_manager, chunker: DocumentChunker = None):
        """
        Initialize chunk retrieval manager.

        Args:
            embedding_manager: EmbeddingManager instance for encoding.
            chunker: DocumentChunker instance. If None, creates one with config values.
        """
        self.embedding_manager = embedding_manager
        self.chunker = chunker or DocumentChunker()
        self.chunk_map = []  # Maps chunk_id -> doc_id
        self.chunks = []  # List of chunk texts

    def build_chunk_index(self, anthology_sample) -> None:
        """
        Build chunk index for the anthology sample.

        Steps:
        1. Chunk each document's full text
        2. Track which doc each chunk belongs to
        3. Encode all chunks
        4. Build FAISS index

        Args:
            anthology_sample: HuggingFace dataset of documents.
        """
        if config.VERBOSE:
            print("Building chunk index...")

        # Chunk all documents
        for i, doc in enumerate(anthology_sample):
            full_text = doc.get("full_text", "") or ""

            chunks = self.chunker.chunk(full_text)

            for chunk in chunks:
                self.chunks.append(chunk)
                self.chunk_map.append(i)

        if config.VERBOSE:
            print(f"✓ Created {len(self.chunks)} chunks from {len(anthology_sample)} docs")

        # Encode chunks
        if config.VERBOSE:
            print("Encoding chunks...")

        chunk_embeddings = self.embedding_manager.encode(
            self.chunks, show_progress=True
        )

        # Build FAISS index
        self.embedding_manager.build_index(chunk_embeddings, strategy="chunks")

        if config.VERBOSE:
            print(f"✓ Chunk retrieval ready!")

    def retrieve(self, query_embedding, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k documents using chunk-aware aggregation.

        Steps:
        1. Search FAISS for top chunks (5-10x k)
        2. Aggregate scores per document (max pooling)
        3. Rank documents and return top-k

        Args:
            query_embedding: Query embedding (1, embedding_dim).
            k: Number of documents to return.

        Returns:
            Tuple of (doc_indices, aggregated_scores).
        """
        # Fetch more chunks than needed for aggregation
        chunks_to_fetch = max(k * 10, 50)

        # Search FAISS chunk index
        chunk_indices, chunk_scores = self.embedding_manager.search(
            query_embedding, strategy="chunks", k=chunks_to_fetch
        )

        # Aggregate scores per document (max pooling)
        doc_scores = {}
        for chunk_idx, chunk_score in zip(chunk_indices, chunk_scores):
            if chunk_idx == -1:
                continue

            doc_idx = self.chunk_map[chunk_idx]

            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = chunk_score
            else:
                # Max pooling: keep highest chunk score for each doc
                doc_scores[doc_idx] = max(doc_scores[doc_idx], chunk_score)

        # Rank documents
        ranked_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        top_docs = ranked_docs[:k]
        doc_indices = [doc_id for doc_id, _ in top_docs]
        doc_scores_result = [score for _, score in top_docs]

        return doc_indices, doc_scores_result