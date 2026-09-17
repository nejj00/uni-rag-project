"""
Long document handling strategies.

Implements chunking and summarization strategies for handling long documents
in dense retrieval pipelines.
"""

from typing import List, Tuple

import config
from vector_db import to_query_vector, chunk_point_id


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
    """Manages chunked document retrieval via a Qdrant vector store."""

    COLLECTION_NAME = "chunks"

    def __init__(self, embedder, vector_store, chunker: DocumentChunker = None):
        """
        Initialize chunk retrieval manager.

        Args:
            embedder: Embedder instance for encoding text.
            vector_store: VectorStore instance for indexing/search.
            chunker: DocumentChunker instance. If None, creates one with config values.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker or DocumentChunker()

    def build_chunk_index(self, anthology_sample) -> None:
        """
        Build chunk index for the anthology sample.

        Steps:
        1. Chunk each document's full text, tracking which doc each chunk came from
        2. Encode all chunks
        3. Upload to Qdrant, storing each chunk's source acl_id, its position
           within that document, and its raw text as payload (replaces the
           old in-memory chunk_map -> the DB is the single source of truth
           for the chunk -> document mapping, and also lets you fetch a
           document's chunks back out - see vector_db for filtered scroll).
           Each point's id is derived from (acl_id, position within its own
           document), not global chunk position, so re-ingestion is
           idempotent even if the sample composition/order changes between runs.

        Args:
            anthology_sample: HuggingFace dataset of documents.
        """
        if config.VERBOSE:
            print("Building chunk index...")

        chunks = []
        chunk_acl_ids = []
        chunk_positions = []
        chunk_ids = []

        for doc in anthology_sample:
            full_text = doc.get("full_text", "") or ""
            acl_id = doc["acl_id"]

            for position, chunk in enumerate(self.chunker.chunk(full_text)):
                chunks.append(chunk)
                chunk_acl_ids.append(acl_id)
                chunk_positions.append(position)
                chunk_ids.append(chunk_point_id(acl_id, position))

        if config.VERBOSE:
            print(f"✓ Created {len(chunks)} chunks from {len(anthology_sample)} docs")
            print("Encoding chunks...")

        chunk_embeddings = self.embedder.encode(chunks, show_progress=True)

        payloads = [
            {"acl_id": acl_id, "position": position, "text": text}
            for acl_id, position, text in zip(chunk_acl_ids, chunk_positions, chunks)
        ]
        self.vector_store.build_index(
            self.COLLECTION_NAME, chunk_embeddings, payloads=payloads, ids=chunk_ids
        )

        if config.VERBOSE:
            print(f"✓ Chunk retrieval ready!")

    def retrieve(self, query_embedding, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-k documents using chunk-aware aggregation.

        Steps:
        1. Search Qdrant for top chunks (5-10x k), resolving each hit
           straight to its source document's acl_id via payload
        2. Aggregate scores per document (max pooling)
        3. Rank documents and return top-k

        Args:
            query_embedding: Query embedding (1, embedding_dim).
            k: Number of documents to return.

        Returns:
            Tuple of (acl_ids, aggregated_scores).
        """
        # Fetch more chunks than needed for aggregation
        chunks_to_fetch = max(k * 10, 50)

        acl_id_hits, chunk_scores = self.vector_store.query_points(
            self.COLLECTION_NAME,
            to_query_vector(query_embedding),
            limit=chunks_to_fetch,
            id_payload_key="acl_id",
        )

        # Aggregate scores per document (max pooling)
        doc_scores = {}
        for acl_id, chunk_score in zip(acl_id_hits, chunk_scores):
            if acl_id not in doc_scores:
                doc_scores[acl_id] = chunk_score
            else:
                # Max pooling: keep highest chunk score for each doc
                doc_scores[acl_id] = max(doc_scores[acl_id], chunk_score)

        # Rank documents
        ranked_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        top_docs = ranked_docs[:k]
        doc_indices = [doc_id for doc_id, _ in top_docs]
        doc_scores_result = [score for _, score in top_docs]

        return doc_indices, doc_scores_result