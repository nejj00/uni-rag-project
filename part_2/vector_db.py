import uuid
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Fixed namespace for name-based (v5) point IDs, so the same natural key
# always hashes to the same UUID across processes/runs.
_ID_NAMESPACE = uuid.NAMESPACE_DNS


def to_query_vector(embedding: np.ndarray) -> List[float]:
    """
    Flatten a (1, embedding_dim) query embedding into a plain list of floats,
    the shape qdrant-client expects for a single nearest-neighbor query.
    """
    return embedding[0].tolist()


def acl_id_to_point_id(acl_id: str) -> str:
    """
    Deterministically derive a Qdrant-compatible point id (a UUID) from a
    document's natural key. Qdrant only accepts unsigned ints or UUIDs as
    point ids, not arbitrary strings, so acl_id itself can't be used
    directly. Hashing it this way means the same paper always maps to the
    same point id regardless of its position in whatever sample it's part
    of, so re-ingestion overwrites in place instead of duplicating or
    drifting when the sample composition/order changes between runs.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, acl_id))


def chunk_point_id(acl_id: str, chunk_position: int) -> str:
    """
    Same idea as acl_id_to_point_id, but for a chunk: keyed on the chunk's
    position within its own document (not its position in the overall
    chunk list), since that's the part that stays stable across resampling.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, f"{acl_id}:{chunk_position}"))


class VectorStore:
    def __init__(self, url: str, api_key: str = None):
        self.client = QdrantClient(url=url, api_key=api_key)

    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection already exists (e.g. from a previous run)."""
        return self.client.collection_exists(collection_name)

    def create_collection(self, collection_name: str, vector_size: int, distance_metric: Distance = Distance.COSINE):
        """
        Create a collection in Qdrant.

        Args:
            collection_name: Name of the collection.
            vector_size: Size of the vectors.
            distance_metric: Distance metric to use ("Cosine", "Euclidean", "Dot").
        """

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(collection_name, vectors_config=VectorParams(size=vector_size, distance=distance_metric))

    def build_index(
        self,
        collection_name: str,
        embeddings: np.ndarray,
        payloads: Optional[List[dict]] = None,
        ids: Optional[List[int]] = None,
        distance_metric: Distance = Distance.COSINE,
        batch_size: int = 256,
    ):
        """
        Create the collection if needed and upload a full set of embeddings
        to it, batching the upload so large datasets don't go through in a
        single request.

        Args:
            collection_name: Name of the collection.
            embeddings: Embedding matrix of shape (n_docs, embedding_dim).
            payloads: Optional list of payload dicts, one per embedding.
                Defaults to an empty payload for each point.
            ids: Optional list of point ids, one per embedding. Defaults to
                the embedding's row position (0..n-1).
            distance_metric: Distance metric to use if the collection doesn't exist yet.
            batch_size: Number of points uploaded per request.
        """
        self.create_collection(collection_name, embeddings.shape[1], distance_metric)

        if ids is None:
            ids = list(range(len(embeddings)))
        if payloads is None:
            payloads = [{} for _ in ids]

        points = (
            PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=payloads[i])
            for i in range(len(embeddings))
        )

        self.client.upload_points(collection_name=collection_name, points=points, batch_size=batch_size)

    def upsert_points(self, collection_name: str, points: list):
        """
        Upsert points into a collection.

        Args:
            collection_name: Name of the collection.
            points: List of PointStruct objects to upsert.
        """
        self.client.upsert(collection_name=collection_name, points=points)
    
    def query_points(self, collection_name: str, query_vector: list, limit: int = 5, id_payload_key: str = None):
        """
        Query points from a collection.

        Args:
            collection_name: Name of the collection.
            query_vector: The vector to query against.
            limit: Number of results to return.
            id_payload_key: If set, read the returned index from this payload
                field instead of the point's own id. Use this for collections
                where points are sub-units of a document (e.g. chunks), so the
                payload's document index is returned instead of the chunk's
                own id.
        """
        results = self.client.query_points(collection_name=collection_name, query=query_vector, limit=limit)

        if id_payload_key:
            doc_indices = [point.payload[id_payload_key] for point in results.points]
        else:
            doc_indices = [point.id for point in results.points]

        scores = [point.score for point in results.points]

        return doc_indices, scores