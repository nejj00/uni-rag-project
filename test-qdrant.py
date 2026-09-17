from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

# client.create_collection(
#     collection_name="test_collection",
#     vectors_config=VectorParams(size=4, distance=Distance.COSINE),
# )

client.upsert(
    collection_name="test_collection",
    points=[
        PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "hello"}),
        PointStruct(id=2, vector=[0.9, 0.1, 0.1, 0.1], payload={"text": "world"}),
    ],
)

results = client.query_points(
    collection_name="test_collection",
    query=[0.1, 0.2, 0.3, 0.4],
    limit=1,
).points

for r in results:
    print(r.id, r.score, r.payload)