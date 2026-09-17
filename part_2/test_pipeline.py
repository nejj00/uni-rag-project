"""
Manual smoke test: exercise all three retrieval strategies through
RAGPipeline against a real query, without loading the LLM (no GPU needed).
"""

import json

import config
from embeddings import Embedder
from vector_db import VectorStore
from data_loader import load_cached_sample
from rag_pipeline import RAGPipeline

SAMPLE_SIZE = 100

embedder = Embedder()
embedder.load_model()

vector_store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

sample = load_cached_sample().select(range(SAMPLE_SIZE))

with open(config.QUERIES_FILE) as f:
    query = json.load(f)["queries"][0]["q"]

print(f"\nQuery: {query}\n")

pipeline = RAGPipeline(
    anthology_sample=sample,
    embedder=embedder,
    vector_store=vector_store,
    llm_generator=None,  # not touching the LLM for this test
)

for strategy in ["dense", "chunks", "hierarchical"]:
    print(f"=== {strategy} ===")
    indices, scores = pipeline.retrieve(query, strategy=strategy, k=5)
    for acl_id, score in zip(indices, scores):
        doc = pipeline.by_acl_id[acl_id]
        print(f"  [{score:.4f}] {doc['title']}")
    print()
