"""
Manual smoke test: run the LangGraph retrieval agent against a real query.

Requires ANTHROPIC_API_KEY set in the environment - get one at
console.anthropic.com (API access is billed separately from any claude.ai
subscription).
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

import config
from embeddings import Embedder
from vector_db import VectorStore
from data_loader import load_cached_sample
from rag_pipeline import RAGPipeline
from agent import build_retrieval_tools, build_graph

if not config.ANTHROPIC_API_KEY:
    raise SystemExit(
        "ANTHROPIC_API_KEY is not set. Export it in your shell before running this script."
    )

SAMPLE_SIZE = 100

embedder = Embedder()
embedder.load_model()

vector_store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

sample = load_cached_sample().select(range(SAMPLE_SIZE))

with open(config.QUERIES_FILE) as f:
    query = json.load(f)["queries"][0]["q"]

pipeline = RAGPipeline(
    anthology_sample=sample,
    embedder=embedder,
    vector_store=vector_store,
    llm_generator=None,  # the agent's own model handles generation, not this
)

tools = build_retrieval_tools(pipeline)
model = ChatAnthropic(model=config.ANTHROPIC_MODEL, api_key=config.ANTHROPIC_API_KEY)
graph = build_graph(tools, model)

system_prompt = SystemMessage(
    content=(
        "You are a research assistant answering questions about ACL Anthology "
        "papers using the search tools available to you. Always search before "
        "answering, and cite the acl_id of any paper you reference."
    )
)

print(f"\nQuery: {query}\n")

result = graph.invoke({"messages": [system_prompt, HumanMessage(content=query)]})

for message in result["messages"]:
    label = message.__class__.__name__
    if getattr(message, "tool_calls", None):
        for call in message.tool_calls:
            print(f"--- {label}: calling {call['name']}({call['args']}) ---")
    elif label == "ToolMessage":
        print(f"--- ToolMessage ---\n{message.content}\n")
    else:
        print(f"--- {label} ---\n{message.content}\n")
