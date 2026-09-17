"""
Manual smoke test: run the LangGraph retrieval agent against a local
llama.cpp server instead of a hosted API.

Prerequisite: llama-server running with tool-calling enabled, e.g.:

    llama-server --hf-repo Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --jinja

This exposes an OpenAI-compatible endpoint at http://localhost:8080/v1,
which is why the client below is ChatOpenAI (pointed at localhost) rather
than anything llama.cpp-specific - same client works against any
OpenAI-compatible server, including this one and a bigger one on another
machine (just change LLAMACPP_BASE_URL).
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import config
from embeddings import Embedder
from vector_db import VectorStore
from data_loader import load_cached_sample
from rag_pipeline import RAGPipeline
from agent import build_retrieval_tools, build_graph

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
model = ChatOpenAI(
    model=config.LLAMACPP_MODEL_LABEL,
    base_url=config.LLAMACPP_BASE_URL,
    api_key="not-needed",  # llama-server doesn't check this by default
    temperature=0.1,  # small models ramble/repeat more at higher temperature
    max_tokens=400,  # hard cap - nothing else stops a runaway generation
)
graph = build_graph(tools, model, force_first_tool_call=True)

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
