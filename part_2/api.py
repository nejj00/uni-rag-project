"""
FastAPI wrapper around the LangGraph retrieval agent, backed by the local
llama.cpp server (see agent.py / test_agent_local.py for the standalone
script version this is built from).

Run:
    uvicorn api:app --reload

Requires:
    - Qdrant running (docker compose up -d) with dense/chunks/abstracts
      collections already built
    - llama-server running: llama serve --hf-repo Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import config
from embeddings import Embedder
from vector_db import VectorStore
from data_loader import load_cached_sample
from rag_pipeline import RAGPipeline
from agent import build_retrieval_tools, build_graph

SAMPLE_SIZE = 100

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a research assistant answering questions about ACL Anthology "
        "papers using the search tools available to you. Always search before "
        "answering, and cite the acl_id of any paper you reference."
    )
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything expensive (loading the embedding model, connecting to
    # Qdrant, building the graph) happens once here, not per-request.
    embedder = Embedder()
    embedder.load_model()

    vector_store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    sample = load_cached_sample().select(range(SAMPLE_SIZE))

    pipeline = RAGPipeline(
        anthology_sample=sample,
        embedder=embedder,
        vector_store=vector_store,
        llm_generator=None,  # the agent's own model handles generation
    )

    tools = build_retrieval_tools(pipeline)
    model = ChatOpenAI(
        model=config.LLAMACPP_MODEL_LABEL,
        base_url=config.LLAMACPP_BASE_URL,
        api_key="not-needed",
        temperature=0.1,
        max_tokens=400,
    )
    state["graph"] = build_graph(tools, model, force_first_tool_call=True)

    yield

    state.clear()


app = FastAPI(title="ACL RAG Agent API", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "ready": "graph" in state}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    graph = state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    try:
        result = graph.invoke(
            {"messages": [SYSTEM_PROMPT, HumanMessage(content=request.query)]}
        )
    except Exception as e:
        # e.g. llama-server unreachable mid-request - fail as a clean 502,
        # not an unhandled 500
        raise HTTPException(status_code=502, detail=f"Agent invocation failed: {e}")

    tool_calls = [
        f"{call['name']}({call['args']})"
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]

    return AskResponse(answer=result["messages"][-1].content, tool_calls=tool_calls)
