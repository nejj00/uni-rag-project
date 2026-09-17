"""
LangGraph agent that chooses among the three retrieval strategies (dense,
chunks, hierarchical) as tools, using a hosted Claude model for reasoning.

This is a hand-built ReAct-style loop (not langgraph.prebuilt.create_react_agent)
so the state machine itself - the actual "agentic" part - stays visible:

    agent (LLM decides: answer, or call a tool?)
      -> tools (execute whichever tool(s) it picked)
      -> back to agent, with the tool results appended
    ... repeats until the LLM answers without calling a tool,
        or until max_tool_calls rounds are used up, in which case
        force_final_answer makes it answer with what it already has ...
      -> END

A failed tool call (hallucinated tool name, bad args, or the tool itself
raising) becomes an error ToolMessage fed back to the model instead of
crashing the whole run - the model can react to "that failed", a Python
exception can't be recovered from mid-graph.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from rag_pipeline import RAGPipeline


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_rounds: int


def _format_hits(pipeline: RAGPipeline, acl_ids, scores) -> str:
    """Render retrieval hits as text the model can read and cite from."""
    if not acl_ids:
        return "No results found."

    lines = []
    for acl_id, score in zip(acl_ids, scores):
        doc = pipeline.by_acl_id[acl_id]
        abstract = (doc.get("abstract") or "")[:300]
        lines.append(f"- [{score:.3f}] acl_id={acl_id} \"{doc['title']}\"\n  {abstract}")
    return "\n".join(lines)


def build_retrieval_tools(pipeline: RAGPipeline, k: int = 3) -> list:
    """
    Wrap RAGPipeline's three retrieval strategies as LangGraph tools.

    Each tool's docstring is what the model actually reads to decide which
    one fits a given question - it's doing real work here, not documentation.
    """

    @tool
    def search_dense(query: str) -> str:
        """Search whole documents (title + abstract + authors combined) for a
        broad topical match. Best when the question is about a paper's general
        subject or theme rather than a specific buried detail."""
        acl_ids, scores = pipeline.retrieve_dense(query, k=k)
        return _format_hits(pipeline, acl_ids, scores)

    @tool
    def search_chunks(query: str) -> str:
        """Search fine-grained ~200-word passages within documents. Best when
        the question asks about a specific method, number, definition, or
        detail that could be buried in a paper's body text rather than its
        abstract."""
        acl_ids, scores = pipeline.retrieve_chunks(query, k=k)
        return _format_hits(pipeline, acl_ids, scores)

    @tool
    def search_hierarchical(query: str) -> str:
        """Two-stage search: filter candidate papers by abstract similarity,
        then deeply score their full-text sections. A reasonable general-
        purpose default when you're not sure which other search fits."""
        acl_ids, scores = pipeline.retrieve_hierarchical(query, k=k)
        return _format_hits(pipeline, acl_ids, scores)

    return [search_dense, search_chunks, search_hierarchical]


def build_graph(tools: list, model, max_tool_calls: int = 5, force_first_tool_call: bool = False):
    """
    Compile the agent/tools state machine described in the module docstring.

    Args:
        tools: Tools available to the model.
        model: A chat model supporting .bind_tools().
        max_tool_calls: Cap on tool-call rounds before the model is forced
            to answer with whatever it has gathered so far, rather than
            looping indefinitely.
        force_first_tool_call: Some small/local models reliably produce a
            correct tool call when tool_choice="required", but with the
            default "auto" they'll often just describe searching in plain
            text instead of actually calling anything - a judgment failure,
            not a calling-mechanics failure. If True, tool_choice is forced
            to "required" only for the first turn of a fresh query (before
            any tool has run), then relaxed to "auto" afterwards so the
            model can still choose to answer once it has results. Leave
            False for models (e.g. hosted Claude) that already choose
            reliably on their own - forcing every turn would stop them from
            ever answering without a tool.
    """
    model_required = model.bind_tools(tools, tool_choice="required") if force_first_tool_call else None
    model_auto = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def call_model(state: AgentState) -> dict:
        use_required = force_first_tool_call and state.get("tool_call_rounds", 0) == 0
        model_with_tools = model_required if use_required else model_auto
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def call_tools(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool_fn = tools_by_name.get(tool_call["name"])
            if tool_fn is None:
                content = (
                    f"Error: no such tool '{tool_call['name']}'. "
                    f"Available tools: {list(tools_by_name)}."
                )
            else:
                try:
                    content = str(tool_fn.invoke(tool_call["args"]))
                except Exception as e:
                    content = f"Error calling {tool_call['name']}: {e}"
            results.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

        return {
            "messages": results,
            "tool_call_rounds": state.get("tool_call_rounds", 0) + 1,
        }

    def force_final_answer(state: AgentState) -> dict:
        # Tool-call budget exhausted: ask once more, without tools bound, so
        # the model can't request another round and must answer with what
        # it has already gathered.
        nudge = SystemMessage(
            content=(
                f"You've used the maximum of {max_tool_calls} tool-call rounds. "
                "Answer now using only the information already gathered above - "
                "do not request another tool call."
            )
        )
        response = model.invoke(state["messages"] + [nudge])
        return {"messages": [response]}

    def route(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return END
        if state.get("tool_call_rounds", 0) >= max_tool_calls:
            return "force_final_answer"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("force_final_answer", force_final_answer)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "force_final_answer": "force_final_answer", END: END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("force_final_answer", END)

    return graph.compile()
