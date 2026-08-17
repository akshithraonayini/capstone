from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from contextlib import ExitStack

from state import GraphState

from agents.retriever_agent import retriever_agent
from agents.response_agent import response_agent
from agents.evaluator_agent import evaluator_agent


builder = StateGraph(GraphState)

# Nodes
builder.add_node("retriever", retriever_agent)
builder.add_node("response", response_agent)
builder.add_node("evaluator", evaluator_agent)

# Flow
builder.add_edge(START, "retriever")
builder.add_edge("retriever", "response")
builder.add_edge("response", "evaluator")
builder.add_edge("evaluator", END)


# Keep SQLite connection alive for the lifetime of the application
_stack = ExitStack()

checkpointer = _stack.enter_context(
    SqliteSaver.from_conn_string("chatbot.db")
)

graph = builder.compile(
    checkpointer=checkpointer
)