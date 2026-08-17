from typing import TypedDict
from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    query: str
    retrieved_context: list[str]
    answer: str
    evaluation: dict
    messages: list[BaseMessage]