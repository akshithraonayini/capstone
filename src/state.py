from typing import TypedDict
from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):

    # User query
    query: str

    # RAG
    retrieved_context: list[str]

    # Generated response
    answer: str

    # RAGAS evaluation
    evaluation: dict

    # Chat history
    messages: list[BaseMessage]

    # Guardrail
    guardrail_blocked: bool
    guardrail_reason: str

    # Number of retrieval attempts
    retrieval_attempts: int