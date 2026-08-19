from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from contextlib import ExitStack

from state import GraphState

from agents.guardrail_agent import guardrail_agent
from agents.retriever_agent import retriever_agent
from agents.response_agent import response_agent
from agents.evaluator_agent import evaluator_agent

# BUILD GRAPH

builder = StateGraph(GraphState)

# NODES

builder.add_node(
    "guardrail",
    guardrail_agent
)

builder.add_node(
    "retriever",
    retriever_agent
)

builder.add_node(
    "response",
    response_agent
)

builder.add_node(
    "evaluator",
    evaluator_agent
)

# START → GUARDRAIL

builder.add_edge(
    START,
    "guardrail"
)

# GUARDRAIL ROUTER

def guardrail_router(state: GraphState):

    if state.get(
        "guardrail_blocked",
        False
    ):

        print(
            "\n[ROUTER] Guardrail BLOCKED request"
        )

        return "blocked"

    print(
        "\n[ROUTER] Guardrail PASSED"
    )

    return "allowed"


builder.add_conditional_edges(
    "guardrail",

    guardrail_router,

    {
        "allowed": "retriever",
        "blocked": END,
    },
)

# RETRIEVER → RESPONSE

builder.add_edge(
    "retriever",
    "response"
)

# RESPONSE → EVALUATOR

builder.add_edge(
    "response",
    "evaluator"
)


# RAGAS ROUTER

def ragas_router(state: GraphState):

    evaluation = state.get(
        "evaluation",
        {}
    )

    faithfulness = evaluation.get(
        "faithfulness",
        0.0
    )

    answer_relevancy = evaluation.get(
        "answer_relevancy",
        0.0
    )

    attempts = state.get(
        "retrieval_attempts",
        0
    )

    print("\n========== RAGAS ROUTER ==========")

    print(
        f"Faithfulness:     {faithfulness:.4f}"
    )

    print(
        f"Answer Relevancy: {answer_relevancy:.4f}"
    )

    print(
        f"Retrieval Attempts: {attempts}"
    )

    # Threshold

    threshold = 0.7

    # Good evaluation

    if (
        faithfulness >= threshold
        and
        answer_relevancy >= threshold
    ):

        print(
            "[ROUTER] RAGAS PASSED → END"
        )

        return "end"

    # Maximum retry protection

    if attempts >= 2:

        print(
            "[ROUTER] Maximum retrieval retries reached → END"
        )

        return "end"

    # Low score → retrieve again

    print(
        "[ROUTER] RAGAS LOW → RETRY RETRIEVAL"
    )

    return "retry"

# EVALUATOR → CONDITIONAL ROUTING

builder.add_conditional_edges(
    "evaluator",

    ragas_router,

    {
        "retry": "retriever",
        "end": END,
    },
)
# SQLITE CHECKPOINTER

_stack = ExitStack()

checkpointer = _stack.enter_context(
    SqliteSaver.from_conn_string(
        "chatbot.db"
    )
)
# COMPILE

graph = builder.compile(
    checkpointer=checkpointer
)