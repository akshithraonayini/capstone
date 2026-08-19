import os
import math
import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
)

from state import GraphState
from rag.vectorstore import get_embeddings


load_dotenv()


def evaluator_agent(state: GraphState) -> GraphState:

    print("\n========== EVALUATOR AGENT ==========")

    query = state["query"]
    answer = state["answer"]
    contexts = state["retrieved_context"]

    # -----------------------------------------
    # Groq LLM used by RAGAS
    # -----------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # -----------------------------------------
    # Hugging Face embeddings
    # -----------------------------------------

    embeddings = get_embeddings()

    # Wrap for RAGAS
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # -----------------------------------------
    # Create RAGAS sample
    # -----------------------------------------

    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
    )

    # -----------------------------------------
    # RAGAS evaluation
    #
    # We score each metric directly via single_turn_ascore
    # instead of ragas.evaluate(). The high-level evaluate()
    # builds an EvaluationResult that parses callback traces,
    # which crashes with "IndexError: list index out of range"
    # (parse_run_traces) when the tracer collects no root trace.
    # Direct scoring bypasses that path entirely and still
    # returns the same numeric scores.
    # -----------------------------------------

    faithfulness_metric = Faithfulness(llm=ragas_llm)
    relevancy_metric = ResponseRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        strictness=1,
    )

    async def _score():
        f = await faithfulness_metric.single_turn_ascore(sample)
        r = await relevancy_metric.single_turn_ascore(sample)
        return f, r

    # Disable LangSmith tracing during scoring so scoring does not
    # spam the console with "CERTIFICATE_VERIFY_FAILED" upload errors
    # on networks that block api.smith.langchain.com.
    saved_tracing = {
        key: os.environ.get(key)
        for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    }
    for key in saved_tracing:
        os.environ[key] = "false"
    try:
        faithfulness, answer_relevancy = asyncio.run(_score())
    finally:
        for key, value in saved_tracing.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    faithfulness = float(faithfulness)
    answer_relevancy = float(answer_relevancy)

    # Handle NaN
    if math.isnan(faithfulness):
        faithfulness = 0.0

    if math.isnan(answer_relevancy):
        answer_relevancy = 0.0

    # -----------------------------------------
    # Display scores
    # -----------------------------------------

    print("\nRAGAS RESULTS")
    print("-------------------------")

    print(
        f"Faithfulness:     {faithfulness:.4f}"
    )

    print(
        f"Answer Relevancy: {answer_relevancy:.4f}"
    )

    # -----------------------------------------
    # Interpretation
    # -----------------------------------------

    if faithfulness >= 0.8:

        faithfulness_text = (
            "High - answer is well supported by "
            "the retrieved context."
        )

    elif faithfulness >= 0.6:

        faithfulness_text = (
            "Moderate - answer is mostly supported "
            "by the retrieved context."
        )

    else:

        faithfulness_text = (
            "Low - answer may contain unsupported "
            "information."
        )

    if answer_relevancy >= 0.8:

        relevancy_text = (
            "High - answer directly addresses "
            "the user's question."
        )

    elif answer_relevancy >= 0.6:

        relevancy_text = (
            "Moderate - answer is reasonably relevant."
        )

    else:

        relevancy_text = (
            "Low - answer may not directly address "
            "the question."
        )

    print("\nINTERPRETATION")
    print("-------------------------")

    print(
        f"Faithfulness: {faithfulness_text}"
    )

    print(
        f"Answer Relevancy: {relevancy_text}"
    )

    # -----------------------------------------
    # Return evaluation
    # -----------------------------------------

    return {
        **state,
        "evaluation": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "interpretation": {
                "faithfulness": faithfulness_text,
                "answer_relevancy": relevancy_text,
            },
        },
    }