import os
import math

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
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

    # -----------------------------------------
    # Create RAGAS sample
    # -----------------------------------------

    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
    )

    dataset = EvaluationDataset(
        samples=[sample]
    )

    # -----------------------------------------
    # RAGAS evaluation
    # -----------------------------------------

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=1),
        ],
        llm=llm,
        embeddings=embeddings,
    )

    scores = result.to_pandas().iloc[0].to_dict()

    faithfulness = float(
        scores.get("faithfulness", 0.0)
    )

    answer_relevancy = float(
        scores.get("answer_relevancy", 0.0)
    )

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