from src.graph import graph


def main():

    state = {
        "query": "How many work from home days are allowed per month?"
    }

    config = {
        "configurable": {
            "thread_id": "test_user"
        }
    }

    result = graph.invoke(
        state,
        config=config
    )

    print("\n\n========================================")
    print("FINAL RESULT")
    print("========================================")

    print("\nQuestion:")
    print(result["query"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nRAGAS Evaluation:")
    print("----------------------------------------")

    evaluation = result["evaluation"]

    print(
        f"Faithfulness: "
        f"{evaluation['faithfulness']:.4f}"
    )

    print(
        f"Answer Relevancy: "
        f"{evaluation['answer_relevancy']:.4f}"
    )

    print("\nInterpretation:")

    print(
        f"- Faithfulness: "
        f"{evaluation['interpretation']['faithfulness']}"
    )

    print(
        f"- Answer Relevancy: "
        f"{evaluation['interpretation']['answer_relevancy']}"
    )


if __name__ == "__main__":
    main()