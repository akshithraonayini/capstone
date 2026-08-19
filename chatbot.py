import os

import certifi
import truststore

from src.graph import graph

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

truststore.inject_into_ssl()


def main():

    print("=" * 50)
    print("       ENTERPRISE KNOWLEDGE ASSISTANT")
    print("=" * 50)
    print("Type 'exit' to quit.")
    print()

    thread_id = "employee_001"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:

        try:
            query = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        if not query:
            continue

        state = {
            "query": query
        }

        result = graph.invoke(
            state,
            config=config
        )

        if result.get("guardrail_blocked", False):

            print("\nAssistant:")
            print(result.get(
                "answer",
                "I can't process that request."
            ))

            print("\n[Guardrail: BLOCKED]")
            print(
                f"Reason: "
                f"{result.get('guardrail_reason', 'Policy violation.')}"
            )

            print("\n" + "-" * 50)

            continue

        print("\nAssistant:")
        print(result.get(
            "answer",
            "I couldn't generate an answer."
        ))

        evaluation = result.get("evaluation")

        if evaluation:

            faithfulness = evaluation.get(
                "faithfulness"
            )

            relevancy = evaluation.get(
                "answer_relevancy"
            )

            print("\n[RAGAS]")
            print(f"Faithfulness: {faithfulness}")
            print(f"Answer Relevancy: {relevancy}")

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()