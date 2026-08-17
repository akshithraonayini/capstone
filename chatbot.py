from src.graph import graph


def main():

    print("=" * 50)
    print("   ENTERPRISE KNOWLEDGE ASSISTANT")
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

        query = input("You: ").strip()

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

        print("\nAssistant:")
        print(result["answer"])

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()