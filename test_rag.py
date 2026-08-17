from src.rag.vectorstore import create_vectorstore
from src.rag.retriever import get_retriever
from dotenv import load_dotenv
load_dotenv()

def main():
    print("Creating vector database...")

    create_vectorstore()

    print("Vector database created.")

    retriever = get_retriever()

    query = "How many work from home days are allowed per month?"

    documents = retriever.invoke(query)

    print("\nRetrieved Documents:\n")

    for i, document in enumerate(documents, start=1):
        print(f"--- Document {i} ---")
        print(document.page_content)
        print("Source:", document.metadata.get("source"))
        print()


if __name__ == "__main__":
    main()