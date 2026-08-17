from mcp_client import call_mcp
from rag.vectorstore import get_vectorstore


def extract_mcp_result(result):
    """
    Extract text from FastMCP tool response.
    """

    if hasattr(result, "structuredContent"):
        structured = result.structuredContent

        if isinstance(structured, dict):
            return structured.get("result", "")

    return ""


def retriever_agent(state):

    query = state["query"]

    print("\n========== RETRIEVER AGENT ==========")
    print(f"Query: {query}")

    # ------------------------------------------------
    # MCP INTEGRATION
    # ------------------------------------------------

    print("\n[MCP] Listing knowledge documents...")

    documents_result = call_mcp(
        "list_knowledge_documents"
    )

    documents = extract_mcp_result(documents_result)

    print(f"[MCP] Available documents: {documents}")

    # ------------------------------------------------
    # RAG RETRIEVAL
    # ------------------------------------------------

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    retrieved_docs = retriever.invoke(query)

    print(
        f"\nRetrieved {len(retrieved_docs)} documents from Chroma"
    )

    contexts = []

    for i, doc in enumerate(retrieved_docs, 1):

        print(f"\n--- Context {i} ---")
        print(doc.page_content)

        contexts.append(doc.page_content)

    return {
        "retrieved_context": contexts
    }