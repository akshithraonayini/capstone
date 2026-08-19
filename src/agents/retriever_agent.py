import os

from mcp_client import call_mcp
from rag.vectorstore import get_vectorstore, create_vectorstore


def extract_mcp_result(result):
    """
    Extract the payload from a FastMCP tool response.

    FastMCP wraps a tool's return value in `structuredContent`. For a tool
    that returns a scalar/list, the value lives under the "result" key.
    """

    if hasattr(result, "structuredContent"):
        structured = result.structuredContent

        if isinstance(structured, dict):
            return structured.get("result", "")

    return ""


def retriever_agent(state):

    query = state["query"]
    attempts = state.get("retrieval_attempts", 0)
    attempts += 1

    print("\n========== RETRIEVER AGENT ==========")
    print(f"Query: {query}")

    # ------------------------------------------------------------------
    # MCP: discover the available knowledge base (tool #1)
    # ------------------------------------------------------------------

    print("\n[MCP] Listing knowledge documents...")

    documents_result = call_mcp("list_knowledge_documents")
    available_documents = extract_mcp_result(documents_result)

    print(f"[MCP] Available documents: {available_documents}")

    # ------------------------------------------------------------------
    # RAG: semantic search over Chroma to find the most relevant chunks
    # ------------------------------------------------------------------

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)

    # Auto-heal an empty index (first run / cleared store).
    if len(retrieved_docs) == 0:
        print("[RAG] Empty index detected — rebuilding from source documents...")
        create_vectorstore()
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)

    print(f"\nRetrieved {len(retrieved_docs)} chunks from Chroma")

    # ------------------------------------------------------------------
    # Identify which source documents those chunks came from, ranked by
    # how often each document appears in the semantic results.
    # ------------------------------------------------------------------

    ranked_sources = []
    for doc in retrieved_docs:
        source_path = doc.metadata.get("source", "")
        filename = os.path.basename(source_path)
        if filename and filename not in ranked_sources:
            ranked_sources.append(filename)

    print(f"Relevant source documents: {ranked_sources}")

    # ------------------------------------------------------------------
    # MCP: read the FULL text of the most relevant source document(s)
    # (tool #2). This "chunk -> parent document" expansion grounds the
    # answer in the complete, authoritative policy via MCP rather than in
    # possibly-fragmented chunks alone.
    # ------------------------------------------------------------------

    contexts = []

    for filename in ranked_sources[:2]:

        if available_documents and filename not in available_documents:
            # Skip anything the MCP server doesn't actually serve.
            continue

        print(f"\n[MCP] Reading full document: {filename}")

        document_result = call_mcp(
            "read_knowledge_document",
            {"filename": filename},
        )

        full_text = extract_mcp_result(document_result)

        if full_text:
            print(f"[MCP] Retrieved {len(full_text)} chars from {filename}")
            contexts.append(f"[{filename}]\n{full_text}")

    # Fallback: if MCP returned nothing usable, ground on the raw chunks.
    if not contexts:
        print("[RAG] Falling back to raw retrieved chunks for context.")
        contexts = [doc.page_content for doc in retrieved_docs]

    for i, context in enumerate(contexts, 1):
        print(f"\n--- Context {i} ---")
        print(context)

    return {
        "retrieved_context": contexts,
        "retrieval_attempts": attempts,
    }
