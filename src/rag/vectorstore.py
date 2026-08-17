from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from .loader import load_documents, split_documents


PERSIST_DIRECTORY = "chroma_db"


def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vectorstore():
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    return vectorstore


def get_vectorstore():
    """
    Load the existing Chroma vector database.
    """

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

    return vectorstore