from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings


PERSIST_DIRECTORY = "chroma_db"


def get_retriever():

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )