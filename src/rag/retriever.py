from .vectorstore import get_vectorstore


def get_retriever(k: int = 3):
    """
    Return a retriever backed by the persistent Chroma store and the same
    local FastEmbed embeddings used everywhere else in the app. (Previously
    this used a separate remote HuggingFace endpoint, which was inconsistent
    with how the index is actually built.)
    """

    return get_vectorstore().as_retriever(
        search_kwargs={"k": k}
    )
