from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding

from .loader import load_documents, split_documents


PERSIST_DIRECTORY = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Keep the downloaded model inside the project (not the OS temp dir) so it
# survives reboots and is portable.
FASTEMBED_CACHE = ".fastembed_cache"


def _force_gcs_source():
    """huggingface.co is blocked on this network, so drop the HF source from
    fastembed's registry and let it download from the Google Cloud Storage
    mirror (storage.googleapis.com), which is reachable."""
    for desc in TextEmbedding._list_supported_models():
        if desc.model == EMBED_MODEL and desc.sources.hf is not None:
            object.__setattr__(desc.sources, "hf", None)


class FastEmbedEmbeddings(Embeddings):
    """LangChain embeddings backed by fastembed (local ONNX, no HF Inference API)."""

    def __init__(self, model_name=EMBED_MODEL, cache_dir=FASTEMBED_CACHE):
        _force_gcs_source()
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts):
        return [v.tolist() for v in self._model.embed(list(texts))]

    def embed_query(self, text):
        return list(self._model.embed([text]))[0].tolist()


def get_embeddings():
    return FastEmbedEmbeddings()


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
