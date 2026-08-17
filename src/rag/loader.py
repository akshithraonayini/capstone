from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path("data/documents")


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)