"""Rebuild the Chroma vector index using local embeddings.

Applies the same SSL/proxy setup as chatbot.py so the embedding model
can be downloaded from the Hugging Face hub through a corporate proxy.
"""
import os
import certifi
import truststore

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
truststore.inject_into_ssl()

from src.rag.vectorstore import create_vectorstore


def main():
    vs = create_vectorstore()
    print("DOC COUNT:", vs._collection.count())


if __name__ == "__main__":
    main()
