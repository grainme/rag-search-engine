from constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
)
from lib.chunk_utils import chunk_doc, semantic_chunk_doc
from lib.semantic_search import ChunkedSemanticSearch

from data import load_movies


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = chunk_doc(text, chunk_size, overlap)

    print(f"Chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks):
        print(f"{idx + 1}. {chunk}")


def semantic_chunk_text(
    text: str,
    chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunk_doc(text, chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks):
        print(f"{idx + 1}. {chunk}")


def embed_chunks():
    documents = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int):
    documents = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    _ = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    results = chunked_semantic_search.search_chunks(query, limit)
    for res in results:
        print(res["title"], res["score"], sep=" - ")
