import json
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
from constants import DEFAULT_MODEL_NAME, DEFAULT_SEARCH_LIMIT, DOCUMENT_PREVIEW_LENGTH
from lib.chunk_utils import semantic_chunk_doc
from models import Movie
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from data import CACHE_MOVIE_EMBEDDINGS, load_movies

Embedding = NDArray[np.float32]
CHUNK_EMBEDDINGS_PATH = Path("cache/chunk_embeddings.npy")
CHUNK_METADATA_PATH = Path("cache/chunk_metadata.json")
SCORE_PRECISION = 4


class SearchResult(TypedDict):
    score: float
    title: str
    description: str
    id: NotRequired[int]
    document: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float


class SemanticSearch:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: Embedding | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        similarities: list[tuple[float, Movie]] = []
        for idx, movie_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(movie_embedding, query_embedding)
            similarities.append((similarity_score, self.documents[idx]))

        similarities.sort(key=lambda k: k[0], reverse=True)

        return [
            {"score": score, "title": movie.title, "description": movie.description}
            for score, movie in similarities[:limit]
        ]

    def load_or_create_embeddings(self, documents: list[Movie]) -> Embedding:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc.id] = doc

        if CACHE_MOVIE_EMBEDDINGS.exists():
            self.embeddings = np.asarray(np.load(CACHE_MOVIE_EMBEDDINGS), dtype=np.float32)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def build_embeddings(self, documents: list[Movie]) -> Embedding:
        self.documents = documents
        self.document_map = {}
        movie_texts: list[str] = []

        for doc in documents:
            self.document_map[doc.id] = doc
            movie_texts.append(f"{doc.title}: {doc.description}")

        self.embeddings = np.asarray(
            self.model.encode(movie_texts, show_progress_bar=True), dtype=np.float32
        )

        CACHE_MOVIE_EMBEDDINGS.parent.mkdir(exist_ok=True, parents=True)
        np.save(CACHE_MOVIE_EMBEDDINGS, self.embeddings)
        return self.embeddings

    def generate_embedding(self, text: str) -> Embedding:
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        embedding: Embedding = np.asarray(self.model.encode([text]), dtype=np.float32)
        return embedding[0]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings: Embedding | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> Embedding:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc.id] = doc

        chunks: list[str] = []
        metadata: list[ChunkMetadata] = []

        for doc_idx, doc in enumerate(documents):
            if not doc.description or not doc.description.strip():
                continue

            current_chunks = semantic_chunk_doc(doc.description, chunk_size=4, overlap=1)
            chunks.extend(current_chunks)

            for chunk_idx, chunk in enumerate(current_chunks):
                chunks.append(chunk)
                metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(current_chunks),
                    }
                )

        self.chunk_embeddings = np.asarray(
            self.model.encode(chunks, show_progress_bar=True), dtype=np.float32
        )
        self.chunk_metadata = metadata

        CHUNK_EMBEDDINGS_PATH.parent.mkdir(exist_ok=True, parents=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with CHUNK_METADATA_PATH.open("w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> Embedding:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc.id] = doc

        if CHUNK_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.asarray(np.load(CHUNK_EMBEDDINGS_PATH), dtype=np.float32)
            with CHUNK_METADATA_PATH.open("r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    # TODO: ugly code, refactor this "Chunked Semantic Search" - ch5/L5
    def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
        if self.chunk_embeddings is None or self.chunk_metadata is None or self.documents is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores: list[ChunkScore] = []
        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embedding, np.asarray(chunk_embedding))
            chunk_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[idx]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                    "score": similarity_score,
                }
            )

        best_score_by_movie_idx: dict[int, float] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if (
                movie_idx not in best_score_by_movie_idx
                or score >= best_score_by_movie_idx[movie_idx]
            ):
                best_score_by_movie_idx[movie_idx] = score

        ranked_movie_indices = sorted(
            best_score_by_movie_idx.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[SearchResult] = []
        for movie_idx, score in ranked_movie_indices[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc.id,
                    title=doc.title,
                    document=doc.description[:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )
        return results


def cosine_similarity(vec1: Embedding, vec2: Embedding) -> float:
    # cos(theta) = dot_product(vec1, vec2) / (norm(vec1) * norm(vec2))
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query(query: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: usually short description
        score: similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "description": document,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
