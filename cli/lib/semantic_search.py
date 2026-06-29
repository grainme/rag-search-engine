import json
import os
from pathlib import Path
from typing import TypedDict

import numpy as np
from constants import DEFAULT_MODEL_NAME, DEFAULT_SEARCH_LIMIT
from lib.chunk_utils import semantic_chunk_doc
from models import Movie
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from data import CACHE_MOVIE_EMBEDDINGS, load_movies

Embedding = NDArray[np.float32]


class SearchResult(TypedDict):
    score: float
    title: str
    description: str


class SemanticSearch:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        # the embeddings of documents
        self.embeddings: Embedding | None = None
        # all the documents
        self.documents: list[Movie] | None = None
        # maps doc_id to document
        self.document_map: dict[int, Movie] = {}

    def search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarity_scores: list[tuple[float, Movie]] = []
        for idx, embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(embedding, query_embedding)
            similarity_scores.append((similarity_score, self.documents[idx]))
        # sorting by similarity score
        similarity_scores.sort(key=lambda t: t[0], reverse=True)
        return [
            {"score": ss[0], "title": ss[1].title, "description": ss[1].description}
            for ss in similarity_scores[:limit]
        ]

    def generate_embedding(self, text: str) -> Embedding:
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        model_output = np.asarray(self.model.encode([text]), dtype=np.float32)
        embedding = model_output[0]
        return embedding

    def build_embeddings(
        self, documents: list[Movie], path: Path = CACHE_MOVIE_EMBEDDINGS
    ) -> Embedding:
        movies_strs: list[str] = []
        self.documents = documents
        self.document_map = {}

        for doc in documents:
            self.document_map[doc.id] = doc
            movies_strs.append(f"{doc.title}: {doc.description}")

        embeddings = np.asarray(
            self.model.encode(movies_strs, show_progress_bar=True), dtype=np.float32
        )
        self.embeddings = embeddings

        path.parent.mkdir(exist_ok=True)
        np.save(path, embeddings)
        return embeddings

    def load_or_create_embeddings(
        self, documents: list[Movie], path: Path = CACHE_MOVIE_EMBEDDINGS
    ) -> Embedding:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc.id] = doc

        if path.exists():
            embeddings = np.asarray(np.load(path), dtype=np.float32)
            self.embeddings = embeddings
            if len(embeddings) == len(documents):
                return embeddings

        return self.build_embeddings(documents, path)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc.id] = doc

        chunks = []
        # metadata about each chunk
        metadata = []
        for doc_idx, doc in enumerate(documents):
            if not doc.description:
                continue

            # sentences (4), overlap (1)
            curr_chunk = semantic_chunk_doc(doc.description, 4, 1)
            chunks.extend(curr_chunk)

            for chunk_idx, chunk in enumerate(curr_chunk):
                metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(curr_chunk),
                    }
                )

        # ask bootBot about the encode method (specifically "dtype")
        self.chunk_embeddings = self.model.encode(
            chunks,
            show_progress_bar=True,
        )
        self.chunk_metadata = metadata

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks)},
                f,
                indent=2,
            )

        # what type is this?
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc.id] = doc

        # TODO: very ugly code
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists(
            "cache/chunk_metadata.json"
        ):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    # TODO: ugly code, refactor this "Chunked Semantic Search" - ch5/L5
    def search_chunks(self, query: str, limit: int = 10):
        if (
            self.chunk_embeddings is None
            or self.chunk_metadata is None
            or self.documents is None
        ):
            raise ValueError("Fields are not initialized")

        query_embedding = self.generate_embedding(query)
        # this stores "chunk score" dicts
        chunks_score = []

        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            # TODO: ndarray?
            similarity_score = cosine_similarity(
                query_embedding, np.asarray(chunk_embedding)
            )
            chunks_score.append(
                {
                    "chunk_idx": idx,
                    # TODO: i can do better here?
                    "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                    "score": similarity_score,
                }
            )
        # the naming is very weird!
        movie_chunk_score = {}
        for chunk_score in chunks_score:
            score = chunk_score["score"]
            if (chunk_score["movie_idx"] not in movie_chunk_score) or (
                score >= movie_chunk_score[chunk_score["movie_idx"]]
            ):
                movie_chunk_score[chunk_score["movie_idx"]] = score

        movies_sorted_by_score = sorted(
            [k for k in movie_chunk_score.keys()],
            key=lambda x: movie_chunk_score[x],
            reverse=True,
        )[:limit]

        # TODO: step 8
        return [self.documents[k].title for k in movies_sorted_by_score]


#
# HELPERS
#
def cosine_similarity(vec1: Embedding, vec2: Embedding) -> float:
    # cos(theta) = dot_product(vec1, vec2) / (norm(vec1) * norm(vec2))
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def embed_query(query: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query, limit)

    for result in results:
        print(f"{result['title']}: {result['description']} ({result['score']})")


def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def verify_embeddings() -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
