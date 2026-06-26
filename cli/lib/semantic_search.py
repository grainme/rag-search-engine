import numpy as np
from models import Movie
from mpmath.libmp.backend import os
from sentence_transformers import SentenceTransformer

from data import load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # the embeddings of documents
        self.embeddings = None
        # all the documents
        self.documents = None
        # maps doc_id to document
        self.document_map = {}

    def search(self, query: str, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        similarity_scores: list[tuple[float, Movie]] = []
        for idx, embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(
                np.array(embedding), np.array(query_embedding)
            )
            similarity_scores.append((similarity_score, self.documents[idx]))
        # sorting by similarity score
        similarity_scores.sort(key=lambda t: t[0], reverse=True)
        return [
            {"score": ss[0], "title": ss[1].title, "description": ss[1].description}
            for ss in similarity_scores[:limit]
        ]

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("text is empty")
        model_output = self.model.encode([text])
        embedding = model_output[0]
        return embedding

    def build_embeddings(self, documents: list[Movie]):
        movies_strs: list[str] = []
        self.documents = documents
        for doc in documents:
            self.document_map[doc.id] = doc
            movies_strs.append(f"{doc.title}: {doc.description}")

        self.embeddings = self.model.encode(movies_strs, show_progress_bar=True)

        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc.id] = doc

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # cos(teta) = (norm(vec1) * norm(vec2)) / dot_product(vec1, vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


# TODO: THIS IS THE NAIVE APPROACH, REFACTOR & IMPROVEMENT IS NEEDED.
def chunk_doc(text: str, chunk_size: int, overlap: int) -> list[str]:
    text_split = text.split()
    chunks: list[str] = []
    accumulted_chunk: list[str] = []

    for i in range(len(text_split)):
        if len(accumulted_chunk) >= chunk_size:
            if len(chunks) > 0 and overlap > 0:
                last_chunk_overlap = chunks[-1].split()[-overlap:]
                accumulted_chunk = [" ".join(last_chunk_overlap)] + accumulted_chunk
            chunks.append(" ".join(accumulted_chunk))
            accumulted_chunk = []
        accumulted_chunk.append(text_split[i])

    # edge case
    if accumulted_chunk:
        if len(chunks) > 0 and overlap > 0:
            last_chunk_overlap = chunks[-1].split()[-overlap:]
            accumulted_chunk = [" ".join(last_chunk_overlap)] + accumulted_chunk
        chunks.append(" ".join(accumulted_chunk))

    return chunks
