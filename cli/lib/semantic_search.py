import numpy as np
from models import Movie
from mpmath.libmp.backend import os
from sentence_transformers import SentenceTransformer

from data import load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

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
