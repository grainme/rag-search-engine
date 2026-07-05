import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from models import Movie, SearchResponse

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
CACHE_DIRECTORY = Path(__file__).resolve().parent.parent / "cache"

MOVIES_PATH = DATA_DIRECTORY / "movies.json"
STOP_WORDS_PATH = DATA_DIRECTORY / "stopwords.txt"

CACHE_INDEX = CACHE_DIRECTORY / "index.pkl"
CACHE_DOCMAP = CACHE_DIRECTORY / "docmap.pkl"
CACHE_TERM_FREQUENCIES = CACHE_DIRECTORY / "term_frequencies.pkl"
CACHE_DOC_LENGTH = CACHE_DIRECTORY / "doc_lengths.pkl"
CACHE_MOVIE_EMBEDDINGS = CACHE_DIRECTORY / "movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = CACHE_DIRECTORY / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIRECTORY / "chunk_metadata.json"


def load_movies(path: Path = MOVIES_PATH) -> list[Movie]:
    with path.open(encoding="utf-8") as file:
        response = SearchResponse.model_validate(json.load(file))

    return response.movies


def load_stop_words(path: Path = STOP_WORDS_PATH) -> set[str]:
    with path.open(encoding="utf-8") as file:
        return set(file.read().splitlines())


def write_index(index: dict[str, set[int]], path: Path = CACHE_INDEX):
    path.parent.mkdir(exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(index, file)


def read_index(path: Path = CACHE_INDEX):
    with path.open("rb") as file:
        loaded_index_data = pickle.load(file)
    return loaded_index_data


def write_docmap(docmap: dict[int, Any], path: Path = CACHE_DOCMAP):
    path.parent.mkdir(exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(docmap, file)


def read_docmap(path: Path = CACHE_DOCMAP):
    with path.open("rb") as file:
        loaded_docmap_data = pickle.load(file)
    return loaded_docmap_data


def write_term_frequencies(term_frquencies: Counter, path: Path = CACHE_TERM_FREQUENCIES):
    path.parent.mkdir(exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(term_frquencies, file)


def read_term_frequencies(path: Path = CACHE_TERM_FREQUENCIES):
    with path.open("rb") as file:
        loaded_term_frequencies = pickle.load(file)
    return loaded_term_frequencies


def write_doc_length(doc_length: dict[int, int], path: Path = CACHE_DOC_LENGTH):
    path.parent.mkdir(exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(doc_length, file)


def read_doc_length(path: Path = CACHE_DOC_LENGTH):
    with path.open("rb") as file:
        loaded_doc_length = pickle.load(file)
    return loaded_doc_length
