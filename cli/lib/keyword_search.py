from pydoc import doc
from typing import Optional

from models import Movie
from preprocessing import InvertedIndex, TextPreprocessor
from typing_extensions import deprecated

from data import load_movies


def search_command(
    query: str,
    text_preprocessor: TextPreprocessor,
    inverted_index: InvertedIndex,
    limit: int = 5,
) -> list[Movie]:

    print("Searching for:", query)
    inverted_index.load()

    seen, result = set(), []
    query_tokens = text_preprocessor.tokenize(query)

    for _, qtoken in enumerate(query_tokens):
        matched_docs_ids = inverted_index.get_documents(qtoken)
        for doc_id in matched_docs_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            result.append(inverted_index.docmap[doc_id])
            if len(result) >= limit:
                return result
    return result


def build_command(inverted_index: InvertedIndex):
    inverted_index.build()
    inverted_index.save()


def tf_command(inverted_index: InvertedIndex, doc_id: int, term: str) -> int:
    inverted_index.load()
    return inverted_index.get_tf(doc_id, term)


def idf_command(inverted_index: InvertedIndex, term: str) -> float:
    inverted_index.load()
    idf_score = inverted_index.get_idf(term)

    return idf_score


def tfidf_command(inverted_index: InvertedIndex, doc_id: int, term: str) -> float:
    tf = tf_command(inverted_index, doc_id, term)
    idf = idf_command(inverted_index, term)
    return tf * idf


def bm25_idf_command(inverted_index: InvertedIndex, term: str) -> float:
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)


def bm25_tf_command(
    inverted_index: InvertedIndex,
    doc_id: int,
    term: str,
    k1: float,
    b: float,
) -> float:
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term, k1, b)


def bm25_search(inverted_index: InvertedIndex, query: str, limit: int = 5):
    inverted_index.load()
    return inverted_index.bm25_search(query, limit)


# -------- #
@deprecated("use search_command instead")
def search_command_old(
    term: str, text_preprocessor: TextPreprocessor, limit: int = 50
) -> list[Movie]:
    print("Searching for:", term)
    movies: list[Movie] = load_movies()
    result: list[Movie] = []
    for idx, movie in enumerate(movies):
        if text_preprocessor.has_substring_token_match(term, movie.title):
            result.append(movie)
        if len(result) >= limit:
            break
    return result
