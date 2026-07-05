from constants import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from models import Movie
from preprocessing import InvertedIndex, TextPreprocessor, create_inverted_index
from typing_extensions import deprecated

from data import load_movies


def search_command(
    query: str,
    inverted_index: InvertedIndex,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Movie]:
    inverted_index.load_or_build()

    seen, result = set(), []
    query_tokens = inverted_index.text_preprocessor.tokenize(query)

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


def build_command(inverted_index: InvertedIndex) -> None:
    inverted_index.build()
    inverted_index.save()


def tf_command(inverted_index: InvertedIndex, doc_id: int, term: str) -> int:
    inverted_index.load_or_build()
    return inverted_index.get_tf(doc_id, term)


def idf_command(inverted_index: InvertedIndex, term: str) -> float:
    inverted_index.load_or_build()
    idf_score = inverted_index.get_idf(term)

    return idf_score


def tfidf_command(inverted_index: InvertedIndex, doc_id: int, term: str) -> float:
    inverted_index.load_or_build()
    tf = inverted_index.get_tf(doc_id, term)
    idf = inverted_index.get_idf(term)
    return tf * idf


def bm25_idf_command(inverted_index: InvertedIndex, term: str) -> float:
    inverted_index.load_or_build()
    return inverted_index.get_bm25_idf(term)


def bm25_tf_command(
    inverted_index: InvertedIndex,
    doc_id: int,
    term: str,
    k1: float,
    b: float,
) -> float:
    inverted_index.load_or_build()
    return inverted_index.get_bm25_tf(doc_id, term, k1, b)


def bm25_search(
    inverted_index: InvertedIndex,
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[tuple[Movie, float]]:
    inverted_index.load_or_build()
    return inverted_index.bm25_search(query, limit)


def keyword_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    inverted_index = create_inverted_index()
    results = search_command(query, inverted_index, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for idx, movie in enumerate(results, 1):
        print(f"{idx}. {movie.title}")


def build_index() -> None:
    inverted_index = create_inverted_index()
    build_command(inverted_index)


def term_frequency(doc_id: int, term: str) -> None:
    inverted_index = create_inverted_index()
    term_token = inverted_index.text_preprocessor.tokenize_single_term(term)
    tf = tf_command(inverted_index, doc_id, term_token)
    print(tf)


def inverse_document_frequency(term: str) -> None:
    inverted_index = create_inverted_index()
    term_token = inverted_index.text_preprocessor.tokenize_single_term(term)
    idf = idf_command(inverted_index, term_token)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_score(doc_id: int, term: str) -> None:
    inverted_index = create_inverted_index()
    term_token = inverted_index.text_preprocessor.tokenize_single_term(term)
    tf_idf = tfidf_command(inverted_index, doc_id, term_token)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25_idf(term: str) -> None:
    inverted_index = create_inverted_index()
    term_token = inverted_index.text_preprocessor.tokenize_single_term(term)
    score = bm25_idf_command(inverted_index, term_token)
    print(f"BM25 IDF score of '{term}': {score:.2f}")


def bm25_tf(
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> None:
    inverted_index = create_inverted_index()
    term_token = inverted_index.text_preprocessor.tokenize_single_term(term)
    score = bm25_tf_command(inverted_index, doc_id, term_token, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {score:.2f}")


def bm25_keyword_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    inverted_index = create_inverted_index()
    results = bm25_search(inverted_index, query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for idx, (movie, score) in enumerate(results, 1):
        print(f"{idx}. {movie.title} (score: {score:.2f})")


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
