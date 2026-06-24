import argparse

from constants import BM25_B, BM25_K1
from lib.keyword_search import (
    bm25_idf_command,
    bm25_search,
    bm25_tf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from preprocessing import InvertedIndex, TextPreprocessor

from data import load_stop_words


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index structure")

    tf_parser = subparsers.add_parser("tf", help="Term frequency for a term in a doc")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search")

    tf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    tf_parser.add_argument("term", type=str, help="Term to search")

    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF for a term in a doc")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to search")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Search query"
    )

    args = parser.parse_args()

    stop_words = load_stop_words()
    text_preprocessor = TextPreprocessor(stop_words)
    inverted_index = InvertedIndex(text_preprocessor)

    match args.command:
        case "search":
            result = search_command(args.query, text_preprocessor, inverted_index)
            for idx, movie in enumerate(result):
                print(f"{idx + 1}. {movie.title}")
        case "build":
            build_command(inverted_index)
        case "tf":
            doc_id = args.doc_id
            term = args.term
            term_token = text_preprocessor.tokenize_single_term(term)

            tf = tf_command(inverted_index, doc_id, term_token)
            print(tf)
        case "idf":
            term = args.term
            term_token = text_preprocessor.tokenize_single_term(term)

            idf = idf_command(inverted_index, term_token)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            doc_id = args.doc_id
            term = args.term
            term_token = text_preprocessor.tokenize_single_term(term)

            tf_idf = tfidf_command(inverted_index, doc_id, term_token)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            term = args.term
            term_token = text_preprocessor.tokenize_single_term(term)

            bm25idf = bm25_idf_command(inverted_index, term_token)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            doc_id = args.doc_id
            term = args.term
            k1 = args.k1
            b = args.b
            term_token = text_preprocessor.tokenize_single_term(term)

            bm25tf = bm25_tf_command(inverted_index, doc_id, term_token, k1, b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            query = args.query
            limit = args.limit

            result = bm25_search(inverted_index, query, limit)
            for movie, score in result:
                print(f"({movie.id}) {movie.title} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
