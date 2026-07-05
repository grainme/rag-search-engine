import argparse

from constants import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from lib.keyword_search import (
    bm25_idf,
    bm25_keyword_search,
    bm25_tf,
    build_index,
    inverse_document_frequency,
    keyword_search,
    term_frequency,
    tfidf_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT
    )

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
        "--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            keyword_search(args.query, args.limit)
        case "build":
            build_index()
        case "tf":
            term_frequency(args.doc_id, args.term)
        case "idf":
            inverse_document_frequency(args.term)
        case "tfidf":
            tfidf_score(args.doc_id, args.term)
        case "bm25idf":
            bm25_idf(args.term)
        case "bm25tf":
            bm25_tf(args.doc_id, args.term, args.k1, args.b)
        case "bm25search":
            bm25_keyword_search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
