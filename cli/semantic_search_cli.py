import argparse

from constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
)
from lib.chunking import chunk_text, embed_chunks, search_chunked, semantic_chunk_text
from lib.semantic_search import (
    embed_query,
    embed_text,
    semantic_search,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Print the embedding model information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str)

    subparsers.add_parser("verify_embeddings", help="Embed text")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed Query")
    embed_query_parser.add_argument("query", type=str)

    search_parser = subparsers.add_parser("search", help="Search semantically")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument(
        "--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk long documents")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument(
        "--chunk-size", type=int, nargs="?", default=DEFAULT_CHUNK_SIZE
    )
    chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=DEFAULT_CHUNK_OVERLAP
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunk long documents"
    )
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, nargs="?", default=DEFAULT_SEMANTIC_CHUNK_SIZE
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=DEFAULT_CHUNK_OVERLAP
    )

    subparsers.add_parser("embed_chunks", help="Embed chunks")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search that queries chunk embeddings and aggregates results",
    )
    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument(
        "--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
