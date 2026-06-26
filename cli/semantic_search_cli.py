import argparse

from lib.semantic_search import (
    SemanticSearch,
    chunk_doc,
    embed_query,
    embed_text,
    verify_embeddings,
    verify_model,
)

from data import load_movies


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
    search_parser.add_argument("--limit", type=int, nargs="?", default=5)

    chunk_parser = subparsers.add_parser("chunk", help="Chunk long documents")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200)
    chunk_parser.add_argument("--overlap", type=int, nargs="?")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            text = args.text
            embed_text(text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            query = args.query
            embed_query(query)
        case "search":
            query = args.query
            limit = args.limit
            semantic_search = SemanticSearch()
            documents = load_movies()
            semantic_search.load_or_create_embeddings(documents)
            result = semantic_search.search(query, limit)
            for e in result:
                print(f"{e['title']}: {e['description']} ({e['score']})")
        case "chunk":
            text = args.text
            chunk_size = args.chunk_size
            overlap = args.overlap
            chunks = chunk_doc(text, chunk_size, overlap)
            print(f"Chunking {len(text)} characters")
            for idx, chunk in enumerate(chunks):
                print(f"{idx + 1}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
