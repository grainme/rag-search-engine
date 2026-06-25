import argparse

from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Print the embedding model information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str)

    subparsers.add_parser("verify_embeddings", help="Embed text")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            text = args.text
            embed_text(text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
