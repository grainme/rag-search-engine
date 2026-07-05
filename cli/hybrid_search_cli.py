import argparse

from lib.hybrid_search import normalize, weighted_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize values (range 0-1)")
    normalize_parser.add_argument("inputs", nargs="*")

    weighted_search_parser = subparser.add_parser(
        "weighted-search", help="Hybrid search with configurable alpha"
    )
    weighted_search_parser.add_argument("query", type=str)
    weighted_search_parser.add_argument("--alpha", type=float, nargs="?", default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, nargs="?", default=5)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.inputs)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
