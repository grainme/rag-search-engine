import argparse

from lib.hybrid_search import normalize, rrf_search, weighted_search


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

    rrf_search_parser = subparser.add_parser("rrf-search", help="RRF search with configurable k")
    rrf_search_parser.add_argument("query", type=str)
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        nargs="?",
        default=60,
        help="RRF k parameter controlling weight distribution (default=60)",
    )
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=5)
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.inputs)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
