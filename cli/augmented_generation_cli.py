import argparse

from lib.augmented_generation import citations, question, rag, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Multi-document summarization pipeline that synthesizes search results"
    )
    summarize_parser.add_argument("query", type=str)
    summarize_parser.add_argument("--limit", type=int, nargs="?", default=5)

    citations_parser = subparsers.add_parser(
        "citations", help="Citation-aware answer command that references its sources"
    )
    citations_parser.add_argument("query", type=str)
    citations_parser.add_argument("--limit", type=int, nargs="?", default=5)

    question_parser = subparsers.add_parser(
        "question", help="Question-answering command for direct responses."
    )
    question_parser.add_argument("question", type=str)
    question_parser.add_argument("--limit", type=int, nargs="?", default=5)

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag(args.query)
        case "summarize":
            summarize(args.query)
        case "citations":
            citations(args.query)
        case "question":
            question(args.question)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
