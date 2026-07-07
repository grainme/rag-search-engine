import argparse
import json
from pathlib import Path

from lib.evaluation import f1_score, precision_at_k, recall_at_k
from lib.hybrid_search import HybridSearch

from data import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    hybrid_search = HybridSearch(load_movies())

    with Path("data/golden_dataset.json").open(encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"k={limit}")
    for tt in golden_data["test_cases"]:
        query = tt["query"]
        relevant_docs = tt["relevant_docs"]

        rrf_search_results = hybrid_search.rrf_search(query, 60, limit)
        retrieved_docs = [res["doc"].title for res in rrf_search_results]

        precision_metric = precision_at_k(retrieved_docs, relevant_docs)
        recall_metric = recall_at_k(retrieved_docs, relevant_docs)
        f1_metric = f1_score(retrieved_docs, relevant_docs)

        print(
            f"- Query: {query}\n  - Precision@{limit}: {precision_metric:.4f}\n  - Recall@{limit}: {recall_metric:.4f}\n  - F1 Score: {f1_metric:.4f}\n  - Retrieved: {', '.join([doc for doc in retrieved_docs])}\n  - Relevant: {', '.join(relevant_docs)}\n"
        )


if __name__ == "__main__":
    main()
