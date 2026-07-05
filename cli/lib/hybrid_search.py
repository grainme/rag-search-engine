from lib.semantic_search import ChunkedSemanticSearch, SearchResult
from models import Movie
from preprocessing import create_inverted_index

from data import load_movies


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        # semantic search
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        # keyword search
        self.inverted_index = create_inverted_index()
        self.inverted_index.load_or_build()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[Movie, float]]:
        return self.inverted_index.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        if not self.semantic_search.documents:
            raise ValueError("No documents loaded.")

        bm25_results: list[tuple[Movie, float]] = self._bm25_search(query, 500 * limit)
        semantic_search_results: list[SearchResult] = self.semantic_search.search_chunks(
            query, 500 * limit
        )

        bm25_scores_normalized = (
            _normalize([score for _, score in bm25_results]) if bm25_results else []
        )
        semantic_scores_normalized = (
            _normalize([result["score"] for result in semantic_search_results])
            if semantic_search_results
            else []
        )

        bm25_scores_by_doc_id = {
            movie.id: bm25_scores_normalized[idx] for idx, (movie, _) in enumerate(bm25_results)
        }
        semantic_scores_by_doc_id = {
            result["id"]: semantic_scores_normalized[idx]
            for idx, result in enumerate(semantic_search_results)
        }

        doc_scores: dict[int, dict] = {}
        for doc_id in set(bm25_scores_by_doc_id) | set(semantic_scores_by_doc_id):
            movie = self.semantic_search.document_map[doc_id]
            bm25_score = bm25_scores_by_doc_id.get(doc_id, 0)
            semantic_score = semantic_scores_by_doc_id.get(doc_id, 0)
            hybrid_score = _hybrid_score(bm25_score, semantic_score, alpha)

            doc_scores[doc_id] = {
                "doc": movie,
                "keyword_score": bm25_score,
                "semantic_score": semantic_score,
                "hybrid_score": hybrid_score,
            }

        results = sorted(
            list(doc_scores), key=lambda k: doc_scores[k]["hybrid_score"], reverse=True
        )

        return [doc_scores[k] for k in results]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implmented yet.")


def _hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    # the choice of alpha depends on the query type (exact match, conceptual, mixed)
    return alpha * bm25_score + (1 - alpha) * semantic_score


def _normalize(values: list) -> list[float]:
    values = list(map(float, values))
    max_val = max(values)
    min_val = min(values)
    if min_val == max_val:
        # print(*(f"* {1:.4f}" for _ in range(len(values))), sep="\n")
        return [1.000 for _ in range(len(values))]

    # min-max normalization
    values = list(map(lambda x: (x - min_val) / (max_val - min_val), values))
    # print(*(f"* {score:.4f}" for score in values), sep="\n")

    return values


## FOR COMMANDS
def weighted_search(query: str, alpha: float, limit: int) -> None:
    hybrid_search = HybridSearch(load_movies())

    results = hybrid_search.weighted_search(query, alpha, limit)

    for idx, entry in enumerate(results[:limit]):
        doc: Movie = entry["doc"]
        hybrid_score = entry["hybrid_score"]
        semantic_score = entry["semantic_score"]
        keyword_score = entry["keyword_score"]

        print(
            f"{idx}. {doc.title}\nHybrid Score: {hybrid_score}\nBM25: {keyword_score}, Semantic: {semantic_score}\n{doc.description[:10]}"
        )


def normalize(values: list) -> None:
    values = list(map(float, values))
    max_val = max(values)
    min_val = min(values)
    if min_val == max_val:
        print(*(f"* {1:.4f}" for _ in range(len(values))), sep="\n")

    # min-max normalization
    values = list(map(lambda x: (x - min_val) / (max_val - min_val), values))
    print(*(f"* {score:.4f}" for score in values), sep="\n")
