from time import sleep

from constants import DOCUMENT_PREVIEW_LENGTH
from lib.semantic_search import ChunkedSemanticSearch, SearchResult
from models import (
    EnhanceMethod,
    Movie,
    RerankMethod,
    RRFSearchResult,
    WeightedSearchResult,
)
from preprocessing import create_inverted_index
from sentence_transformers import CrossEncoder
from test_llm import enhance_query, rerank_batch, rerank_individual

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

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[WeightedSearchResult]:
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

        doc_scores: dict[int, WeightedSearchResult] = {}
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

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[RRFSearchResult]:
        bm25_results: list[tuple[Movie, float]] = self._bm25_search(query, 500 * limit)
        semantic_search_results: list[SearchResult] = self.semantic_search.search_chunks(
            query, 500 * limit
        )

        bm25_results_sorted = sorted(bm25_results, key=lambda t: t[1], reverse=True)
        semantic_search_results_sorted = sorted(
            semantic_search_results, key=lambda x: x["score"], reverse=True
        )

        bm25_ranks_by_doc_id = {
            movie.id: idx + 1 for idx, (movie, _) in enumerate(bm25_results_sorted)
        }
        semantic_ranks_by_doc_id = {
            result["id"]: idx + 1 for idx, result in enumerate(semantic_search_results_sorted)
        }

        doc_scores: dict[int, RRFSearchResult] = {}
        for doc_id in set(bm25_ranks_by_doc_id) & set(semantic_ranks_by_doc_id):
            movie = self.semantic_search.document_map[doc_id]
            bm25_rank = bm25_ranks_by_doc_id[doc_id]
            semantic_rank = semantic_ranks_by_doc_id[doc_id]

            rrf_bm25_rank = _rrf_score(bm25_rank, k)
            rrf_semantic_rank = _rrf_score(semantic_rank, k)
            rrf_score = rrf_bm25_rank + rrf_semantic_rank

            doc_scores[doc_id] = {
                "doc": movie,
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "rrf_score": rrf_score,
            }

        results = sorted(list(doc_scores), key=lambda k: doc_scores[k]["rrf_score"], reverse=True)

        return [doc_scores[k] for k in results]


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def _hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def _normalize(values: list) -> list[float]:
    values = list(map(float, values))
    max_val = max(values)
    min_val = min(values)
    if min_val == max_val:
        return [1.000 for _ in range(len(values))]

    # min-max normalization
    values = list(map(lambda x: (x - min_val) / (max_val - min_val), values))

    return values


## FOR COMMANDS


def rrf_search(
    query: str,
    k: int,
    limit: int,
    enhance: EnhanceMethod,
    rerank_method: RerankMethod,
) -> None:
    output_limit = limit

    if enhance:
        pre_query = query
        query = enhance_query(enhance, pre_query)
        print(f"Enhanced query ({enhance}): '{pre_query}' -> '{query}'\n")

    if rerank_method:
        limit = 5 * limit
        print(f"Re-ranking top {output_limit} results using {rerank_method} method...")

    hybrid_search = HybridSearch(load_movies())

    results: list[RRFSearchResult] = hybrid_search.rrf_search(query, k, limit)[:limit]

    if rerank_method:
        if rerank_method == "individual":
            results_reranked: list[RRFSearchResult] = []
            for res in results:
                reranked = res.copy()
                reranked["re_rank_score"] = rerank_individual(query, res["doc"])
                results_reranked.append(reranked)
                sleep(3)

            results = sorted(
                results_reranked,
                key=lambda x: x.get("re_rank_score", 0.0),
                reverse=True,
            )
        elif rerank_method == "batch":
            ranked_movie_ids = rerank_batch(query, results)
            ranks_by_doc_id: dict[int, int] = {
                movie_id: rank for rank, movie_id in enumerate(ranked_movie_ids, start=1)
            }

            # if the LLM forgot to return some movie ID, give it a bad fallback rank
            fallback_rank = len(results) + 1
            results_reranked: list[RRFSearchResult] = []
            for res in results:
                reranked = res.copy()
                reranked["re_rank_rank"] = ranks_by_doc_id.get(res["doc"].id, fallback_rank)
                results_reranked.append(reranked)

            results = sorted(
                results_reranked,
                key=lambda x: x.get("re_rank_rank", fallback_rank),
            )
        elif rerank_method == "cross_encoder":
            pairs: list[tuple[str, str]] = [
                (query, f"{res['doc'].title} - {res['doc'].description}") for res in results
            ]
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            # `predict` returns a list of numbers, one for each pair
            scores = cross_encoder.predict(pairs)

            results_reranked: list[RRFSearchResult] = []
            for idx, res in enumerate(results):
                reranked = res.copy()
                reranked["cross_encoder_score"] = float(scores[idx])
                results_reranked.append(reranked)

            results = sorted(
                results_reranked, key=lambda x: x.get("cross_encoder_score", 0), reverse=True
            )

    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
    for idx, entry in enumerate(results[:output_limit], start=1):
        doc: Movie = entry["doc"]
        rrf_score = entry["rrf_score"]
        semantic_rank = entry["semantic_rank"]
        bm25_rank = entry["bm25_rank"]
        re_rank_score = entry.get("re_rank_score")
        re_rank_rank = entry.get("re_rank_rank")
        cross_encoder_score = entry.get("cross_encoder_score")

        print(f"{idx}. {doc.title}")
        if cross_encoder_score is not None:
            print(f"   Cross Encoder Score: {cross_encoder_score:.3f}")
        if re_rank_score is not None:
            print(f"   Re-rank Score: {re_rank_score:.3f}/10")
        if re_rank_rank is not None:
            print(f"   Re-rank Rank: {re_rank_rank}")
        print(f"   RRF Score: {rrf_score:.3f}")
        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
        print(f"   {doc.description[:DOCUMENT_PREVIEW_LENGTH]}\n")


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
