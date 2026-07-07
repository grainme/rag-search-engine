from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel

EnhanceMethod = Literal["spell", "rewrite", "expand"] | None
RerankMethod = Literal["individual", "batch", "cross_encoder"] | None


class Movie(BaseModel):
    id: int
    title: str
    description: str


class SearchResponse(BaseModel):
    movies: list[Movie]


class WeightedSearchResult(TypedDict):
    doc: Movie
    keyword_score: float
    semantic_score: float
    hybrid_score: float


class RRFSearchResult(TypedDict):
    doc: Movie
    bm25_rank: int
    semantic_rank: int
    rrf_score: float
    re_rank_score: NotRequired[float]
    re_rank_rank: NotRequired[int]
    cross_encoder_score: NotRequired[float]
