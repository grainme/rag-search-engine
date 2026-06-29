from typing import TypedDict

from pydantic import BaseModel


class Movie(BaseModel):
    id: int
    title: str
    description: str


class SearchResponse(BaseModel):
    movies: list[Movie]


# class ChunkMetadata(TypedDict, total=False):
     # the index of the doc in self.documents
#     movie_idx: int
     # the index of the chunk within the doc
#     chunk_idx: int
     # the total number of chunks in the doc
#     total_chunks: int
