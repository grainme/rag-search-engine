from pydantic import BaseModel


class Movie(BaseModel):
    id: int
    title: str
    description: str


class SearchResponse(BaseModel):
    movies: list[Movie]
