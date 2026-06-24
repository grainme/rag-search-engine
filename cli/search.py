from .models import Movie
from .preprocessing import TextPreprocessor


class KeywordSearchEngine:
    def __init__(
        self,
        movies: list[Movie],
        text_preprocessor: TextPreprocessor,
    ) -> None:
        self._movies = movies
        self._text_preprocessor = text_preprocessor

    def search(self, query: str, limit: int = 50) -> list[str]:
        search_result = [f"Searching for: {query}"]

        for movie in self._movies:
            if self._text_preprocessor.has_substring_token_match(query, movie.title):
                search_result.append(movie.title)

        return search_result[:limit]
