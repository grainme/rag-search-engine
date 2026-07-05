import math
import string
from collections import Counter

from constants import BM25_B, BM25_K1
from models import Movie
from nltk.stem import PorterStemmer

from data import (
    load_movies,
    load_stop_words,
    read_doc_length,
    read_docmap,
    read_index,
    read_term_frequencies,
    write_doc_length,
    write_docmap,
    write_index,
    write_term_frequencies,
)


class TextPreprocessor:
    def __init__(self, stop_words: set[str]) -> None:
        self._stop_words = {normalize_text(word) for word in stop_words}
        self._stemmer = PorterStemmer()

    def tokenize(self, text: str) -> list[str]:
        tokens = normalize_text(text).split()
        return [
            self._stemmer.stem(token)
            for token in tokens
            if token not in self._stop_words
        ]

    def has_substring_token_match(self, query: str, title: str) -> bool:
        query_tokens = self.tokenize(query)
        title_tokens = self.tokenize(title)
        return any(
            query_token in title_token
            for query_token in query_tokens
            for title_token in title_tokens
        )

    def tokenize_single_term(self, term: str) -> str:
        tokens = self.tokenize(term)
        if len(tokens) == 0:
            raise ValueError(f"{term} tokenized to no terms")
        if len(tokens) > 1:
            raise ValueError(f"{term} tokenized to multiple terms")
        return tokens[0]


class InvertedIndex:
    def __init__(self, text_preprocessor: TextPreprocessor) -> None:
        # mapping tokens to sets of docs IDs
        self.index: dict[str, set[int]] = {}
        # mapping docs IDs to their full doc object
        self.docmap: dict[int, Movie] = {}
        self.text_preprocessor: TextPreprocessor = text_preprocessor
        self.term_frequencies: Counter = Counter()
        self.doc_lengths: dict[int, int] = {}

    def save(self):
        write_index(self.index)
        write_docmap(self.docmap)
        write_term_frequencies(self.term_frequencies)
        write_doc_length(self.doc_lengths)

    def load(self):
        self.index = read_index()
        self.docmap = read_docmap()
        self.term_frequencies = read_term_frequencies()
        self.doc_lengths = read_doc_length()

    def load_or_build(self) -> None:
        try:
            self.load()
        except FileNotFoundError:
            self.build()
            self.save()

    def build(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = Counter()
        self.doc_lengths = {}

        movies = load_movies()
        for movie in movies:
            doc_id = movie.id
            text = f"{movie.title} {movie.description}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, text)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term)
        if not doc_ids:
            return []
        return sorted(doc_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get((doc_id, term)) or 0

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        # log is "needed" because the inner field can be huge!
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        # BM25 formula: log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    # ---- b is the normalization param (tunable prameter)
    # - BM25 saturation formula: (tf * (k1 + 1) / (tf + k1)) (before length normalization for biggg docs)
    # - Length normalization factor: length_norm = 1 - b + b * (doc_length / avg_doc_length)
    # > Applied to term frequency: tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id) or 0

        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        saturated_tf_score = tf * (k1 + 1) / (tf + k1 * length_norm)
        return saturated_tf_score

    def bm25(self, doc_id: int, term: str):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_idf * bm25_tf

    def bm25_search(self, query: str, limit: int) -> list[tuple[Movie, float]]:
        query_tokens = self.text_preprocessor.tokenize(query)
        doc_bm25_score: dict[int, float] = {}

        for qtoken in query_tokens:
            for doc_id in self.index.get(qtoken, []):
                # breakpoint()
                if doc_id not in doc_bm25_score:
                    doc_bm25_score[doc_id] = 0
                doc_bm25_score[doc_id] += self.bm25(doc_id, qtoken)

        docs_id_sorted = sorted(
            doc_bm25_score, key=lambda doc_id: doc_bm25_score[doc_id], reverse=True
        )

        result: list[tuple[Movie, float]] = []
        for doc_id in docs_id_sorted:
            if len(result) >= limit:
                break
            movie = self.docmap[doc_id]
            movie_score = doc_bm25_score.get(doc_id, 0)
            result.append((movie, movie_score))
        return result

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def __add_document(self, doc_id: int, text: str) -> None:
        ttokens = self.text_preprocessor.tokenize(text)
        self.doc_lengths[doc_id] = len(ttokens)
        for tt in ttokens:
            self.index.setdefault(tt, set())
            self.index[tt].add(doc_id)
            self.term_frequencies[(doc_id, tt)] += 1


def remove_punctuation(text: str) -> str:
    translation_table = text.maketrans("", "", string.punctuation)
    return text.translate(translation_table)


def normalize_text(text: str) -> str:
    return remove_punctuation(text.lower())


def create_inverted_index() -> InvertedIndex:
    stop_words = load_stop_words()
    text_preprocessor = TextPreprocessor(stop_words)
    return InvertedIndex(text_preprocessor)
