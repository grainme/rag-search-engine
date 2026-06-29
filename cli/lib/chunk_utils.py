import re


#
# Low level helpers
#
def chunk_doc(text: str, chunk_size: int, overlap: int) -> list[str]:
    word_chunks = _chunk_items(text.split(), chunk_size, overlap)
    return [" ".join(words) for words in word_chunks]


# this is not semantic (the ML meaning)
def semantic_chunk_doc(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    chunks = _chunk_items(sentences, chunk_size, overlap)
    return [" ".join(sentences) for sentences in chunks]


def _split_sentences(text: str) -> list[str]:
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence]


def _chunk_items(items: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[list[str]] = []

    for start in range(0, len(items), step):
        chunk = items[start : start + chunk_size]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(chunk)

    return chunks
