from lib.hybrid_search import HybridSearch
from llm import ask_llm

from data import load_movies


def rag(query: str):
    docs = load_movies()
    hybrid_search = HybridSearch(docs)

    results = hybrid_search.rrf_search(query, limit=5)
    retrieved_docs = [
        f"[{idx}] - {res['doc'].title} : {res['doc'].description}\n"
        for idx, res in enumerate(results, start=1)
    ]

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {retrieved_docs}

    Answer:"""

    print("Search Results:")
    for doc in retrieved_docs:
        # this is a workaround: ugly code but okay :)
        print(f"- {doc.split('-')[0].strip()}")

    llm_response = ask_llm(prompt)
    print("RAG Response:\n", llm_response)


def summarize(query: str):
    docs = load_movies()
    hybrid_search = HybridSearch(docs)

    results = hybrid_search.rrf_search(query, limit=5)
    retrieved_docs = [
        f"[{idx}] - {res['doc'].title} : {res['doc'].description}\n"
        for idx, res in enumerate(results, start=1)
    ]

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {retrieved_docs}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    print("Search Results:")
    for doc in retrieved_docs:
        # this is a workaround: ugly code but okay :)
        print(f"- {doc.split('-')[0].strip()}")

    llm_response = ask_llm(prompt)
    print("LLM Summary::\n", llm_response)


def citations(query: str):
    docs = load_movies()
    hybrid_search = HybridSearch(docs)

    results = hybrid_search.rrf_search(query, limit=5)
    retrieved_docs = [
        f"[{idx}] - {res['doc'].title} : {res['doc'].description}\n"
        for idx, res in enumerate(results, start=1)
    ]

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Hoopla, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {retrieved_docs}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    print("Search Results:")
    for doc in retrieved_docs:
        # this is a workaround: ugly code but okay :)
        print(f"- {doc.split(':')[0].strip()}")

    llm_response = ask_llm(prompt)
    print("LLM Answer:\n", llm_response)


def question(question: str):
    docs = load_movies()
    hybrid_search = HybridSearch(docs)

    results = hybrid_search.rrf_search(question, limit=5)
    retrieved_docs = [
        f"[{idx}] - {res['doc'].title} : {res['doc'].description}\n"
        for idx, res in enumerate(results, start=1)
    ]

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

    Question: {question}

    Documents:
    {retrieved_docs}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""

    print("Search Results:")
    for doc in retrieved_docs:
        # this is a workaround: ugly code but okay :)
        print(f"- {doc.split(':')[0].strip()}")

    llm_response = ask_llm(prompt)
    print("Answer:\n", llm_response)
