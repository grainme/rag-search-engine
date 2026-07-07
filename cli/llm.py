import json
import os
from time import sleep

from dotenv import load_dotenv
from models import EnhanceMethod, Movie, RRFSearchResult
from openai import OpenAI
from openai.types.chat import ChatCompletion

_ = load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MAX_RERANK_ATTEMPTS = 3


def enhance_query(enhance: EnhanceMethod, query: str) -> str:
    match enhance:
        case "spell":
            return fix_spell(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand(query)
        case _:
            return query


def fix_spell(query: str) -> str:
    response: ChatCompletion = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
                """,
            }
        ],
    )

    assert response.choices[0].message.content is not None
    return response.choices[0].message.content


def rewrite_query(query: str) -> str:
    response: ChatCompletion = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """,
            }
        ],
    )

    assert response.choices[0].message.content is not None
    return response.choices[0].message.content


def expand(query: str) -> str:
    response: ChatCompletion = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": f"""Expand the user-provided movie search query below with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}"
                """,
            }
        ],
    )

    assert response.choices[0].message.content is not None
    return response.choices[0].message.content


def rerank_individual(query: str, doc: Movie) -> float:
    last_error: Exception | None = None

    for attempt in range(MAX_RERANK_ATTEMPTS):
        try:
            response: ChatCompletion = client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
                max_tokens=10,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.title or ""} - {doc.description or ""}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:""",
                    }
                ],
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("empty response")

            result = float(content.strip())
            if result < 0 or result > 10:
                raise ValueError(f"score must be between 0 and 10: {result}")

            return result
        except Exception as e:
            last_error = e
            if attempt < MAX_RERANK_ATTEMPTS - 1:
                sleep(3)

    raise RuntimeError("Score generated by the LLM is not a NUMBER:\n ", last_error)


def rerank_batch(query: str, docs: list[RRFSearchResult]) -> list[int]:
    last_error: Exception | None = None

    for attempt in range(MAX_RERANK_ATTEMPTS):
        try:
            doc_list_str = "\n".join(
                f"{entry['doc'].id}: {entry['doc'].title} - {entry['doc'].description}"
                for entry in docs
            )

            response: ChatCompletion = client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
                max_tokens=256,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Rank the movies listed below by relevance to the following search query.

                        Query: "{query}"

                        Movies:
                        {doc_list_str}

                        Return the movie IDs in order of relevance, best match first.

                        Your response must be a raw JSON array of integers.
                        Do not wrap the JSON in Markdown. Do not use a ```json code block.
                        Do not include any explanatory text.

                        For example:
                        [75, 12, 34, 2, 1]

                        Ranking:""",
                    }
                ],
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("empty response")

            return json.loads(content.strip())
        except Exception as e:
            last_error = e
            if attempt < MAX_RERANK_ATTEMPTS - 1:
                sleep(3)

    raise RuntimeError(
        "Rankings generated by the LLM are not a JSON array of integers:\n ",
        last_error,
    )


def evaluate_rrf_results(query: str, results: list) -> list:
    formatted_results = "\n".join([res["doc"].title for res in results])

    response: ChatCompletion = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        max_tokens=256,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(formatted_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers other than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]""",
            }
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("empty response")

    return json.loads(content.strip())
