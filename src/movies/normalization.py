from typing import TypedDict, List, Dict, Optional

class Movie(TypedDict):
    id: int
    title: str
    description: str
    cast: List[str]
    genre: List[str]  # list of genre names


def normalize_movie_from_tmdb(details: Dict, enriched_description: Optional[str] = None) -> Movie:
    """
    Convert a TMDB /movie/{id}?append_to_response=credits,external_ids response into our Movie schema.
    If enriched_description is provided and non-empty, use it as description.
    Otherwise, fallback to TMDB's overview.
    - id: details["id"]
    - title: details["title"] or details["original_title"]
    - description: enriched_description or details["overview"] or ""
    - cast: top N cast names from details["credits"]["cast"], e.g., first 5 entries' "name"
    - genre: list of all genre names from details["genres"], or [] if missing
    """
    movie_id = details.get("id", -1)
    title = details.get("title") or details.get("original_title") or "Unknown"
    title = title.strip() if isinstance(title, str) else "Unknown"
    # Description: prefer enriched_description if provided and non-empty
    description = (enriched_description or "").strip()
    if not description:
        description = details.get("overview", "")
        description = description.strip() if isinstance(description, str) else ""
    # Cast
    cast_list = []
    credits = details.get("credits", {})
    cast_entries = credits.get("cast", [])
    for entry in cast_entries[:5]:
        name = entry.get("name", "").strip()
        if name:
            cast_list.append(name)
    # Genre: list of all genre names
    genres = [g["name"].strip() for g in details.get("genres", []) if g.get("name") and isinstance(g["name"], str)]
    return Movie(
        id=movie_id,
        title=title,
        description=description,
        cast=cast_list,
        genre=genres,
    )

# Example doctest/unit test
if __name__ == "__main__":
    fake_details = {
        "id": 123,
        "title": "Paddington ",
        "overview": "A cute British bear loves marmalade.",
        "credits": {
            "cast": [
                {"name": "Ben Whishaw "},
                {"name": "Hugh Bonneville"},
                {"name": "Sally Hawkins"},
                {"name": "Julie Walters"},
                {"name": "Jim Broadbent"},
                {"name": "Extra Actor"}
            ]
        },
        "genres": [
            {"id": 35, "name": "Comedy "},
            {"id": 10751, "name": "Family"}
        ]
    }
    normalized = normalize_movie_from_tmdb(fake_details)
    print(normalized)
    # Expected output:
    # {'id': 123, 'title': 'Paddington', 'description': 'A cute British bear loves marmalade.',
    #  'cast': ['Ben Whishaw', 'Hugh Bonneville', 'Sally Hawkins', 'Julie Walters', 'Jim Broadbent'],
    #  'genre': ['Comedy', 'Family']}
