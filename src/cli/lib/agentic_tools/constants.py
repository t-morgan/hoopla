"""Shared constants for agentic search tools."""

# Canonical genre → list of synonyms/related phrases
GENRE_SYNONYMS: dict[str, list[str]] = {
    "horror": ["horror", "scary", "terrifying", "frightening"],
    "adventure": ["adventure", "expedition", "journey"],
    "drama": ["drama", "dramatic", "melodrama"],
    "comedy": ["comedy", "funny", "hilarious", "humor"],
    "thriller": ["thriller", "suspense", "tense", "edge of your seat"],
    "action": ["action", "fighting", "combat", "martial arts"],
    "romance": ["romance", "romantic", "love story", "love"],
    "fantasy": ["fantasy", "magical", "wizard", "sorcery"],
    "sci-fi": [
        "sci-fi",
        "science fiction",
        "sci fi",
        "futuristic",
        "space",
        "space opera",
    ],
    "animation": ["animation", "animated", "cartoon"],
    "family": ["family", "kids", "family-friendly"],
    "mystery": ["mystery", "whodunit", "detective story"],
}

