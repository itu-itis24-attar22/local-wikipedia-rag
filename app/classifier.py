"""Rule-based person/place query classifier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.config import ENTITIES_PATH, load_entities_config


QueryType = Literal["person", "place", "both", "unknown"]

COMPARISON_WORDS = {
    "compare",
    "compared",
    "comparison",
    "vs",
    "versus",
    "difference between",
    "differences between",
    "similarities between",
}

PERSON_KEYWORDS = {
    "who",
    "person",
    "people",
    "scientist",
    "artist",
    "writer",
    "poet",
    "inventor",
    "discover",
    "discovered",
    "born",
    "died",
    "footballer",
    "singer",
    "painter",
    "president",
}

PLACE_KEYWORDS = {
    "where",
    "place",
    "places",
    "located",
    "location",
    "country",
    "city",
    "landmark",
    "monument",
    "tower",
    "museum",
    "wall",
    "mount",
    "mountain",
    "canyon",
    "used for",
}

KEYWORD_ENTITY_HINTS = {
    "electricity": ["Nikola Tesla"],
    "electrical": ["Nikola Tesla"],
    "alternating current": ["Nikola Tesla"],
    "turkey": ["Hagia Sophia"],
    "istanbul": ["Hagia Sophia"],
}

OFF_DOMAIN_PATTERNS = {
    "president of mars",
    "john doe",
    "random unknown person",
}


@dataclass(frozen=True)
class Classification:
    query_type: QueryType
    matched_people: list[str]
    matched_places: list[str]
    hinted_entities: list[str]
    is_comparison: bool
    force_unknown: bool = False

    @property
    def matched_entities(self) -> list[str]:
        return self.matched_people + self.matched_places + self.hinted_entities


def normalize_query(query: str) -> str:
    """Normalize a query for rule matching."""
    return re.sub(r"\s+", " ", query.lower()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower())
    return re.search(rf"(?<!\w){escaped}(?!\w)", text) is not None


def _entity_names() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = load_entities_config(ENTITIES_PATH)
    return data["people"], data["places"]


def _find_matches(query_normalized: str, entities: list[dict[str, object]]) -> list[str]:
    matches: list[str] = []
    for entity in entities:
        name = str(entity["name"])
        candidates = [name, *[str(alias) for alias in entity.get("aliases", [])]]
        if any(_contains_phrase(query_normalized, candidate) for candidate in candidates):
            matches.append(name)
    return matches


def _keyword_hints(query_normalized: str) -> list[str]:
    hints: list[str] = []
    for keyword, entities in KEYWORD_ENTITY_HINTS.items():
        if _contains_phrase(query_normalized, keyword):
            hints.extend(entities)
    return list(dict.fromkeys(hints))


def _has_any_keyword(query_normalized: str, keywords: set[str]) -> bool:
    return any(_contains_phrase(query_normalized, keyword) for keyword in keywords)


def classify_query(query: str) -> Classification:
    """Classify a user query as person, place, both, or unknown."""
    query_normalized = normalize_query(query)
    force_unknown = any(pattern in query_normalized for pattern in OFF_DOMAIN_PATTERNS)
    people, places = _entity_names()
    matched_people = _find_matches(query_normalized, people)
    matched_places = _find_matches(query_normalized, places)
    hints = _keyword_hints(query_normalized)
    is_comparison = any(_contains_phrase(query_normalized, word) for word in COMPARISON_WORDS)

    hinted_people = [name for name in hints if any(person["name"] == name for person in people)]
    hinted_places = [name for name in hints if any(place["name"] == name for place in places)]

    if force_unknown and not matched_people and not matched_places:
        return Classification(
            query_type="unknown",
            matched_people=[],
            matched_places=[],
            hinted_entities=[],
            is_comparison=is_comparison,
            force_unknown=True,
        )

    all_people = list(dict.fromkeys(matched_people + hinted_people))
    all_places = list(dict.fromkeys(matched_places + hinted_places))

    if all_people and all_places:
        query_type: QueryType = "both"
    elif all_people:
        query_type = "person"
    elif all_places:
        query_type = "place"
    else:
        person_clue = _has_any_keyword(query_normalized, PERSON_KEYWORDS)
        place_clue = _has_any_keyword(query_normalized, PLACE_KEYWORDS)
        if person_clue and place_clue:
            query_type = "both"
        elif person_clue:
            query_type = "person"
        elif place_clue:
            query_type = "place"
        else:
            query_type = "both"

    if is_comparison and not (all_people or all_places):
        query_type = "both"

    return Classification(
        query_type=query_type,
        matched_people=matched_people,
        matched_places=matched_places,
        hinted_entities=hints,
        is_comparison=is_comparison,
        force_unknown=False,
    )
