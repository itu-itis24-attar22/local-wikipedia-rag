"""Retrieval tests."""

from app.retriever import build_retrieval_plan


def test_named_entity_retrieval_filters():
    plan = build_retrieval_plan("Who was Albert Einstein?")
    assert {"entity_name": "Albert Einstein"} in [search["where"] for search in plan.searches]

    plan = build_retrieval_plan("Where is Hagia Sophia located?")
    assert {"entity_name": "Hagia Sophia"} in [search["where"] for search in plan.searches]


def test_comparison_retrieval_filters_both_entities():
    plan = build_retrieval_plan("Compare Albert Einstein and Nikola Tesla.")
    filters = [search["where"] for search in plan.searches]
    assert {"entity_name": "Albert Einstein"} in filters
    assert {"entity_name": "Nikola Tesla"} in filters


def test_keyword_hint_retrieval_filters():
    plan = build_retrieval_plan("Which famous place is located in Turkey?")
    assert {"entity_name": "Hagia Sophia"} in [search["where"] for search in plan.searches]

    plan = build_retrieval_plan("Which person is associated with electricity?")
    assert {"entity_name": "Nikola Tesla"} in [search["where"] for search in plan.searches]
