"""Classifier tests."""

from app.classifier import classify_query


def test_required_classifier_examples():
    cases = {
        "Who was Albert Einstein?": ("person", ["Albert Einstein"]),
        "Where is the Eiffel Tower located?": ("place", ["Eiffel Tower"]),
        "Compare Einstein and Tesla": ("person", ["Albert Einstein", "Nikola Tesla"]),
        "Compare Eiffel Tower and Statue of Liberty": (
            "place",
            ["Eiffel Tower", "Statue of Liberty"],
        ),
        "Which famous place is located in Turkey?": ("place", ["Hagia Sophia"]),
        "Which person is associated with electricity?": ("person", ["Nikola Tesla"]),
        "Who is the president of Mars?": ("unknown", []),
    }

    for query, (expected_type, expected_entities) in cases.items():
        result = classify_query(query)
        assert result.query_type == expected_type
        for entity in expected_entities:
            assert entity in result.matched_entities
