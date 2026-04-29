"""Chunker tests."""

from app.chunker import RawSection, chunk_sections


def test_chunk_sections_splits_large_text_with_overlap():
    words = [f"word{i}" for i in range(25)]
    chunks = chunk_sections(
        [RawSection(title="Lead", paragraphs=[" ".join(words)])],
        target_words=10,
        overlap_words=2,
    )

    assert len(chunks) == 3
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk["section_title"] == "Lead" for chunk in chunks)
    assert all(chunk["text"].strip() for chunk in chunks)
    assert chunks[1]["text"].split()[:2] == chunks[0]["text"].split()[-2:]


def test_chunk_sections_keeps_section_titles():
    chunks = chunk_sections(
        [
            RawSection(
                title="Lead",
                paragraphs=["This is the first paragraph about an example entity."],
            ),
            RawSection(
                title="Work",
                paragraphs=["This is another paragraph that should keep its section title."],
            ),
        ],
        target_words=40,
        overlap_words=5,
    )

    assert {chunk["section_title"] for chunk in chunks} == {"Lead", "Work"}
