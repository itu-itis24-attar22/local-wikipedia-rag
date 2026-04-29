"""Command-line chat UI."""

from __future__ import annotations

import argparse

from app.rag_pipeline import RAGResponse, answer_question


def print_response(response: RAGResponse, *, show_context: bool = False) -> None:
    print(f"\nAnswer:\n{response.answer}\n")
    if not show_context:
        return
    print("Retrieved context:")
    for index, chunk in enumerate(response.retrieved_chunks, start=1):
        metadata = chunk.metadata
        print(
            f"\n[{index}] {metadata.get('entity_name', '')} "
            f"({metadata.get('entity_type', '')}) - {metadata.get('section_title', '')}"
        )
        print(f"Source: {metadata.get('source_url', '')}")
        print(chunk.text)


def ask_once(question: str, *, show_context: bool = False) -> None:
    response = answer_question(question)
    print_response(response, show_context=show_context)


def interactive_chat(*, show_context: bool = False) -> None:
    print("Local Wikipedia RAG Assistant CLI")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            ask_once(question, show_context=show_context)
        except Exception as exc:
            print(f"Local RAG pipeline error: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local Wikipedia RAG assistant.")
    parser.add_argument("question", nargs="*", help="Optional one-shot question.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if question:
        try:
            ask_once(question, show_context=args.show_context)
        except Exception as exc:
            print(f"Local RAG pipeline error: {exc}")
            raise SystemExit(1) from exc
    else:
        interactive_chat(show_context=args.show_context)


if __name__ == "__main__":
    main()
