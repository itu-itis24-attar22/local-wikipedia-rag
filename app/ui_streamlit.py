"""Streamlit chat UI."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHROMA_COLLECTION
from app.database import reset_database
from app.ingest import clear_raw_files
from app.rag_pipeline import RAGResponse, answer_question


def reset_chroma_collection() -> None:
    """Delete the Chroma collection when Chroma is installed."""
    try:
        from app.vector_store import get_client

        client = get_client()
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
    except Exception:
        pass


def reset_local_system() -> None:
    """Reset generated local data."""
    reset_database(force=True)
    clear_raw_files()
    reset_chroma_collection()


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_context(response: RAGResponse) -> None:
    for index, chunk in enumerate(response.retrieved_chunks, start=1):
        metadata = chunk.metadata
        label = (
            f"{index}. {metadata.get('entity_name', 'Unknown')} "
            f"- {metadata.get('section_title', 'Section')}"
        )
        with st.expander(label):
            st.caption(
                f"{metadata.get('entity_type', '')} | "
                f"chunk {metadata.get('chunk_index', '')} | "
                f"{metadata.get('source_url', '')}"
            )
            st.write(chunk.text)


def main() -> None:
    st.set_page_config(page_title="Local Wikipedia RAG", layout="wide")
    initialize_session()

    with st.sidebar:
        st.header("Local Controls")
        show_context = st.checkbox("Show retrieved context", value=False)
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        confirm_reset = st.checkbox("Confirm local data reset")
        if st.button("Reset local data", disabled=not confirm_reset, use_container_width=True):
            reset_local_system()
            st.session_state.messages = []
            st.success("Local data reset.")

    st.title("Local Wikipedia RAG Assistant")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if show_context and message["role"] == "assistant" and message.get("response"):
                render_context(message["response"])

    query = st.chat_input("Ask about a configured famous person or place")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching local Wikipedia data..."):
            try:
                response = answer_question(query)
                st.write(response.answer)
                if show_context:
                    render_context(response)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "response": response,
                    }
                )
            except Exception as exc:
                error_message = f"Local RAG pipeline error: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "response": None}
                )


if __name__ == "__main__":
    main()
