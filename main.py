"""Entry point helpers for the Local Wikipedia RAG Assistant."""

from app.config import PROJECT_ROOT


def main() -> None:
    print(f"Local Wikipedia RAG Assistant project root: {PROJECT_ROOT}")
    print("Run the UI with: streamlit run app/ui_streamlit.py")


if __name__ == "__main__":
    main()
