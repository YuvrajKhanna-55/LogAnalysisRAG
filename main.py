"""
Log Analysis RAG - CLI entry point.
Ingest log files and query them using RAG.
"""

import argparse
import glob
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "logfiles"
STORE_DIR = Path(__file__).parent / "data" / "vector_store"


def ingest(log_files: list[str], max_lines: int | None = None, clean_patterns: list[str] | None = None) -> None:
    """Parse log files, generate embeddings, and build vector store."""
    try:
        from log_parser import LogParser
        from log_embeddings import LogEmbedder
        from retrievalvectorsdb import VectorStore
    except Exception as exc:
        print(f"Failed to load ingestion dependencies: {exc}")
        print(f"Interpreter: {sys.executable}")
        print("Activate the project venv and reinstall dependencies if needed.")
        return

    all_docs = []

    for filepath in log_files:
        print(f"\nParsing: {filepath}")
        parser = LogParser(filepath, max_lines=max_lines, clean_patterns=clean_patterns)
        docs = parser.to_documents()
        print(f"  -> {len(docs)} log entries parsed")
        all_docs.extend(docs)

    if not all_docs:
        print("No log entries found. Check your log files.")
        return

    print(f"\nTotal documents: {len(all_docs)}")

    embedder = LogEmbedder()
    chunks, embeddings = embedder.embed_documents(all_docs)

    store = VectorStore()
    store.build(chunks, embeddings)
    store.save(str(STORE_DIR))
    print("\nIngestion complete!")


def query(question: str) -> None:
    """Query the log knowledge base."""
    try:
        from log_embeddings import LogEmbedder
        from retrievalvectorsdb import VectorStore
        from analyzevectors import LogAnalyzer
    except Exception as exc:
        print(f"Failed to load query dependencies: {exc}")
        print(f"Interpreter: {sys.executable}")
        print("Activate the project venv and reinstall dependencies if needed.")
        return

    embedder = LogEmbedder()
    store = VectorStore()
    store.load(str(STORE_DIR))

    analyzer = LogAnalyzer(vector_store=store, embedder=embedder)
    try:
        result = analyzer.analyze(question)
    except ValueError as exc:
        print(str(exc))
        return

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n{'='*60}")
    print(f"Based on {len(result['context'])} retrieved log chunks (model: {result['model']})")


def main():
    parser = argparse.ArgumentParser(description="Log Analysis RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest log files into vector store")
    ingest_parser.add_argument(
        "--files", nargs="*", help="Specific log files to ingest (default: all in data/logfiles/)"
    )
    ingest_parser.add_argument(
        "--max-lines", type=int, default=None,
        help="Maximum lines to parse per file (useful for large files)"
    )
    ingest_parser.add_argument(
        "--clean-patterns", nargs="*", default=None,
        help="Patterns to remove from logs. Options: uuid, ipv4, ipv6, pid, tid, email, memory_address, url"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the log knowledge base")
    query_parser.add_argument("question", help="Question to ask about the logs")

    args = parser.parse_args()

    if args.command == "ingest":
        if args.files:
            log_files = args.files
        else:
            log_files = sorted(glob.glob(str(DATA_DIR / "*.log")))
        if not log_files:
            print(f"No log files found in {DATA_DIR}")
            return
        print(f"Found {len(log_files)} log files to ingest")
        ingest(log_files, max_lines=args.max_lines, clean_patterns=args.clean_patterns)

    elif args.command == "query":
        query(args.question)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
