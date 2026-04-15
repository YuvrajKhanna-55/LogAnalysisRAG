"""
Streamlit User-Interface for the Log Analysis RAG System.
"""
import glob
from datetime import datetime
from pathlib import Path
import streamlit as st
from log_parser import LogParser
from log_embeddings import LogEmbedder
from retrievalvectorsdb import VectorStore
from analyzevectors import LogAnalyzer

DATA_DIR = Path(__file__).parent / "data" / "logfiles"
STORE_DIR = Path(__file__).parent / "data" / "vector_store"


def clear_vector_store_files() -> tuple[int, int]:
    """Delete persisted vector-store files and return (deleted, failed) counts."""
    files_to_remove = ["faiss.index", "chunks.json", "bm25.pkl", "metadata.json"]
    deleted = 0
    failed = 0

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    for filename in files_to_remove:
        path = STORE_DIR / filename
        if path.exists():
            try:
                path.unlink()
                deleted += 1
            except OSError:
                failed += 1

    return deleted, failed


@st.cache_resource
def load_embedder():
    return LogEmbedder()


@st.cache_resource
def load_vector_store():
    store = VectorStore()
    if (STORE_DIR / "faiss.index").exists():
        store.load(str(STORE_DIR))
        return store
    return None


def ingest_logs(log_files: list[str], max_lines: int | None, clean_patterns: list[str] | None = None) -> str:
    """Run the ingestion pipeline and return status message."""
    all_docs = []

    for filepath in log_files:
        parser = LogParser(filepath, max_lines=max_lines, clean_patterns=clean_patterns)
        docs = parser.to_documents()
        all_docs.extend(docs)

    if not all_docs:
        return "No log entries found. Check your log files."

    embedder = load_embedder()
    chunks, embeddings = embedder.embed_documents(all_docs, show_progress=False)

    store = VectorStore()
    store.build(chunks, embeddings)
    store.save(str(STORE_DIR))

    # Clear cached store so it reloads
    load_vector_store.clear()
    
    return f"Ingested {len(all_docs)} log entries into {len(chunks)} chunks."


def save_uploaded_logs(uploaded_files) -> list[str]:
    """Persist uploaded files into data/logfiles and return saved paths."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    for uploaded in uploaded_files:
        original_name = Path(uploaded.name).name
        destination = DATA_DIR / original_name

        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = DATA_DIR / f"{destination.stem}_{timestamp}{destination.suffix}"

        destination.write_bytes(uploaded.getbuffer())
        saved_paths.append(str(destination))

    return saved_paths


def main():
    st.set_page_config(page_title="Log Analysis RAG", page_icon="🔍", layout="wide")
    st.title("🔍 Log Analysis RAG Based System")
    st.markdown("Analyze system logs using Retrieval-Augmented Generation.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Reset old vector-store artifacts once per new Streamlit session.
    if "vector_store_reset_done" not in st.session_state:
        deleted, failed = clear_vector_store_files()
        load_vector_store.clear()
        st.session_state["vector_store_reset_done"] = True
        st.session_state["vector_store_reset_msg"] = (deleted, failed)

    # ── Sidebar: Ingestion ──
    with st.sidebar:
        st.header("📂 Log Ingestion")

        if "vector_store_reset_msg" in st.session_state:
            deleted, failed = st.session_state["vector_store_reset_msg"]
            if failed:
                st.warning(
                    f"Startup reset partially completed: deleted {deleted} files, failed to delete {failed}."
                )
            elif deleted:
                st.info(f"Startup reset completed: deleted {deleted} old vector-store files.")
            else:
                st.info("Startup reset completed: no old vector-store files were found.")

        if st.button("🧹 Clear Vector Store"):
            deleted, failed = clear_vector_store_files()
            load_vector_store.clear()
            if failed:
                st.error(
                    f"Clear completed with issues: deleted {deleted} files, failed to delete {failed}."
                )
            elif deleted:
                st.success(f"Vector store cleared. Deleted {deleted} files.")
            else:
                st.info("Vector store is already empty.")

        st.markdown("---")

        uploaded_files = st.file_uploader(
            "Upload log files",
            type=["log", "txt"],
            accept_multiple_files=True,
            help="Uploaded files are stored in data/logfiles and can be ingested immediately.",
        )


        if st.button("⬆️ Upload & Ingest", disabled=not uploaded_files):
            with st.spinner("Saving uploaded files and building vector store..."):
                uploaded_paths = save_uploaded_logs(uploaded_files or [])
                msg = ingest_logs(uploaded_paths, max_lines=None)
            st.success(f"Uploaded {len(uploaded_paths)} file(s). {msg}")
 
        st.markdown("---")

        store = load_vector_store()
        if store and store.index is not None:
            st.success(f"Vector store loaded: {len(store.chunks)} chunks")
        else:
            st.info("No vector store found. Ingest logs first.")

    store = load_vector_store()

    if store is None or store.index is None:
        st.info("👈 Ingest  system log files from the sidebar to get started and Retrieve errors or ask questions related to it.")
        return

    # Example queries
    with st.expander("📋 Example queries"):
        examples = [
            "What are the most common errors in the logs?",
            "Are there any critical warnings or failures?",
            "Show me authentication or permission-related issues.",
            "What recurring patterns indicate system instability?",
            "Summarize the main activities seen in these logs.",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state["query_input"] = ex

    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get("query_input", ""),
        height=80,
        placeholder="e.g., What errors occurred in the logs?",
    )

    def build_analyzer() -> LogAnalyzer:
        return LogAnalyzer(
            vector_store=store,
            embedder=load_embedder(),
            top_k=10,
            alpha=0.7,
        )

    if st.button("🔍 Analyze", type="primary", disabled=not query):
        try:
            analyzer = build_analyzer()
            with st.spinner("Retrieving context and generating analysis..."):
                result = analyzer.analyze(query)
        except ValueError as exc:
            st.error(str(exc))
            return

        # Display answer
        st.subheader("📊 Analysis")
        st.markdown(result["answer"])

        # Display retrieved context
        with st.expander(f"📄 Retrieved Context ({len(result['context'])} chunks)"):
            for i, (chunk, score) in enumerate(
                zip(result["context"], result["scores"])
            ):
                st.markdown(f"**Chunk {i+1}** (score: {score:.4f})")
                st.code(chunk, language="log")

    # ── Retrieval-only mode ──
    st.markdown("---")
    with st.expander("🔎 Retrieval Only (no LLM)"):
        search_query = st.text_input("Search logs directly:", placeholder="e.g., ERROR kernel")
        if search_query:
            analyzer = build_analyzer()
            results = analyzer.retrieve(search_query)
            for i, (chunk, score) in enumerate(results):
                st.markdown(f"**Result {i+1}** (score: {score:.4f})")
                st.code(chunk, language="log")


if __name__ == "__main__":
    main()
