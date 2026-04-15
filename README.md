# Log Analysis RAG System

A Retrieval-Augmented Generation pipeline for analyzing system log files. Parses any log format using heuristics, generates vector embeddings, and uses hybrid search (FAISS + BM25) with an LLM to answer questions about your logs.

## Installation

Create virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

Set up your Groq API key. Create .env and add:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

The system loads from .env (preferred) or .env.example (fallback).

## Architecture

The pipeline flows through five stages:

1. log_parser: Parses logs in any format using heuristic extraction
2. log_embeddings: Generates embeddings via sentence-transformers
3. retrievalvectorsdb: Stores chunks using FAISS (dense) + BM25 (sparse) hybrid search
4. analyzevectors: RAG analyzer with Groq LLM for question answering
5. app.py: Streamlit web UI for upload, search, and analysis

```
data/logfiles/*.log
    ↓
log_parser (heuristic extraction)
    ↓
log_embeddings (sentence-transformers)
    ↓
retrievalvectorsdb (FAISS + BM25)
    ↓
analyzevectors (retrieve + reranking + LLM)
    ↓
Streamlit (web interface UI)
```

## Supported Log Formats

The generalized parser works with any log file format: Syslog, application logs, custom formats, unstructured text, web server logs, database logs, cloud platform logs, or any format you provide. It extracts timestamps, log levels, components, and messages using intelligent heuristics without format-specific code.

## Quick Start

CLI Usage:

```bash
# Ingest logs (limit lines for large files)
python main.py ingest --max-lines 10000

# Ingest specific files with pattern cleaning
python main.py ingest --files data/logfiles/*.log --clean-patterns ipv4 uuid pid

# Query the indexed logs
python main.py query "What errors are most common?"
```

Web UI:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then upload logs via sidebar, optionally remove sensitive data, and ask questions in the main panel. The interface shows analysis results plus retrieved log context.

## Project Structure

```
main.py                      CLI entry point
app.py                       Streamlit web UI
log_parser/                  Log parsing with heuristics
  parser.py                  Main parser with fallback
  log_cleaner.py             Pattern removal utilities
  generic_parser.py          Format-agnostic parser
log_embeddings/              Embedding generation
  embedder.py                Sentence-transformers wrapper
retrievalvectorsdb/          Vector storage and search
  vector_store.py            FAISS + BM25 hybrid search
analyzevectors/              RAG analyzer
  analyzer.py                Retrieval + reranking + LLM
data/
  logfiles/                  Raw log files
  vector_store/              Persisted FAISS index
requirements.txt             Python dependencies
pyproject.toml               Project configuration
```

## Key Parameters

max_lines: Max lines per file (default: None, all lines)
clean_patterns: Remove patterns like ipv4, uuid, pid, tid, email, memory_address, url (default: None)
chunk_size: Characters per chunk (default: 512)
top_k: Retrieved chunks for context (default: 10)
alpha: Hybrid search weight, 1.0=dense only, 0.0=sparse only (default is 0.7)
model: LLM model for analysis (default: llama-3.1-8b-instant)

## Pattern Cleaning

Remove sensitive data before embedding and searching. Supported patterns:

uuid: 550e8400-e29b-41d4-a716-446655440000 → [UUID]
ipv4: 192.168.1.1 → [IP]
ipv6: 2001:db8::1 → [IP]
pid: [pid: 12345] → [PID]
tid: [tid: 5678] → [TID]
email: user@example.com → [EMAIL]
memory_address: 0xdeadbeef → [MEMORY_ADDRESS]
url: https://example.com → [URL]

CLI example:

```bash
python main.py ingest --files logs.txt --clean-patterns ipv4 uuid pid
```

Python API:

```python
from log_parser import LogParser
parser = LogParser("logs.txt", clean_patterns=["ipv4", "uuid"])
docs = parser.to_documents()
```

## Log Parser

The log parser extracts structured data from any format via heuristic matching. Each log line becomes a dictionary with fields: timestamp, level, component, message, raw, source_file, format.

Three key classes:

LogCleaner: Removes patterns from log text, extracts timestamp/level/component fields.

```python
from log_parser import LogCleaner
cleaner = LogCleaner(remove_patterns=["ipv4", "uuid"])
cleaned = cleaner.clean("Request from 192.168.1.100 (UUID: 550e8400-...)")
level = cleaner.extract_log_level("ERROR: Database connection failed")
component = cleaner.extract_component("nginx: Connection refused")
```

GenericLogParser: Parses logs with unknown formats using heuristics.

```python
from log_parser import GenericLogParser
parser = GenericLogParser(clean_patterns=["ipv4", "uuid"])
entry = parser.parse_line("2024-11-25 12:34:56 [ERROR] kernel: Device eth0 brought up")
is_logfile = parser.is_likely_log_file(first_lines)
```

LogParser: Main entry point supporting both known and unknown formats.

```python
from log_parser import LogParser
parser = LogParser("application.log", max_lines=10000, clean_patterns=["ipv4", "uuid"])
for entry in parser.stream():
    print(entry["level"], entry["component"], entry["message"])


docs = parser.to_documents()
```

## Retrieval and Reranking

The analyzer retrieves log chunks using hybrid search: dense vectors (FAISS) combined with keyword scores (BM25). Retrieved chunks are then reranked with a cross-encoder model (ms-marco-MiniLM) to improve relevance before passing to the LLM.

```python
from analyzevectors import LogAnalyzer

analyzer = LogAnalyzer(
    vector_store=store,
    embedder=embedder,
    top_k=10,
    alpha=0.7,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_multiplier=4
)

# Retrieve and rerank
results = analyzer.retrieve(query)

# Full analysis with LLM
result = analyzer.analyze(query)
```
The retrieve method fetches top_k * rerank_multiplier candidates, then reranks to the final top_k.

## Troubleshooting

If running with system Python shows TensorFlow/Keras errors, use the project virtual environment:

```bash
.\.venv\Scripts\python.exe main.py ingest --max-lines 5000
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If logs aren't parsed, check they contain timestamps or log levels. Try parser.stream() to inspect raw entries.

If pattern cleaning seems too aggressive, use only patterns you need:

```python
parser = LogParser("logs.txt", clean_patterns=["uuid"])  # Only remove UUIDs
```

Performance tips: Use max_lines to limit large files. Stream entries for line-by-line processing. Avoid unnecessary cleaning patterns. Use the venv Python interpreter.

## Examples

Ingest multiple files and ask questions:

```bash
python main.py ingest --files file1.log file2.log file3.log --clean-patterns ipv4 uuid
python main.py query "What errors occurred and when?"
```

Stream large logs for custom processing:

```python
from log_parser import LogParser

parser = LogParser("large.log", max_lines=100000, clean_patterns=["ipv4"])
for entry in parser.stream():
    if entry["level"] == "ERROR":
        print(f"{entry['timestamp']}: {entry['message']}")
```

Use the Streamlit Web UI  for interactive analysis. Upload logs via sidebar, select patterns to remove, ask questions, and review analysis with context.
