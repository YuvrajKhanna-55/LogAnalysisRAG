"""
Generalized log parser module for any log file format.
Uses heuristic-based extraction to parse timestamps, log levels, components, and messages from unstructured log files.
"""

from pathlib import Path
from typing import Generator
from .log_cleaner import LogCleaner
from .generic_parser import GenericLogParser

class LogParser:

    def __init__(
        self,
        filepath: str,
        max_lines: int | None = None,
        clean_patterns: list[str] | None = None,
    ):
        """
        Initialize parser.
        Args:
            filepath: Path to log file
            max_lines: Maximum lines to parse
            clean_patterns: Patterns to remove from log messages.
                          Options: uuid, ipv4, ipv6, pid, tid, memory_address,
                                 hex_string, url, email
        """
        self.filepath = Path(filepath)
        self.max_lines = max_lines
        self.clean_patterns = clean_patterns or []
        self.generic_parser = GenericLogParser(clean_patterns=self.clean_patterns)
        self.cleaner = LogCleaner(remove_patterns=self.clean_patterns)

    def parse(self) -> list[dict]:
        """Parse the log file and return list of structured entries."""
        return list(self.stream())

    def stream(self) -> Generator[dict, None, None]:
        count = 0
        with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if self.max_lines and count >= self.max_lines:
                    break
                entry = self.generic_parser.parse_line(line)
                if entry:
                    entry["source_file"] = self.filepath.name
                    # Keep raw line before cleaning for reference
                    entry["raw"] = line.strip()
                    count += 1
                    yield entry

    def to_documents(self) -> list[str]:
        """
        Convert parsed entries to text documents suitable for embedding.
        Constructs documents from parsed fields.
        """
        docs = []
        for entry in self.stream():
            level = entry.get("level", "UNKNOWN")
            component = entry.get("component", "unknown")
            message = entry.get("message", "")
            source = entry.get("source_file", "") 
            doc = f"[{source}] [{level}] {component}: {message}"
            docs.append(doc)

        return docs

