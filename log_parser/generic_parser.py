"""
Generic log format parser for unknown/unstructured log formats.
Provides heuristic-based extraction of log structure when known patterns fail.
"""

import re
from typing import Optional
from .log_cleaner import LogCleaner


class GenericLogParser:
    """
    Parses logs with unknown formats using heuristics.
    Extracts: timestamp, level, component, and message.
    """

    def __init__(self, clean_patterns: Optional[list[str]] = None):
        """
        Initialize generic parser.

        Args:
            clean_patterns: Patterns to remove from parsed text.
        """
        self.cleaner = LogCleaner(remove_patterns=clean_patterns)

    def parse_line(self, line: str) -> dict:
        """
        Parse a single log line using heuristics.

        Returns dict with keys: timestamp, level, component, message, raw
        Unknown fields are None or empty string.
        """
        if not line or not line.strip():
            return None

        raw = line.strip()

        # Extract components
        timestamp = self.cleaner.extract_timestamp(raw)
        level = self.cleaner.extract_log_level(raw)
        component = self.cleaner.extract_component(raw)

        # Determine message: preferably content after level or component
        message = raw
        if level:
            # Remove level from message to avoid duplication
            message = re.sub(r'\b' + level + r'\b', '', raw, flags=re.IGNORECASE).strip()

        message = self.cleaner.clean(message)

        return {
            "timestamp": timestamp,
            "level": level or "UNKNOWN",
            "component": component or "unknown",
            "message": message,
            "raw": raw,
        }

    def is_likely_log_file(self, first_lines: list[str]) -> bool:
        """
        Heuristic check: is this likely a log file?

        Checks if multiple lines contain timestamps or log levels.
        """
        if not first_lines:
            return False

        score = 0
        check_count = min(10, len(first_lines))

        for line in first_lines[:check_count]:
            # Check for timestamp
            if self.cleaner.extract_timestamp(line):
                score += 1
            # Check for log level
            if self.cleaner.extract_log_level(line):
                score += 1

        # If 50%+ of sampled lines have timestamp or level, likely a log file
        return score >= check_count
