"""
Log cleaning and pattern extraction utilities.
Removes/replaces sensitive or redundant patterns to focus on meaningful content.
"""

import re
from typing import Optional

class LogCleaner:
    """Cleans log messages by removing/normalizing common patterns."""

    # Patterns to optionally remove or replace
    PATTERNS = {
        "uuid": re.compile(
            r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
            re.IGNORECASE
        ),
        "ipv4": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        "ipv6": re.compile(
            r'\b(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}\b',
            re.IGNORECASE
        ),
        "pid": re.compile(r'\[pid[:\s]+(\d+)\]|\bpid[=:\s]+(\d+)\b'),
        "tid": re.compile(r'\[tid[:\s]+(\d+)\]|\btid[=:\s]+(\d+)\b'),
        "memory_address": re.compile(r'\b0x[0-9a-f]+\b', re.IGNORECASE),
        "hex_string": re.compile(r'\b[0-9a-f]{16,}\b', re.IGNORECASE),
        "repeated_whitespace": re.compile(r'\s{2,}'),
        "url": re.compile(
            r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        ),
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    }

    def __init__(self, remove_patterns: Optional[list[str]] = None):
        """
        Initialize cleaner.
        Args:
            remove_patterns: List of pattern names to remove.
                            Supported: uuid, ipv4, ipv6, pid, tid, memory_address,
                                     hex_string, url, email
        """
        self.remove_patterns = remove_patterns or []

    def clean(self, text: str) -> str:
        """
        Clean log text by altering patterns.
        Args:
            text: Raw log message text
        Returns:
            Cleaned text with patterns removed/replaced.
        """
        if not text:
            return ""

        result = text

        for pattern_name in self.remove_patterns:
            if pattern_name in self.PATTERNS:
                pattern = self.PATTERNS[pattern_name]

                if pattern_name in ("pid", "tid"):
                    result = pattern.sub(f"[{pattern_name.upper()}]", result)
                elif pattern_name in ("ipv4", "ipv6"):
                    result = pattern.sub("[IP]", result)
                else:
                    result = pattern.sub(f"[{pattern_name.upper()}]", result)
        result = self.PATTERNS["repeated_whitespace"].sub(" ", result)
        result = result.strip()
        result = result.lower()

    def extract_log_level(self, text: str) -> Optional[str]:
        """
        Extract log level from text.
        Returns one of: CRITICAL, ERROR, WARN, WARNING, INFO, DEBUG, TRACE, etc.
        """
        text_upper = text.upper()

        levels = ["CRITICAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG", "TRACE"]
        for level in levels:
            if level in text_upper:
                return level
        return None

    def extract_component(self, text: str) -> Optional[str]:
        """
        Attempt to extract a component/module name from log text.
        """
        # Pattern: word followed by colon (common in many log formats)
        match = re.search(r'([A-Za-z_][A-Za-z0-9_.]*)\s*:', text)
        if match:
            return match.group(1)

        match = re.search(r'([A-Za-z_][A-Za-z0-9_.]{3,})', text)
        if match:
            component = match.group(1)
            if 3 < len(component) < 50 and "the" not in component.lower():
                return component

        return None

    def extract_timestamp(self, text: str) -> Optional[str]:
        """
        Attempt to extract a timestamp from log text.
        Returns the timestamp string if found, None otherwise.
        """
        patterns = [
            r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',  
            r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',     
            r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',      
            r'\d{4}\.\d{2}\.\d{2}',                        
            r'\d{2}:\d{2}:\d{2}',                          
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return None
