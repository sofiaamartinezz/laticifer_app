"""Utilities for turning worker exceptions into concise user-facing text."""
from __future__ import annotations


def user_error_message(error) -> str:
    """Extract a useful message from an exception or Qt worker exc_info tuple."""
    if isinstance(error, tuple) and len(error) > 1:
        error = error[1]
    if isinstance(error, FileNotFoundError):
        return f"File not found: {error.filename or str(error)}"
    if isinstance(error, PermissionError):
        return "Permission denied. Check that the file is not locked and the folder is writable."
    message = str(error).strip()
    return message or type(error).__name__
