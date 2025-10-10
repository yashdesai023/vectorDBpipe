import os
import re
from pathlib import Path
from typing import List


def ensure_dir(path: str):
    """
    Ensure a directory exists; create it if it doesn't.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Basic text cleaning function.
    Removes unwanted whitespace, newlines, and special symbols.
    """
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Splits a long text into smaller overlapping chunks for embedding.
    """
    tokens = text.split()
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def list_files_in_dir(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Lists all files in a directory filtered by extension (if provided).
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    if extensions:
        return [str(f) for f in directory.glob("**/*") if f.suffix.lower() in extensions]
    else:
        return [str(f) for f in directory.glob("**/*") if f.is_file()]
