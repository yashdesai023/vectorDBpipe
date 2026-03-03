import os
import re
from pathlib import Path
from typing import List


def ensure_dir(path: str):
    """Ensure a directory exists; create it if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    Collapses whitespace and removes non-ASCII characters.
    """
    text = re.sub(r'\s+', ' ', text)             # Collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Word-level fixed-size chunking with overlap.
    Splits text by whitespace tokens; each chunk is at most `chunk_size` tokens
    with `overlap` tokens of context carried forward.

    :param text: Input cleaned text string.
    :param chunk_size: Max words per chunk.
    :param overlap: Number of words to overlap between consecutive chunks.
    :return: List of text chunk strings.
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


def chunk_text_sentences(
    text: str,
    max_tokens: int = 400,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Sentence-boundary sliding-window chunking.

    Splits text into individual sentences first (on `.`, `!`, `?`), then
    groups sentences into chunks that do not exceed `max_tokens` words.
    `overlap_sentences` trailing sentences from the previous chunk are
    prepended to the next chunk to preserve cross-boundary context.

    This avoids mid-sentence splits that the fixed word-level chunker
    (`chunk_text`) can produce, improving retrieval quality for both
    dense and sparse RAG pipelines.

    :param text: Input cleaned text string.
    :param max_tokens: Maximum words per chunk.
    :param overlap_sentences: Number of sentences to repeat at the start
                              of the next chunk (sliding window overlap).
    :return: List of text chunk strings.

    Example
    -------
    >>> chunks = chunk_text_sentences("Alice is smart. Bob is kind. Charlie leads.", max_tokens=6)
    >>> # Returns ["Alice is smart. Bob is kind.", "Bob is kind. Charlie leads."]
    """
    # Split on sentence-ending punctuation, keeping the delimiter
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    chunks: List[str] = []
    i = 0

    while i < len(sentences):
        current_chunk: List[str] = []
        current_word_count = 0

        j = i
        while j < len(sentences):
            words_in_sentence = len(sentences[j].split())
            if current_word_count + words_in_sentence > max_tokens and current_chunk:
                break  # Would overflow — emit current chunk first
            current_chunk.append(sentences[j])
            current_word_count += words_in_sentence
            j += 1

        if not current_chunk:
            # Single sentence exceeds max_tokens — include it as-is to avoid infinite loop
            current_chunk = [sentences[i]]
            j = i + 1

        chunks.append(" ".join(current_chunk))

        # Slide forward, keeping `overlap_sentences` for context
        overlap_start = max(i, j - overlap_sentences)
        i = overlap_start if overlap_start > i else j

    return [c for c in chunks if c.strip()]


def list_files_in_dir(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Lists all files in a directory filtered by extension (if provided).

    :param directory: Root directory path.
    :param extensions: List of lowercase extensions to include, e.g. ['.pdf', '.txt'].
    :return: List of absolute file path strings.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    if extensions:
        return [str(f) for f in directory.glob("**/*") if f.suffix.lower() in extensions]
    else:
        return [str(f) for f in directory.glob("**/*") if f.is_file()]
