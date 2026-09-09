"""
Task 3 (Part 1): Chunking Strategies
---------------------------------------
Implements TWO chunking strategies over the knowledge base documents:

1. Fixed-size-with-overlap chunking (character-based sliding window).
2. Sentence-based chunking (splits on sentence boundaries, one or more
   sentences per chunk up to a soft max length).

Each chunk keeps a reference back to its parent doc_id so that later
(Task 5) we can map chunks -> parent documents for document-level scoring.
"""

import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base.documents import KNOWLEDGE_BASE_DOCUMENTS

# ---- Fixed-size-with-overlap config ----
FIXED_CHUNK_SIZE = 220      # characters
FIXED_CHUNK_OVERLAP = 50    # characters

# ---- Sentence-based config ----
SENTENCES_PER_CHUNK = 2     # group N sentences per chunk


def fixed_size_chunk(text: str, chunk_size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP):
    """Slide a fixed-size character window over text with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == text_len:
            break
        start += (chunk_size - overlap)
    return chunks


def sentence_split(text: str):
    """Naive sentence splitter on '.', '!', '?' followed by whitespace."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def sentence_based_chunk(text: str, sentences_per_chunk: int = SENTENCES_PER_CHUNK):
    """Group consecutive sentences into chunks of N sentences each."""
    sentences = sentence_split(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        chunks.append(" ".join(group))
    return chunks


def build_fixed_size_chunks(documents=KNOWLEDGE_BASE_DOCUMENTS):
    """Returns list of {chunk_id, doc_id, topic, text} for fixed-size strategy."""
    all_chunks = []
    for doc in documents:
        pieces = fixed_size_chunk(doc["text"])
        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_fixed_{idx}",
                "doc_id": doc["doc_id"],
                "topic": doc["topic"],
                "text": piece,
            })
    return all_chunks


def build_sentence_chunks(documents=KNOWLEDGE_BASE_DOCUMENTS):
    """Returns list of {chunk_id, doc_id, topic, text} for sentence-based strategy."""
    all_chunks = []
    for doc in documents:
        pieces = sentence_based_chunk(doc["text"])
        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_sent_{idx}",
                "doc_id": doc["doc_id"],
                "topic": doc["topic"],
                "text": piece,
            })
    return all_chunks


if __name__ == "__main__":
    fixed_chunks = build_fixed_size_chunks()
    sentence_chunks = build_sentence_chunks()

    print(f"Fixed-size-with-overlap: {len(fixed_chunks)} chunks from {len(KNOWLEDGE_BASE_DOCUMENTS)} documents")
    print(f"Sentence-based: {len(sentence_chunks)} chunks from {len(KNOWLEDGE_BASE_DOCUMENTS)} documents")

    print("\nSample fixed-size chunk:")
    print(fixed_chunks[0])

    print("\nSample sentence-based chunk:")
    print(sentence_chunks[0])