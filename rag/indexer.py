"""
Task 3 (Part 1): Embedding + Indexing
-----------------------------------------
Embeds both chunk sets (fixed-size and sentence-based) using a free local
SentenceTransformers model, and indexes each strategy into its OWN separate
ChromaDB collection so they can be compared independently in Task 5.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import build_fixed_size_chunks, build_sentence_chunks

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")

FIXED_COLLECTION_NAME = "kb_fixed_size_chunks"
SENTENCE_COLLECTION_NAME = "kb_sentence_chunks"

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def index_chunks(client, collection_name: str, chunks: list):
    """Embed a list of chunk dicts and upsert them into a Chroma collection."""
    model = get_embedding_model()

    # Fresh collection each run for reproducibility during development
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"doc_id": c["doc_id"], "topic": c["topic"]} for c in chunks],
    )
    return collection


def build_and_index_all():
    client = get_chroma_client()

    fixed_chunks = build_fixed_size_chunks()
    sentence_chunks = build_sentence_chunks()

    fixed_collection = index_chunks(client, FIXED_COLLECTION_NAME, fixed_chunks)
    sentence_collection = index_chunks(client, SENTENCE_COLLECTION_NAME, sentence_chunks)

    return fixed_collection, sentence_collection


def query_collection(collection, query_text: str, top_k: int = 3):
    """Query a Chroma collection with a raw query string, returns chunks + distances."""
    model = get_embedding_model()
    query_embedding = model.encode([query_text], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        # Chroma returns squared L2 distance by default; since embeddings are
        # explicitly normalized to unit vectors, this converts distance into
        # a cosine-similarity-equivalent score for readability.
        distance = results["distances"][0][i]
        similarity = 1 - (distance / 2)
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "topic": results["metadatas"][0][i]["topic"],
            "similarity": similarity,
        })
    return hits


if __name__ == "__main__":
    print("Building embeddings and indexing into ChromaDB (this may take ~30s)...")
    fixed_collection, sentence_collection = build_and_index_all()

    print(f"\nFixed-size collection count: {fixed_collection.count()}")
    print(f"Sentence-based collection count: {sentence_collection.count()}")

    sample_query = "What documents do I need for KYC?"
    print(f"\nSample query: '{sample_query}'")

    print("\n-- Fixed-size collection results --")
    for hit in query_collection(fixed_collection, sample_query, top_k=3):
        print(f"  [{hit['similarity']:.3f}] {hit['doc_id']} | {hit['text'][:80]}...")

    print("\n-- Sentence-based collection results --")
    for hit in query_collection(sentence_collection, sample_query, top_k=3):
        print(f"  [{hit['similarity']:.3f}] {hit['doc_id']} | {hit['text'][:80]}...")