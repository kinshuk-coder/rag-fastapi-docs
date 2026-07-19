"""
retrieval/embed.py

Loads chunks.jsonl (produced by ingestion/chunk_docs.py), embeds every
chunk with a sentence-transformers model, and stores them in a local
persistent Chroma collection for retrieval.

Design notes:
- Model: BAAI/bge-small-en-v1.5. Chosen over the more generic
  all-MiniLM-L6-v2 because it's trained specifically for retrieval
  (query <-> passage matching), not general sentence similarity - which
  matters directly for RAG quality. Same embedding size (384 dims), so
  it's a drop-in swap either direction if you want to A/B them later
  in your eval (Milestone 5).
- bge models are trained with an asymmetric convention: passages are
  embedded as-is, but QUERIES should be prefixed with an instruction
  string. Skipping this halves retrieval quality in practice - it's a
  common, easy-to-miss gotcha with this model family. See embed_query().
- Chroma runs embedded (no separate server) and persists to disk at
  CHROMA_DB_PATH, so re-running this script re-embeds and overwrites -
  it does NOT incrementally update. Fine for a project this size.
"""

import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

# Same project-root anchoring pattern as ingestion/*.py - see
# fetch_docs.py's comment for the full explanation. This is what was
# missing when embed.py couldn't find chunks.jsonl: it was looking
# relative to whatever folder you happened to run it from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "chunks.jsonl")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
COLLECTION_NAME = "fastapi_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models expect this exact instruction prefix on QUERIES only,
# not on the documents/passages being indexed. This is a model-specific
# convention, not a general RAG requirement - check the model card of
# whatever embedding model you use, since conventions differ.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_passages(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Embeds document/passage text (NOT queries). No prefix needed for
    bge passages - only queries get the instruction prefix.
    """
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity assumes unit vectors
    )
    return embeddings.tolist()


def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """
    Embeds a user query. Applies the bge-specific instruction prefix -
    forgetting this is the single most common mistake with this model
    family and silently degrades retrieval without throwing any error.
    """
    prefixed = BGE_QUERY_PREFIX + query
    embedding = model.encode([prefixed], normalize_embeddings=True)
    return embedding[0].tolist()


def build_index(chunks: list[dict], model: SentenceTransformer) -> None:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Fresh start each run - simplest correct behavior for a project
    # this size. Delete-then-create avoids stale/duplicate entries if
    # chunk_docs.py's output changed since the last index build.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet - fine
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks with {MODEL_NAME}...")
    embeddings = embed_passages(model, texts)

    # Chroma wants parallel lists: ids, documents, metadatas, embeddings
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "source_file": chunk["source_file"],
            "header_path": chunk["header_path"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    # Chroma has a practical batch-size ceiling for add() in some
    # versions - batch defensively rather than sending everything in
    # one call, which can silently fail or error on larger corpora.
    BATCH_SIZE = 100
    for i in range(0, len(ids), BATCH_SIZE):
        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            documents=texts[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
            embeddings=embeddings[i:i + BATCH_SIZE],
        )

    print(f"Indexed {collection.count()} chunks into Chroma at {CHROMA_DB_PATH}")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    build_index(chunks, model)