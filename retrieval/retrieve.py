"""
retrieval/retrieve.py

Query-time retrieval: embed a question, search the Chroma collection
built by embed.py, return the top-k most relevant chunks.

This is intentionally the ONLY function Milestone 4 (generation) and
Milestone 5 (eval) will call - keeping retrieval logic in one place
means when we upgrade to hybrid search in Milestone 6, only this file
changes; nothing downstream needs to know the difference.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from embed import CHROMA_DB_PATH, COLLECTION_NAME, MODEL_NAME, embed_query


def retrieve(query: str, model: SentenceTransformer, collection, top_k: int = 5) -> list[dict]:
    """
    Returns a list of dicts, each: {chunk_id, text, header_path,
    source_file, distance}. Lower distance = more relevant
    (cosine distance, since the collection was built with
    hnsw:space="cosine").
    """
    query_embedding = embed_query(model, query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # Chroma returns parallel lists nested one level for batch queries -
    # since we only sent one query, everything we want is at index [0].
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "header_path": results["metadatas"][0][i]["header_path"],
            "source_file": results["metadatas"][0][i]["source_file"],
            "distance": results["distances"][0][i],
        })
    return retrieved


def load_retriever():
    """
    Convenience loader so callers (generation.py, eval scripts) don't
    need to know Chroma/model setup details - just call this once and
    pass the results into retrieve().
    """
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


if __name__ == "__main__":
    # Manual sanity-check mode: ask a few known questions, eyeball
    # whether the retrieved chunks actually look relevant. This is the
    # Milestone 3 equivalent of Milestone 2's "print_random_sample" -
    # don't skip actually reading the output.
    model, collection = load_retriever()

    sample_questions = [
        "How do I add a custom exception handler?",
        "What is dependency injection in FastAPI?",
        "How do I validate a request body with Pydantic?",
    ]

    for question in sample_questions:
        print("=" * 80)
        print(f"QUERY: {question}")
        print("-" * 80)
        results = retrieve(question, model, collection, top_k=3)
        for rank, r in enumerate(results, start=1):
            print(f"[{rank}] distance={r['distance']:.4f}  {r['header_path']}  ({r['chunk_id']})")
            preview = r["text"][:150].replace("\n", " ")
            print(f"    {preview}...")
        print()