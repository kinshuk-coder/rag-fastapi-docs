"""
retrieval/retrieve.py

Hybrid retrieval: combines dense (embedding) search with BM25 (keyword)
search, fused via Reciprocal Rank Fusion (RRF), and returns a single
ranked list of chunks.

Why hybrid: our Milestone 5 eval showed dense retrieval alone is
already strong (~96% hit-rate). Hybrid search isn't fixing something
broken - it's targeting dense embeddings' specific, well-known weak
spot: exact keyword/identifier matching. A query like "how do I use
HTTPException?" should match the literal token "HTTPException" with
high confidence; a dense embedding can drift toward semantically
similar but not identical content instead. BM25 catches the literal
match; RRF lets both signals contribute without needing to normalize
two incomparable score scales (cosine distance vs. BM25 score).

This file replaces the dense-only retrieve() from Milestone 3. The
public interface - retrieve(query, ..., top_k) -> list[dict] - is
UNCHANGED in shape, but now takes an extra bm25_index argument, so
generate.py and run_eval.py need one small wiring update (see the
explanation accompanying this file).
"""

import os
import re
import math
import chromadb
from huggingface_hub import InferenceClient
from rank_bm25 import BM25Okapi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
COLLECTION_NAME = "fastapi_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# How many candidates each method contributes to the fusion stage,
# before we cut down to the final top_k. Wider than top_k on purpose -
# fusion needs room to re-rank; if both methods only returned exactly
# top_k each, a chunk ranked #6 by dense but #1 by BM25 would never
# get the chance to surface.
CANDIDATE_POOL_SIZE = 20

# Default number of chunks returned per query.
#
# History: dropped from 5 to 3 during Milestone 6 rate-limit tuning to
# cut tokens/call. Reverted back to 5 after concrete evidence (q005,
# q015 in the eval set) showed top_k=3 was pushing the CORRECT,
# canonical source chunk out of the results entirely - e.g. q005's
# target chunk ranked #1 by BM25 alone and still didn't survive fusion
# into a 3-slot result. The real latency fix turned out to be judge-
# model separation (see GROQ_JUDGE_MODEL below), which eliminated the
# generation/judge budget contention that was the actual bottleneck -
# so top_k no longer needs to be sacrificed for speed.
DEFAULT_TOP_K = 5

# Standard RRF damping constant - see reciprocal_rank_fusion() below
# for how it's used. 60 is the commonly cited default in RRF literature;
# it's not sensitive enough to need tuning for a corpus this size.
RRF_K = 60


# Render's 512 MB instance cannot safely host PyTorch plus two transformer
# models. Query embeddings are therefore produced by Hugging Face Inference;
# the precomputed Chroma vectors remain local. This preserves dense + BM25 RRF
# retrieval while intentionally omitting the former local cross-encoder stage.
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", MODEL_NAME)


def tokenize(text: str) -> list[str]:
    """Simple whitespace/word tokenizer for BM25 - lowercased word tokens."""
    return re.findall(r"\w+", text.lower())


def load_retriever():
    """
    Loads everything needed for low-memory hybrid retrieval:
    - a Hugging Face Inference client (for dense query embeddings)
    - the Chroma collection (for dense search)
    - a BM25 index built from the SAME chunks stored in Chroma (for
      keyword search)
    Building the BM25 index from collection.get() rather than re-reading
    chunks.jsonl directly keeps a single source of truth - if Chroma's
    index and chunks.jsonl ever drifted out of sync, we want BM25 to
    reflect what's actually indexed, not what's on disk elsewhere.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for hosted query embeddings. Add it to your environment secrets.")
    embedding_client = InferenceClient(provider="hf-inference", api_key=hf_token, timeout=30)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    texts = all_data["documents"]
    metadatas = all_data["metadatas"]

    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_index = {
        "bm25": bm25,
        "ids": ids,
        "texts": texts,
        "metadatas": metadatas,
    }

    return embedding_client, collection, bm25_index


def dense_search(query: str, embedding_client: InferenceClient, collection, top_k: int) -> list[dict]:
    """Hosted query embedding plus local Chroma vector search."""
    raw_embedding = embedding_client.feature_extraction(BGE_QUERY_PREFIX + query, model=HF_EMBEDDING_MODEL)
    query_embedding = raw_embedding.tolist()
    if query_embedding and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]
    # The existing Chroma corpus was indexed with normalized BGE vectors.
    # Normalize locally so this works with any compatible HF embedding server,
    # including servers that do not offer their own normalize parameter.
    magnitude = math.sqrt(sum(value * value for value in query_embedding))
    if magnitude == 0:
        raise RuntimeError("Hugging Face returned a zero-length embedding vector.")
    query_embedding = [value / magnitude for value in query_embedding]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    ranked = []
    for i in range(len(results["ids"][0])):
        ranked.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "header_path": results["metadatas"][0][i]["header_path"],
            "source_file": results["metadatas"][0][i]["source_file"],
            "distance": results["distances"][0][i],
        })
    return ranked


def bm25_search(query: str, bm25_index: dict, top_k: int) -> list[dict]:
    """Returns the top_k chunks by BM25 score, in the same dict shape as dense_search."""
    bm25 = bm25_index["bm25"]
    scores = bm25.get_scores(tokenize(query))

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    ranked = []
    for i in top_indices:
        ranked.append({
            "chunk_id": bm25_index["ids"][i],
            "text": bm25_index["texts"][i],
            "header_path": bm25_index["metadatas"][i]["header_path"],
            "source_file": bm25_index["metadatas"][i]["source_file"],
            "bm25_score": float(scores[i]),
        })
    return ranked


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Merges multiple ranked lists of chunks into one, using RRF:
        fused_score(chunk) = sum over each list the chunk appears in of
                              1 / (k + rank_in_that_list)
    Rank is 1-indexed. A chunk missing from a list contributes 0 for
    that list, not a penalty - being retrieved by only one method still
    counts, just less than being retrieved highly by both.

    RRF is a rank-based fusion (uses position, not the raw score value),
    which is exactly why it can combine cosine distance and BM25 score
    despite the two being on completely different, non-comparable scales.
    """
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["chunk_id"]
            fused_scores.setdefault(chunk_id, 0.0)
            fused_scores[chunk_id] += 1.0 / (k + rank)
            chunk_lookup[chunk_id] = chunk  # last write wins for display fields - fine, text/header_path are identical either way

    ranked_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    results = []
    for chunk_id in ranked_chunk_ids[:top_k]:
        chunk = dict(chunk_lookup[chunk_id])  # copy so we don't mutate the cached dict
        chunk["fused_score"] = fused_scores[chunk_id]
        results.append(chunk)
    return results


def retrieve(
    query: str,
    embedding_client: InferenceClient,
    collection,
    bm25_index: dict,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Public retrieval interface: hosted dense query embedding + local BM25,
    fused with RRF. This deployment-oriented mode does not load PyTorch or a
    cross-encoder in the web process.
    """
    dense_results = dense_search(query, embedding_client, collection, top_k=CANDIDATE_POOL_SIZE)
    bm25_results = bm25_search(query, bm25_index, top_k=CANDIDATE_POOL_SIZE)
    return reciprocal_rank_fusion([dense_results, bm25_results], k=RRF_K, top_k=top_k)


if __name__ == "__main__":
    embedding_client, collection, bm25_index = load_retriever()

    sample_questions = [
        "How do I add a custom exception handler?",
        "What is dependency injection in FastAPI?",
        "How do I validate a request body with Pydantic?",
        "How do I use HTTPException?",  # keyword-heavy - good hybrid-search test case
    ]

    for question in sample_questions:
        print("=" * 80)
        print(f"QUERY: {question}")
        print("-" * 80)
        results = retrieve(question, embedding_client, collection, bm25_index, top_k=3)
        for rank, r in enumerate(results, start=1):
            print(f"[{rank}] fused_score={r['fused_score']:.4f}  {r['header_path']}  ({r['chunk_id']})")
            preview = r["text"][:150].replace("\n", " ")
            print(f"    {preview}...")
        print()
