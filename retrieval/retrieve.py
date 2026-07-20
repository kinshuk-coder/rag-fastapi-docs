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
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from embed import CHROMA_DB_PATH, COLLECTION_NAME, MODEL_NAME, embed_query

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

# How many chunks the RRF fusion stage hands to the cross-encoder for
# reranking, BEFORE cutting down to the final DEFAULT_TOP_K. Wider than
# top_k for the same reason CANDIDATE_POOL_SIZE is wider than top_k:
# the cross-encoder needs real candidates to choose among, not just the
# hybrid stage's already-final answer. If this equaled DEFAULT_TOP_K,
# reranking could only ever reorder the same 5 chunks hybrid search
# picked - it could never pull in a 6th-or-lower ranked chunk that the
# cross-encoder judges as actually more relevant.
RERANK_CANDIDATE_POOL_SIZE = 15

# A standard, well-established cross-encoder for passage reranking -
# small enough to run on CPU at query time without noticeable latency
# for a candidate pool this size (15 pairs).
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> list[str]:
    """Simple whitespace/word tokenizer for BM25 - lowercased word tokens."""
    return re.findall(r"\w+", text.lower())


def load_retriever():
    """
    Loads everything needed for hybrid retrieval + reranking:
    - the embedding model (for dense search)
    - the Chroma collection (for dense search)
    - a BM25 index built from the SAME chunks stored in Chroma (for
      keyword search)
    - a cross-encoder reranker (for precision re-scoring of candidates)

    Building the BM25 index from collection.get() rather than re-reading
    chunks.jsonl directly keeps a single source of truth - if Chroma's
    index and chunks.jsonl ever drifted out of sync, we want BM25 to
    reflect what's actually indexed, not what's on disk elsewhere.
    """
    model = SentenceTransformer(MODEL_NAME)
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

    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    return model, collection, bm25_index, reranker


def dense_search(query: str, model: SentenceTransformer, collection, top_k: int) -> list[dict]:
    """Same logic as Milestone 3's retrieve(), renamed - now one of two inputs to fusion."""
    query_embedding = embed_query(model, query)
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


def rerank(query: str, candidates: list[dict], reranker: CrossEncoder, top_k: int) -> list[dict]:
    """
    Re-scores each candidate chunk against the query using a cross-
    encoder (query and chunk text fed TOGETHER into one model, unlike
    the bi-encoder dense search which embeds them separately). Returns
    the top_k candidates sorted by this more precise, more expensive
    score.

    Cross-encoder scores are raw logits, not bounded like cosine
    similarity or comparable to BM25/fused_score - they're only
    meaningful relative to each other WITHIN this one call, not across
    different queries or against the earlier hybrid scores.
    """
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


def retrieve(
    query: str,
    model: SentenceTransformer,
    collection,
    bm25_index: dict,
    reranker: CrossEncoder,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Public retrieval interface: hybrid search (dense + BM25 fused via
    RRF) narrows the full corpus down to RERANK_CANDIDATE_POOL_SIZE
    candidates, then the cross-encoder reranks those candidates down to
    the final top_k. NOTE the added reranker parameter versus
    Milestone 6's version - see the wiring-changes note for callers.
    """
    dense_results = dense_search(query, model, collection, top_k=CANDIDATE_POOL_SIZE)
    bm25_results = bm25_search(query, bm25_index, top_k=CANDIDATE_POOL_SIZE)
    fused_candidates = reciprocal_rank_fusion(
        [dense_results, bm25_results], k=RRF_K, top_k=RERANK_CANDIDATE_POOL_SIZE
    )
    return rerank(query, fused_candidates, reranker, top_k=top_k)


if __name__ == "__main__":
    model, collection, bm25_index, reranker = load_retriever()

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
        results = retrieve(question, model, collection, bm25_index, reranker, top_k=3)
        for rank, r in enumerate(results, start=1):
            print(f"[{rank}] rerank_score={r['rerank_score']:.4f}  (fused_score={r['fused_score']:.4f})  {r['header_path']}  ({r['chunk_id']})")
            preview = r["text"][:150].replace("\n", " ")
            print(f"    {preview}...")
        print()