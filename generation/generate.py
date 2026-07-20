"""
generation/generate.py

Ties retrieval (Milestone 3) to generation: takes a user question,
retrieves relevant chunks, builds a grounded prompt, and calls Groq
to produce an answer.

Design decisions:
- The prompt explicitly instructs the model to answer ONLY from the
  provided context and to say "I don't know" when the context doesn't
  contain the answer. This is a naive, unenforced form of grounding -
  the model can still ignore the instruction - but it gives us a real
  baseline faithfulness number to improve on later (Milestone 8 will
  add stronger grounding + citation enforcement).
- Each retrieved chunk is labeled with a [1], [2], [3]... marker in the
  prompt, and the model is asked to cite which marker(s) support each
  part of its answer. This is the simplest possible citation mechanism -
  good enough to evaluate, not yet robust to a model inventing a citation
  that doesn't actually support the claim (that's a Milestone 8 concern).
- Model: llama-3.1-8b-instant. Chosen for speed/cost as a baseline - if
  Milestone 5 eval shows faithfulness problems that persist even after
  fixing retrieval (Milestones 6-7), that's the signal to try a larger
  Groq model instead, since it would isolate the problem to generation
  rather than retrieval.
"""

import os
import sys
from groq import Groq
from dotenv import load_dotenv

# Make the retrieval/ folder importable regardless of which directory
# this script is run from - same project-root-anchoring principle as
# the ingestion/retrieval path fixes from Milestones 2-3.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "retrieval"))

from retrieve import load_retriever, retrieve, DEFAULT_TOP_K  # noqa: E402
from rate_limiter import TokenRateLimiter  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GROQ_MODEL = "llama-3.1-8b-instant"

# A SECOND, DIFFERENT model used only for judging (eval/run_eval.py).
# Confirmed via console.groq.com/settings/limits that Groq enforces TPM
# limits PER MODEL, not account-wide.
#
# HISTORY: originally set to llama-3.3-70b-versatile (12K TPM vs
# llama-3.1-8b-instant's 6K TPM) to give judging its own separate
# budget. That worked for TPM, but exposed a SEPARATE constraint our
# rate limiter didn't track at all: tokens-PER-DAY (TPD), not just
# per-minute. llama-3.3-70b-versatile has a 100K TPD cap, which several
# eval re-runs in one day exhausted (a real 429 mid-run). Our
# TokenRateLimiter only models a 60-second rolling window - it has no
# concept of a daily budget, so it couldn't see this coming.
#
# Do NOT use groq/compound-mini as an escape hatch for Llama 3.3's daily
# cap. Compound Mini can route requests to Llama 3.3 70B, so its limits are
# ultimately charged to the underlying model. That means it can still return
# a 70B TPD error even though "compound-mini" itself shows no TPD column.
#
# GPT-OSS 20B has its own 200K TPD budget on the current free-tier limits and
# is sufficient for the constrained JSON judge task. An environment variable
# makes this easy to change without editing source (for example when using a
# different Groq tier): GROQ_JUDGE_MODEL=openai/gpt-oss-20b.
#
# NOTE: these limits were read off the Groq console on a specific date
# and may change or vary by account tier - if rate-limit waits (or new
# TPD errors) return unexpectedly, re-check the console rather than
# trusting this comment.
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")

# Per-model TPM budgets, sourced from the same console page. Used to
# size each model's independent rate limiter (see get_rate_limiter()).
MODEL_TPM_LIMITS = {
    GROQ_MODEL: 6000,
    "openai/gpt-oss-20b": 8000,
}
# Conservative fallback for any model used without an entry above -
# better to under-budget (extra waiting) than over-budget (real 429s).
DEFAULT_TPM_FALLBACK = 6000

# One rate limiter PER MODEL, created lazily on first use - see
# get_rate_limiter(). Replaces the old single shared singleton, which
# incorrectly assumed generation and judging drew from one pool.
_rate_limiters: dict[str, TokenRateLimiter] = {}


def get_rate_limiter(model: str) -> TokenRateLimiter:
    if model not in _rate_limiters:
        tpm = MODEL_TPM_LIMITS.get(model)
        if tpm is None:
            print(
                f"WARNING: no known TPM limit for model '{model}' - falling back to "
                f"{DEFAULT_TPM_FALLBACK}. Check console.groq.com/settings/limits and "
                f"add it to MODEL_TPM_LIMITS for accurate pacing."
            )
            tpm = DEFAULT_TPM_FALLBACK
        _rate_limiters[model] = TokenRateLimiter(max_tokens_per_minute=tpm)
    return _rate_limiters[model]


def estimate_message_tokens(messages: list[dict], completion_buffer: int = 500) -> int:
    """
    Rough pre-call token estimate for the rate limiter to decide whether
    to wait. Same word-count-based approximation used in
    ingestion/chunk_docs.py's token_count() - doesn't need to be exact,
    since record_usage() always overwrites with the REAL total_tokens
    from the response afterward. completion_buffer accounts for output
    tokens we can't know until after the call completes.
    """
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.3) + completion_buffer

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about FastAPI \
using ONLY the documentation excerpts provided to you.

Rules:
1. Answer using ONLY information found in the provided context. Do not use \
any outside knowledge, even if you are confident it is correct.
2. Every claim in your answer must be supported by at least one of the \
numbered context excerpts. Cite the excerpt number(s) in square brackets \
right after the claim, like this: "FastAPI supports dependency injection [1]."
3. If the context does not contain enough information to answer the question, \
say "I don't have enough information in the provided documentation to answer \
that." Do not guess or fill gaps with outside knowledge.
4. Keep answers concise and technical - this is for a developer audience.
5. If any context excerpt contains a code example, parameter name, method \
name, or other concrete technical detail relevant to the question, you MUST \
include that concrete detail in your answer - do not merely restate the \
question in different words. A vague answer like "you can do X using Y" is \
NOT acceptable if the context shows exactly HOW to do it; show the actual \
code, parameter, or method name from the context instead."""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    """
    Builds the messages list for the Groq chat completion call.

    retrieved_chunks is the list of dicts returned by retrieve() in
    retrieval/retrieve.py - each has {chunk_id, text, header_path,
    source_file, distance}.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[{i}] (source: {chunk['header_path']})\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    user_message = f"""Context excerpts:

{context_text}

Question: {question}

Answer the question using only the context above, following the citation rules."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def call_groq_raw(client: Groq, messages: list[dict], model: str = GROQ_MODEL, temperature: float = 0.1):
    """
    Makes the actual Groq call, paced by the shared rate limiter.
    Returns the FULL response object (not just the text) so callers
    that need response.usage - like the pre-call estimate correction
    below - can use it. call_groq() and eval/run_eval.py's
    judge_answer() both build on this, so every Groq call in the
    project - generation AND judging - draws from the same tracked
    TPM budget.
    """
    estimated_tokens = estimate_message_tokens(messages)
    limiter = get_rate_limiter(model)
    limiter.wait_if_needed(estimated_tokens)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    actual_tokens = response.usage.total_tokens
    limiter.record_usage(actual_tokens)

    return response


def call_groq(client: Groq, messages: list[dict], model: str = GROQ_MODEL) -> str:
    response = call_groq_raw(client, messages, model, temperature=0.1)
    return response.choices[0].message.content


def generate_answer(
    question: str,
    embed_model,
    collection,
    bm25_index: dict,
    reranker,
    groq_client: Groq,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Full pipeline: retrieve -> build prompt -> call Groq -> return
    answer plus the sources actually used, so callers (CLI, eval script)
    can display or check citations without re-deriving them.

    NOTE: as of Milestone 7, retrieve() is hybrid search + cross-encoder
    reranking, so this now needs a reranker in addition to the
    embedding model, Chroma collection, and bm25_index.
    """
    retrieved_chunks = retrieve(question, embed_model, collection, bm25_index, reranker, top_k=top_k)
    messages = build_prompt(question, retrieved_chunks)
    answer_text = call_groq(groq_client, messages)

    return {
        "question": question,
        "answer": answer_text,
        "sources": [
            {
                "marker": i + 1,
                "header_path": chunk["header_path"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
            }
            for i, chunk in enumerate(retrieved_chunks)
        ],
    }


if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Make sure it's set in your .env file "
            f"at {os.path.join(PROJECT_ROOT, '.env')}"
        )

    groq_client = Groq(api_key=api_key)
    embed_model, collection, bm25_index, reranker = load_retriever()

    sample_questions = [
        "How do I add a custom exception handler?",
        "What is dependency injection in FastAPI?",
        "How do I validate a request body with Pydantic?",
        "How do I make FastAPI send emails automatically?",  # not in docs - should refuse
    ]

    for question in sample_questions:
        print("=" * 80)
        print(f"Q: {question}")
        print("-" * 80)
        result = generate_answer(question, embed_model, collection, bm25_index, reranker, groq_client)
        print(result["answer"])
        print("\nSources:")
        for src in result["sources"]:
            print(f"  [{src['marker']}] {src['header_path']}")
        print()
