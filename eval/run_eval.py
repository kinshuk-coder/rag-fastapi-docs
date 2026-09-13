"""
eval/run_eval.py

Runs the full RAG pipeline (retrieval + generation) against a Q&A eval
set and scores it two ways:

1. RULE-BASED (fast, free, deterministic - no LLM call needed):
   - retrieval hit-rate: did the sources we retrieved actually include
     one of the files we expected, for questions where we have an
     expected_source_hint?
   - citation validity: does the generated answer only cite marker
     numbers that actually exist among the retrieved sources? (catches
     a model inventing a [6] when only 5 sources were given)
   - refusal correctness: for "unanswerable" questions, did the model
     actually refuse, using the fixed refusal phrase from our system
     prompt in generate.py?

2. LLM-AS-JUDGE (the RAGAS-style approach - uses an LLM to rate
   subjective qualities a rule can't capture):
   - faithfulness: is every claim in the answer actually supported by
     the retrieved context, or did the model add outside knowledge?
   - relevance: does the answer actually address the question asked?

   NOTE: this half requires a live Groq call and could not be verified
   end-to-end in the sandbox this was built in (same network
   restriction as generate.py) - the prompt-building and JSON-parsing
   logic were tested against mocked responses, but you should sanity-
   check a few real judge outputs by eye the first time you run this.
"""

import os
import sys
import re
import json
import time
from groq import Groq, RateLimitError
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "retrieval"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))

from retrieve import load_retriever, retrieve, DEFAULT_TOP_K  # noqa: E402
from generate import build_prompt, call_groq, call_groq_raw, GROQ_JUDGE_MODEL  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

EVAL_SET_PATH = os.path.join(PROJECT_ROOT, "eval", "eval_set.jsonl")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "eval", "eval_results.jsonl")


def save_results(results: list[dict]) -> None:
    """Persist after each completed question so a quota error loses no work."""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# This must match the exact refusal phrase from generation/generate.py's
# SYSTEM_PROMPT. If you edit the refusal wording there, update this too -
# it's intentionally a substring match on the distinctive part of the
# phrase, not the whole sentence, so minor wording drift doesn't break it.
REFUSAL_PHRASE = "don't have enough information"


# ---------------------------------------------------------------------
# Rule-based checks (no LLM needed - fully deterministic)
# ---------------------------------------------------------------------

def load_eval_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_hinted_filenames(hint: str) -> list[str]:
    """
    Extracts filename/path tokens from a loose expected_source_hint
    string, supporting two forms:
        'tutorial/body.md or tutorial/body-fields.md'  -> exact files
        'tutorial/security/*'                          -> any file
                                                           under that folder
    Returns an empty list for hints like 'none - should refuse'.
    """
    exact_files = re.findall(r'[\w\-/]+\.md', hint)
    wildcard_prefixes = re.findall(r'([\w\-/]+/)\*', hint)
    return exact_files + wildcard_prefixes


def retrieval_hit(sources: list[dict], hinted_files: list[str]) -> bool:
    """
    True if ANY retrieved source's chunk_id or source_file contains ANY
    of the hinted filenames as a substring. Deliberately lenient (any
    hint matching any retrieved source counts) since for ambiguous/broad
    questions we expect multiple valid answers, not one exact chunk.
    """
    if not hinted_files:
        return False  # no hint to check against (e.g. unanswerable questions)
    for source in sources:
        haystack = source["chunk_id"] + " " + source["source_file"]
        for hint_file in hinted_files:
            if hint_file in haystack:
                return True
    return False


def extract_citation_indices(answer_text: str) -> set[int]:
    """Finds all [N] style citation markers in the answer text."""
    return {int(n) for n in re.findall(r'\[(\d+)\]', answer_text)}


def citation_validity(answer_text: str, num_sources: int) -> dict:
    """
    Checks that every citation marker in the answer refers to a source
    that actually exists (e.g. flags a [6] when only 5 sources were
    retrieved - a model inventing a citation).
    Returns both whether it's valid and how many citations were found,
    since "zero citations" and "one invalid citation" are different
    failure modes worth telling apart when you read results later.
    """
    cited = extract_citation_indices(answer_text)
    invalid = {n for n in cited if n < 1 or n > num_sources}
    return {
        "num_citations_found": len(cited),
        "invalid_citations": sorted(invalid),
        "all_valid": len(invalid) == 0,
    }


def is_refusal(answer_text: str) -> bool:
    """
    True only for a GENUINE full refusal, not a partial answer that
    hedges on one sub-part while still substantively answering the rest.

    A plain substring check on REFUSAL_PHRASE was too blunt: it flagged
    answers like q022's (a real, cited, multi-part answer that ends with
    an honest "I don't have enough information to give a detailed
    example" on the one sub-part it couldn't cover) as a full refusal,
    which overstates the failure - a hedge on part of a compound
    question is a reasonable, honest response, not a refusal.

    The distinguishing signal: a genuine refusal has NOTHING to cite -
    there's no real answer content behind it. A partial hedge has real
    citations supporting the substantive part of the answer. So: only
    count it as a refusal if the phrase appears AND there are no valid
    citations anywhere in the answer.
    """
    contains_refusal_phrase = REFUSAL_PHRASE.lower() in answer_text.lower()
    has_citations = len(extract_citation_indices(answer_text)) > 0
    return contains_refusal_phrase and not has_citations


# ---------------------------------------------------------------------
# LLM-as-judge checks (require a live Groq call)
# ---------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of RAG system outputs. \
You will be given a question, the context excerpts the system had access to, \
and the system's generated answer. Respond ONLY with a JSON object, no other \
text, in exactly this shape:
{"faithfulness": <1-5 integer>, "relevance": <1-5 integer>, "completeness": <1-5 integer>, "reasoning": "<one sentence>"}

faithfulness: 5 = every claim in the answer is directly supported by the \
context, 1 = the answer relies heavily on information not in the context.
relevance: 5 = the answer directly and completely addresses the question, \
1 = the answer is off-topic or non-responsive.
completeness: 5 = the answer includes concrete, actionable detail (code, \
specific parameter/method names, exact steps) drawn from the context - NOT \
just a restatement of the question in different words. 1 = the answer is \
vague or circular (e.g. "you can do X using Y" with no concrete Y shown) \
even though the context contains specific, usable detail the answer failed \
to include. If the context genuinely has no concrete detail to give (a purely \
conceptual question), a clear conceptual explanation can still score 5 here -
completeness is judged against what the CONTEXT actually offers, not an \
absolute standard of code-or-nothing."""


def build_judge_prompt(question: str, context_text: str, answer: str) -> list[dict]:
    user_message = f"""Question: {question}

Context excerpts the system had access to:
{context_text}

System's generated answer:
{answer}

Return the JSON evaluation now."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def parse_judge_response(raw_text: str) -> dict:
    """
    Parses the judge's JSON response. LLMs occasionally wrap JSON in
    markdown fences despite instructions not to - strip those defensively
    before parsing rather than assuming a clean response.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned.strip())
    try:
        parsed = json.loads(cleaned)
        return {
            "faithfulness": int(parsed.get("faithfulness", -1)),
            "relevance": int(parsed.get("relevance", -1)),
            "completeness": int(parsed.get("completeness", -1)),
            "reasoning": parsed.get("reasoning", ""),
            "parse_error": False,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        # A parse failure is itself useful signal - surface it rather
        # than silently defaulting to a score that looks legitimate.
        return {
            "faithfulness": -1,
            "relevance": -1,
            "completeness": -1,
            "reasoning": f"JUDGE PARSE ERROR - raw response: {raw_text[:200]}",
            "parse_error": True,
        }


def judge_answer(groq_client: Groq, question: str, retrieved_chunks: list[dict], answer: str) -> dict:
    context_text = "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(retrieved_chunks, start=1)
    )
    messages = build_judge_prompt(question, context_text, answer)
    # Routed through call_groq_raw (not a direct client call) so this
    # draws from the SAME shared rate limiter as generation calls in
    # generate.py - both count against one real, shared TPM budget.
    # Deliberately a different model from generation. The configured default
    # has a separate daily budget, unlike Compound Mini which can be routed
    # through the already-exhausted Llama 3.3 70B quota.
    response = call_groq_raw(groq_client, messages, model=GROQ_JUDGE_MODEL, temperature=0.0)
    return parse_judge_response(response.choices[0].message.content)


# ---------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------

def run_eval():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(f"GROQ_API_KEY not found in {os.path.join(PROJECT_ROOT, '.env')}")
    groq_client = Groq(api_key=api_key)
    embed_model, collection, bm25_index = load_retriever()

    eval_set = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(eval_set)} eval questions from {EVAL_SET_PATH}")
    print(f"Judge model: {GROQ_JUDGE_MODEL}")

    all_results = []
    judge_available = True

    for item in eval_set:
        question = item["question"]
        category = item["category"]
        hinted_files = parse_hinted_filenames(item.get("expected_source_hint", ""))

        t0 = time.time()
        retrieved_chunks = retrieve(question, embed_model, collection, bm25_index, top_k=DEFAULT_TOP_K)
        t1 = time.time()

        messages = build_prompt(question, retrieved_chunks)
        answer = call_groq(groq_client, messages)
        t2 = time.time()

        sources = [
            {
                "marker": i + 1,
                "header_path": c["header_path"],
                "source_file": c["source_file"],
                "chunk_id": c["chunk_id"],
            }
            for i, c in enumerate(retrieved_chunks)
        ]

        print(
            f"  [{item['id']}] retrieval={t1 - t0:.2f}s  generation={t2 - t1:.2f}s",
            end="  ",
        )

        row = {
            "id": item["id"],
            "question": question,
            "category": category,
            "answer": answer,
            "retrieved_sources": [s["header_path"] for s in sources],
        }

        if category == "unanswerable":
            row["refused_correctly"] = is_refusal(answer)
            print()  # close out the timing line for unanswerable rows (no judge call)
        else:
            row["retrieval_hit"] = retrieval_hit(sources, hinted_files)
            row["citation_check"] = citation_validity(answer, len(sources))
            row["incorrectly_refused"] = is_refusal(answer)  # false refusal is also a bug worth seeing
            t3 = time.time()
            if judge_available:
                try:
                    judge_result = judge_answer(groq_client, question, retrieved_chunks, answer)
                except RateLimitError as exc:
                    judge_available = False
                    judge_result = {
                        "faithfulness": -1,
                        "relevance": -1,
                        "completeness": -1,
                        "reasoning": f"JUDGE RATE LIMITED - {exc}",
                        "parse_error": True,
                        "rate_limited": True,
                    }
                    print("judge=rate-limited; continuing without LLM judging", end="")
            else:
                judge_result = {
                    "faithfulness": -1,
                    "relevance": -1,
                    "completeness": -1,
                    "reasoning": "JUDGE SKIPPED - an earlier judge call hit Groq rate limits.",
                    "parse_error": True,
                    "rate_limited": True,
                }
            t4 = time.time()
            row["judge"] = judge_result
            print(f"judge={t4 - t3:.2f}s")

        all_results.append(row)
        save_results(all_results)

    print_summary(all_results)
    print(f"\nFull results saved to {RESULTS_PATH}")


def print_summary(results: list[dict]):
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)

    unanswerable = [r for r in results if r["category"] == "unanswerable"]
    answerable = [r for r in results if r["category"] != "unanswerable"]

    if unanswerable:
        refusal_rate = sum(r["refused_correctly"] for r in unanswerable) / len(unanswerable)
        print(f"Refusal accuracy (unanswerable questions): {refusal_rate:.0%} ({len(unanswerable)} questions)")

    if answerable:
        hit_rate = sum(r["retrieval_hit"] for r in answerable) / len(answerable)
        citation_valid_rate = sum(r["citation_check"]["all_valid"] for r in answerable) / len(answerable)
        false_refusals = sum(r["incorrectly_refused"] for r in answerable)

        judged = [r for r in answerable if not r["judge"]["parse_error"]]
        if judged:
            avg_faithfulness = sum(r["judge"]["faithfulness"] for r in judged) / len(judged)
            avg_relevance = sum(r["judge"]["relevance"] for r in judged) / len(judged)
            avg_completeness = sum(r["judge"]["completeness"] for r in judged) / len(judged)
        else:
            avg_faithfulness = avg_relevance = avg_completeness = float("nan")

        print(f"Retrieval hit-rate: {hit_rate:.0%} ({len(answerable)} questions)")
        print(f"Citation validity rate: {citation_valid_rate:.0%}")
        print(f"False refusals (should have answered but didn't): {false_refusals}")
        print(f"Avg faithfulness (LLM judge, 1-5): {avg_faithfulness:.2f}")
        print(f"Avg relevance (LLM judge, 1-5): {avg_relevance:.2f}")
        print(f"Avg completeness (LLM judge, 1-5): {avg_completeness:.2f}")
        if len(judged) < len(answerable):
            print(f"WARNING: {len(answerable) - len(judged)} judge calls failed to parse - check eval_results.jsonl")


if __name__ == "__main__":
    run_eval()
