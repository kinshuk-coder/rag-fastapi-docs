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

from retrieve import load_retriever, retrieve  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GROQ_MODEL = "llama-3.1-8b-instant"

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
4. Keep answers concise and technical - this is for a developer audience."""


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


def call_groq(client: Groq, messages: list[dict], model: str = GROQ_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,  # low temperature - we want faithful, not creative
    )
    return response.choices[0].message.content


def generate_answer(
    question: str,
    embed_model,
    collection,
    groq_client: Groq,
    top_k: int = 5,
) -> dict:
    """
    Full pipeline: retrieve -> build prompt -> call Groq -> return
    answer plus the sources actually used, so callers (CLI, eval script)
    can display or check citations without re-deriving them.
    """
    retrieved_chunks = retrieve(question, embed_model, collection, top_k=top_k)
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
    embed_model, collection = load_retriever()

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
        result = generate_answer(question, embed_model, collection, groq_client)
        print(result["answer"])
        print("\nSources:")
        for src in result["sources"]:
            print(f"  [{src['marker']}] {src['header_path']}")
        print()