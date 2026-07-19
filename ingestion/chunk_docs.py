"""
ingestion/chunk_docs.py

Loads markdown files from the FastAPI docs subset, splits them into
structure-aware chunks (by header, then by paragraph/code-block if a
section is too large), and saves the result as JSONL.

Design goals:
- Never split inside a fenced code block.
- Keep header hierarchy as metadata (useful for citations later).
- Keep chunk sizes roughly bounded by max_tokens (rough word-based estimate).
"""

import os
import re
import json
import random

# Same project-root anchoring as fetch_docs.py - see the comment there
# for why this matters.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------
# Step 1: Load raw markdown files
# ---------------------------------------------------------------------

def load_markdown_files(root_dirs: list[str]) -> list[dict]:
    """
    Walks each root_dir recursively, reads every .md file found,
    and returns a list of {"source_path": ..., "raw_text": ...}.
    """
    documents = []
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".md"):
                    filepath = os.path.normpath(os.path.join(dirpath, filename))
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append({
                        "source_path": filepath,
                        "raw_text": content,
                    })
    return documents


# ---------------------------------------------------------------------
# Step 1.5: Resolve FastAPI's custom {* path hl[...] *} snippet syntax
# ---------------------------------------------------------------------
#
# Modern FastAPI docs don't inline code with ``` fences most of the time.
# Instead they reference an external Python file like:
#     {* ../../docs_src/body/tutorial001_py310.py hl[16] *}
# which their doc-site build tool resolves at render time by reading the
# real source file. We do the same thing here at ingestion time, so the
# actual code ends up inside the chunk text instead of a placeholder.
#
# IMPORTANT GOTCHA: the relative path in the reference is NOT relative to
# the .md file's real filesystem location. It's relative to a virtual
# root used by the docs build tooling. Naively joining it with the .md
# file's dirname resolves to the wrong place. The reliable fix is to
# anchor on the literal "docs_src/" folder name in the resolved path and
# treat everything from there onward as relative to the repo root.

SNIPPET_PATTERN = re.compile(r'\{\*\s*(\S+?)(?:\s+[a-z]+\[[^\]]*\])*\s*\*\}')


def resolve_snippet_path(md_file_path: str, rel_path: str, repo_root: str) -> str | None:
    """
    Resolves a snippet reference's relative path to a real, readable file.
    Returns an absolute-ish path (relative to repo_root) or None if it
    can't be resolved.
    """
    naive_join = os.path.normpath(os.path.join(os.path.dirname(md_file_path), rel_path))
    anchor = "docs_src" + os.sep
    idx = naive_join.find(anchor)
    if idx == -1:
        return None
    repo_relative = naive_join[idx:]
    full_path = os.path.join(repo_root, repo_relative)
    return full_path if os.path.exists(full_path) else None


def resolve_snippets_in_text(text: str, md_file_path: str, repo_root: str) -> str:
    """
    Replaces every {* path hl[...] *} reference in `text` with the actual
    source code, wrapped in a fenced code block so downstream chunking
    treats it as an atomic code unit (see split_into_atomic_blocks).

    If a reference can't be resolved, it's left as-is rather than
    silently dropped - better to have a visible gap you can grep for
    than a chunk that quietly lost information.
    """
    def replace_match(match: re.Match) -> str:
        rel_path = match.group(1)
        resolved_path = resolve_snippet_path(md_file_path, rel_path, repo_root)
        if resolved_path is None:
            return match.group(0)  # leave the original placeholder untouched
        with open(resolved_path, "r", encoding="utf-8") as f:
            code = f.read()
        return f"```python\n{code}\n```"

    return SNIPPET_PATTERN.sub(replace_match, text)


# ---------------------------------------------------------------------
# Step 2: Split a document into sections by markdown header
# ---------------------------------------------------------------------

HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')


def split_by_headers(text: str) -> list[tuple[str, str]]:
    """
    Splits markdown text into (header_path, section_text) tuples.

    header_path is the full breadcrumb, e.g. "Path Parameters > Data Validation".
    Lines inside fenced code blocks are never treated as headers, even if
    they start with '#' (e.g. a Python comment).
    """
    lines = text.split("\n")
    sections = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current_lines: list[str] = []
    in_code_block = False

    def flush():
        if current_lines and any(line.strip() for line in current_lines):
            header_path = " > ".join(title for _, title in stack) if stack else "(intro)"
            sections.append((header_path, "\n".join(current_lines).strip()))

    for line in lines:
        stripped = line.strip()

        # Toggle code-block state on fence lines; fence lines are content,
        # so they get appended below regardless of header logic.
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block:
            match = HEADER_PATTERN.match(line)
            if match:
                flush()
                current_lines = []
                level = len(match.group(1))
                title = match.group(2).strip()
                # Pop any headers at the same or deeper level - they're
                # siblings or children of a section that just ended.
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                continue

        current_lines.append(line)

    flush()
    return sections


# ---------------------------------------------------------------------
# Step 3: Split a (possibly too-large) section into atomic blocks
# ---------------------------------------------------------------------

def split_into_atomic_blocks(text: str) -> list[str]:
    """
    Splits text into a list of atomic blocks, where each block is either:
      - one complete fenced code block (```...```), or
      - one paragraph of prose (contiguous non-blank lines).

    These blocks are "atomic" in the sense that later packing logic
    will never split them further.
    """
    lines = text.split("\n")
    blocks = []
    current_block: list[str] = []
    in_code_block = False

    def flush():
        if current_block:
            content = "\n".join(current_block).strip()
            if content:
                blocks.append(content)

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code_block:
                # Starting a code block: flush any prose gathered so far,
                # then start a fresh block containing the fence line.
                flush()
                current_block = [line]
                in_code_block = True
            else:
                # Ending a code block: include the closing fence and
                # flush the whole code block as one atomic unit.
                current_block.append(line)
                blocks.append("\n".join(current_block).strip())
                current_block = []
                in_code_block = False
            continue

        if in_code_block:
            current_block.append(line)
            continue

        # Regular prose: blank line means "paragraph boundary"
        if stripped == "":
            flush()
            current_block = []
        else:
            current_block.append(line)

    flush()
    return blocks


def token_count(text: str) -> int:
    """
    Rough token estimate. Word count * 1.3 is a reasonable proxy for
    English text with code mixed in - not exact, good enough to bound
    chunk sizes. Swap for a real tokenizer later if you need precision.
    """
    return int(len(text.split()) * 1.3)


def split_preserving_code_blocks(text: str, max_tokens: int = 800) -> list[str]:
    """
    Greedily packs atomic blocks (from split_into_atomic_blocks) into
    chunks of at most ~max_tokens, never splitting a block in half.

    Note: if a single block (e.g. a huge code sample) exceeds max_tokens
    on its own, it becomes its own oversized chunk rather than being cut -
    that's an intentional tradeoff (see README notes on chunking strategy).
    """
    blocks = split_into_atomic_blocks(text)
    chunks = []
    current_chunk_blocks: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = token_count(block)
        if current_tokens + block_tokens > max_tokens and current_chunk_blocks:
            chunks.append("\n\n".join(current_chunk_blocks))
            current_chunk_blocks = []
            current_tokens = 0
        current_chunk_blocks.append(block)
        current_tokens += block_tokens

    if current_chunk_blocks:
        chunks.append("\n\n".join(current_chunk_blocks))

    return chunks


# ---------------------------------------------------------------------
# Step 4: Tie it together - full chunking pipeline for one document
# ---------------------------------------------------------------------

def make_relative_doc_path(source_path: str) -> str:
    """
    Strips the local filesystem prefix down to a stable, readable
    path anchored at 'docs/en/docs/' - e.g.
    'data/raw/fastapi_repo/docs/en/docs/tutorial/dependencies/index.md'
    becomes 'tutorial/dependencies/index.md'.

    Falls back to the full source_path if the marker isn't found,
    rather than raising - a chunk_id is still better slightly-uglier
    than the pipeline crashing on an unexpected path shape.
    """
    marker = os.path.join("docs", "en", "docs") + os.sep
    idx = source_path.find(marker)
    if idx == -1:
        return source_path
    return source_path[idx + len(marker):]


def make_chunk(text: str, header_path: str, source_path: str, chunk_index: int) -> dict:
    relative_path = make_relative_doc_path(source_path).replace(os.sep, "/")
    chunk_id = f"{relative_path}::{chunk_index}"
    return {
        "chunk_id": chunk_id,
        "source_file": source_path,
        "header_path": header_path,
        "chunk_index": chunk_index,
        "text": text,
        "token_count_estimate": token_count(text),
    }


def chunk_markdown(text: str, source_path: str, max_tokens: int = 800) -> list[dict]:
    """
    Full chunking pipeline for a single markdown document:
    1. Split by header into sections.
    2. For each section, if it's small enough, keep it as one chunk.
       Otherwise, split it further while preserving code blocks.
    """
    sections = split_by_headers(text)
    chunks = []
    chunk_index = 0

    for header_path, section_text in sections:
        if token_count(section_text) <= max_tokens:
            chunks.append(make_chunk(section_text, header_path, source_path, chunk_index))
            chunk_index += 1
        else:
            sub_texts = split_preserving_code_blocks(section_text, max_tokens)
            for sub_text in sub_texts:
                chunks.append(make_chunk(sub_text, header_path, source_path, chunk_index))
                chunk_index += 1

    return chunks


# ---------------------------------------------------------------------
# Step 5: Save to JSONL
# ---------------------------------------------------------------------

def save_chunks_jsonl(chunks: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Step 6: Inspection helper - print random sample chunks for review
# ---------------------------------------------------------------------

def print_random_sample(chunks: list[dict], n: int = 5):
    sample = random.sample(chunks, min(n, len(chunks)))
    for chunk in sample:
        print("=" * 80)
        print(f"chunk_id:      {chunk['chunk_id']}")
        print(f"header_path:   {chunk['header_path']}")
        print(f"source_file:   {chunk['source_file']}")
        print(f"token_est:     {chunk['token_count_estimate']}")
        print("-" * 80)
        print(chunk["text"])
        print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from fetch_docs import fetch_docs, CLONE_PATH

    doc_root_dirs = fetch_docs()
    raw_documents = load_markdown_files(doc_root_dirs)
    print(f"Loaded {len(raw_documents)} markdown files.")

    all_chunks = []
    unresolved_snippet_count = 0
    for doc in raw_documents:
        resolved_text = resolve_snippets_in_text(
            doc["raw_text"], doc["source_path"], repo_root=CLONE_PATH
        )
        unresolved_snippet_count += len(SNIPPET_PATTERN.findall(resolved_text))
        doc_chunks = chunk_markdown(resolved_text, doc["source_path"], max_tokens=800)
        all_chunks.extend(doc_chunks)

    if unresolved_snippet_count:
        print(f"WARNING: {unresolved_snippet_count} snippet references could not be resolved.")

    print(f"Produced {len(all_chunks)} chunks.")

    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "chunks.jsonl")
    save_chunks_jsonl(all_chunks, output_path)
    print(f"Saved chunks to {output_path}")

    print("\nRandom sample for manual review:\n")
    print_random_sample(all_chunks, n=5)