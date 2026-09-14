# FastAPI Documentation RAG Assistant

A production-style retrieval-augmented generation application that answers questions strictly from FastAPI's tutorial and advanced documentation. The low-memory deployment uses Hugging Face Inference for query embeddings, then combines local Chroma dense retrieval with BM25 using Reciprocal Rank Fusion before generating a cited answer with Groq.

**Live demo:** [rag-fastapi-docs-w2zp.onrender.com](https://rag-fastapi-docs-w2zp.onrender.com)

The project is deliberately evaluation-led: the retrieval and generation choices came from failed eval cases, not from adding components for their own sake.

## Highlights

- Indexed **85 documentation files into 744 structure-aware chunks**, preserving Markdown headings and code examples.
- Hybrid retrieval: hosted `bge-small-en-v1.5` query embeddings + local Chroma/BM25, fused with RRF.
- Grounded generation with numbered source citations and a refusal instruction when the context cannot answer.
- 27-question evaluation suite covering factual, code lookup, conceptual, ambiguous, and unanswerable questions.
- FastAPI service with JSON and SSE endpoints, bounded TTL cache, latency logging, and a minimal browser UI that fits Render's 512 MB instance.

## Architecture

```text
Question
  -> Hugging Face Inference (BGE query embedding) -> local Chroma --+
                                                                    +-> RRF -> top 5 chunks
  -> local keyword retrieval (BM25) --------------------------------+
                                                                         |
                                                                         v
                                           cited grounded prompt -> Groq -> answer + sources
```

## Evaluation results

The latest 27-question evaluation of the low-memory deployment configuration achieved **95% retrieval hit-rate**, **100% citation validity**, **100% refusal accuracy**, **4.95/5 faithfulness**, **5.00/5 relevance**, and **4.95/5 completeness**. A manual spot check still caught an incomplete/circular answer that the judge rated highly, which led to an explicit completeness rubric and a stronger generation instruction. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the full, candid record.

## Run locally

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12, a Groq API key, and a Hugging Face access token
with **Inference Providers** permission.

```powershell
uv sync
```

Create `.env` in the repository root:

```text
GROQ_API_KEY=your_key_here
HF_TOKEN=your_hugging_face_token_here
```

The checked-in Chroma index lets you run the app immediately:

```powershell
uv run fastapi dev
```

## Deploy to Render (512 MB compatible)

The production runtime does not install or load PyTorch, sentence-transformers,
or the cross-encoder. This keeps the app within Render's 512 MB limit; query
embeddings come from Hugging Face Inference instead.

Create a Render **Web Service** from this repository with:

```text
Build Command: uv sync --frozen --no-dev
Start Command: uv run uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Add these Render environment variables as secrets:

```text
GROQ_API_KEY=...
HF_TOKEN=...
```

Create `HF_TOKEN` from Hugging Face Settings → Access Tokens with the
**Inference Providers** permission. The free hosted provider can have cold
starts or quotas, so the API returns a 502 if Hugging Face cannot embed a
query; retrying is usually sufficient.

Open `http://127.0.0.1:8000` for the demo UI, or visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## API

`POST /api/ask`

```json
{"question": "How do I add a custom exception handler?", "top_k": 5}
```

The response contains `answer`, numbered `sources`, `latency_ms`, and `cache_hit`. `POST /api/ask/stream` returns the same generation as server-sent events (`sources`, `token`, then `done`). `GET /health` exposes readiness and cache size.

## Reproduce the pipeline

```powershell
python ingestion/fetch_docs.py
python ingestion/chunk_docs.py
uv sync --group indexing
uv run python retrieval/embed.py
uv run python eval/run_eval.py
```

The evaluation calls Groq and can be limited by your account's rate limits. The client-side token limiter handles per-model TPM pacing; inspect `eval/eval_results.jsonl` after a run for judge parse errors or per-question failures.

## Resume bullet

> Built an evaluation-driven RAG system over 85 FastAPI docs (744 chunks) using BGE embeddings, Chroma, and BM25/RRF hybrid retrieval; achieved 95% retrieval hit-rate, 100% citation/refusal accuracy, and 4.95/5 faithfulness on a 27-question suite, then shipped it as a streaming FastAPI service with caching and source citations.

## Honest limitations and next steps

The deployed low-memory mode does not run the original cross-encoder reranker, so re-run the evaluation before making quality claims for this configuration. The system also does not yet decompose compound questions into sub-queries, so multi-topic prompts can miss independent concepts. Evaluation uses an LLM judge and should be supplemented with human review for high-stakes quality claims. More detail, including examples of the failures that changed the design, is in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## Deploy to FastAPI Cloud

FastAPI Cloud supports this project's `pyproject.toml` and `uv.lock` directly.
Set `GROQ_API_KEY` and `HF_TOKEN` as cloud environment secrets, then run:

```powershell
uv run fastapi deploy
```

The explicit `api.main:app` entrypoint is configured in `pyproject.toml`.
