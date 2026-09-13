---
title: FastAPI Documentation RAG Assistant
emoji: "🔎"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
python_version: 3.11
suggested_hardware: cpu-basic
short_description: Hybrid RAG over FastAPI documentation with cited answers.
---

# FastAPI Documentation RAG Assistant

A production-style retrieval-augmented generation application that answers questions strictly from FastAPI's tutorial and advanced documentation. It combines dense semantic search, BM25 keyword retrieval, Reciprocal Rank Fusion, and cross-encoder reranking before generating a cited answer with Groq.

The project is deliberately evaluation-led: the retrieval and generation choices came from failed eval cases, not from adding components for their own sake.

## Highlights

- Indexed **85 documentation files into 744 structure-aware chunks**, preserving Markdown headings and code examples.
- Hybrid retrieval: `bge-small-en-v1.5` dense search + BM25, fused with RRF and reranked with `ms-marco-MiniLM-L-6-v2`.
- Grounded generation with numbered source citations and a refusal instruction when the context cannot answer.
- 27-question evaluation suite covering factual, code lookup, conceptual, ambiguous, and unanswerable questions.
- FastAPI service with a JSON endpoint, SSE streaming endpoint, startup-time model loading, bounded TTL cache, latency logging, and a minimal browser UI.

## Architecture

```text
Question
  -> dense retrieval (Chroma + BGE) ─┐
                                    ├-> RRF -> cross-encoder reranker -> top 5 chunks
  -> keyword retrieval (BM25) ──────┘                                      |
                                                                           v
                                    cited grounded prompt -> Groq -> answer + sources
```

## Evaluation results

The Milestone 7 evaluation run achieved 100% retrieval hit rate, citation validity, and refusal accuracy, with **4.90/5 faithfulness** on the 27-question set. A manual spot check still caught an incomplete/circular answer that the judge rated highly, which led to an explicit completeness rubric and a stronger generation instruction. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the full, candid record.

## Run locally

Requirements: Python 3.10+ and a Groq API key.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in the repository root:

```text
GROQ_API_KEY=your_key_here
```

The checked-in Chroma index lets you run the app immediately:

```powershell
uvicorn api.main:app --reload
```

The embedding and reranker models load from the local Hugging Face cache by
default, so normal starts do not contact Hugging Face. If this is a new
machine and a model has not yet been cached, download it once with:

```powershell
$env:HF_LOCAL_FILES_ONLY="false"
uvicorn api.main:app --reload
```

After the models finish downloading, stop the server and start it normally.

## Deploy to Hugging Face Spaces

This repository is configured as a Docker Space. Create a new Space and select
**Docker** as its SDK, then push this repository to the Space repository. The
Docker image pre-downloads the embedding and reranker models during its build,
and serves the app on port 7860. Add `GROQ_API_KEY` under **Settings → Secrets**
in the Space; never commit it to the repository.

```powershell
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/fastapi-docs-rag
git push space main
```

The first build is slower because it installs PyTorch and preloads both
retrieval models. Subsequent app starts load those models from the image cache.

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
python retrieval/embed.py
python eval/run_eval.py
```

The evaluation calls Groq and can be limited by your account's rate limits. The client-side token limiter handles per-model TPM pacing; inspect `eval/eval_results.jsonl` after a run for judge parse errors or per-question failures.

## Resume bullet

> Built an evaluation-driven RAG system over 85 FastAPI docs (744 chunks) using BGE embeddings, Chroma, BM25/RRF hybrid retrieval, and cross-encoder reranking; achieved 100% retrieval/citation/refusal accuracy and 4.90/5 faithfulness on a 27-question suite, then shipped it as a streaming FastAPI service with caching and source citations.

## Honest limitations and next steps

The system does not yet decompose compound questions into sub-queries, so multi-topic prompts can miss independent concepts. Evaluation uses an LLM judge and should be supplemented with human review for high-stakes quality claims. More detail, including examples of the failures that changed the design, is in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).
