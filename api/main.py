"""Production-facing API for the FastAPI documentation RAG assistant.

Run with: uvicorn api.main:app --reload
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from groq import Groq
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "retrieval"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from retrieve import DEFAULT_TOP_K, load_retriever, retrieve  # noqa: E402
from generate import GROQ_MODEL, build_prompt, call_groq, get_rate_limiter  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rag_api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1_500)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)


class Source(BaseModel):
    marker: int
    header_path: str
    source_file: str
    chunk_id: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    latency_ms: int
    cache_hit: bool


@dataclass
class CacheEntry:
    response: dict
    created_at: float


class ResponseCache:
    """Small bounded TTL cache; avoids repeat model calls for demo traffic."""
    def __init__(self, max_entries: int = 128, ttl_seconds: int = 900):
        self.max_entries, self.ttl_seconds = max_entries, ttl_seconds
        self.entries: OrderedDict[tuple[str, int], CacheEntry] = OrderedDict()

    def get(self, key: tuple[str, int]) -> dict | None:
        entry = self.entries.get(key)
        if entry is None or time.monotonic() - entry.created_at > self.ttl_seconds:
            self.entries.pop(key, None)
            return None
        self.entries.move_to_end(key)
        return entry.response.copy()

    def put(self, key: tuple[str, int], value: dict) -> None:
        self.entries[key] = CacheEntry(value.copy(), time.monotonic())
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)


def sources_for(chunks: list[dict]) -> list[dict]:
    return [{"marker": i, "header_path": c["header_path"], "source_file": c["source_file"], "chunk_id": c["chunk_id"]} for i, c in enumerate(chunks, 1)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required in .env before starting the API")
    logger.info("Loading retriever components from the local Hugging Face cache...")
    app.state.retriever = await asyncio.to_thread(load_retriever)
    app.state.groq_client, app.state.cache = Groq(api_key=api_key), ResponseCache()
    logger.info("RAG API ready")
    yield


app = FastAPI(title="FastAPI Docs RAG", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


def answer_sync(question: str, top_k: int) -> dict:
    model, collection, bm25_index, reranker = app.state.retriever
    chunks = retrieve(question, model, collection, bm25_index, reranker, top_k=top_k)
    return {"question": question, "answer": call_groq(app.state.groq_client, build_prompt(question, chunks)), "sources": sources_for(chunks)}


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(os.path.join(PROJECT_ROOT, "api", "static", "index.html"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": GROQ_MODEL, "cache_entries": len(app.state.cache.entries)}


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> dict:
    key, started = (request.question.strip().lower(), request.top_k), time.perf_counter()
    cached = app.state.cache.get(key)
    if cached:
        cached.update(latency_ms=round((time.perf_counter() - started) * 1000), cache_hit=True)
        return cached
    try:
        result = await asyncio.to_thread(answer_sync, request.question.strip(), request.top_k)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(502, "The language-model request failed. Please try again.") from exc
    result.update(latency_ms=round((time.perf_counter() - started) * 1000), cache_hit=False)
    app.state.cache.put(key, result)
    logger.info("ask latency_ms=%s cache_hit=false question_chars=%s", result["latency_ms"], len(request.question))
    return result


@app.post("/api/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    """Server-sent events: sources first, then answer tokens, then timing."""
    async def events() -> AsyncIterator[str]:
        started = time.perf_counter()
        try:
            model, collection, bm25_index, reranker = app.state.retriever
            chunks = await asyncio.to_thread(retrieve, request.question.strip(), model, collection, bm25_index, reranker, request.top_k)
            yield f"event: sources\ndata: {json.dumps(sources_for(chunks))}\n\n"
            messages = build_prompt(request.question.strip(), chunks)
            estimated = sum(len(m["content"].split()) for m in messages) * 13 // 10 + 500
            limiter = get_rate_limiter(GROQ_MODEL)
            await asyncio.to_thread(limiter.wait_if_needed, estimated)
            stream = await asyncio.to_thread(app.state.groq_client.chat.completions.create, model=GROQ_MODEL, messages=messages, temperature=0.1, stream=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
            limiter.record_usage(estimated)
            yield f"event: done\ndata: {json.dumps({'latency_ms': round((time.perf_counter() - started) * 1000)})}\n\n"
        except Exception:
            logger.exception("Streaming generation failed")
            yield "event: error\ndata: \"The language-model request failed. Please try again.\"\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
