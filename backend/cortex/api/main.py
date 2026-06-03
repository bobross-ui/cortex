from __future__ import annotations

from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, field_validator

from cortex.config import settings
from cortex.embedding.embedder import SentenceTransformerEmbedder
from cortex.rag.chat import DeepSeekChatClient
from cortex.rag.prompt import build_messages, cited_indices
from cortex.rag.retriever import Source, retrieve
from cortex.store.db import connect


class ChatFilters(BaseModel):
    source_platform: str | None = None
    content_type: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    filters: ChatFilters | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class SourceResponse(BaseModel):
    index: int
    external_id: str
    platform: str
    content_type: str
    url: str | None
    author_handle: str | None
    date: str | None
    snippet: str
    score: float
    cited: bool


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    abstained: bool
    grounded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    connect(settings.database_url).close()
    app.state.pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=8,
        configure=lambda conn: register_vector(conn),
        open=True,
    )
    app.state.embedder = SentenceTransformerEmbedder(
        settings.embed_model,
        settings.embed_dim,
        settings.embed_batch_size,
    )
    app.state.chat = (
        DeepSeekChatClient(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
            settings.chat_temperature,
            settings.chat_max_tokens,
            disable_thinking=settings.deepseek_disable_thinking,
            timeout_s=settings.chat_timeout_s,
            max_retries=settings.chat_max_retries,
        )
        if settings.deepseek_api_key
        else None
    )
    try:
        yield
    finally:
        app.state.pool.close()


def create_app(lifespan_context=lifespan) -> FastAPI:
    api = FastAPI(lifespan=lifespan_context)

    @api.get("/health")
    def health(request: Request) -> dict[str, str]:
        try:
            with request.app.state.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="knowledge base unavailable") from exc
        return {"status": "ok"}

    @api.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        chat_client = getattr(request.app.state, "chat", None)
        if chat_client is None:
            raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY not set")

        sources = _retrieve_or_raise(payload, request)
        messages = build_messages(payload.question, sources)
        answer = _complete_or_raise(chat_client, messages)
        cited = cited_indices(answer, len(sources))
        if sources and not cited:
            retry_messages = [
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "Your answer cited no sources. Re-answer using only the passages above "
                        "and cite each claim with [n]."
                    ),
                },
            ]
            answer = _complete_or_raise(chat_client, retry_messages)
            cited = cited_indices(answer, len(sources))

        return _chat_response(answer, sources, cited)

    @api.post("/chat/stream")
    def chat_stream(payload: ChatRequest, request: Request):
        chat_client = getattr(request.app.state, "chat", None)
        if chat_client is None:
            raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY not set")

        sources = _retrieve_or_raise(payload, request)
        messages = build_messages(payload.question, sources)

        def event_stream():
            chunks = []
            try:
                for chunk in chat_client.stream(messages):
                    chunks.append(chunk)
                    yield _sse("token", {"text": chunk})
            except Exception as exc:
                yield _sse("error", _llm_stream_error_payload(exc))
                return

            answer = "".join(chunks)
            cited = cited_indices(answer, len(sources))
            response = _chat_response(answer, sources, cited)
            yield _sse("sources", response.model_dump())
            yield _sse("done", {})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return api


def _retrieve_or_raise(payload: ChatRequest, request: Request) -> list[Source]:
    try:
        with request.app.state.pool.connection() as conn:
            return retrieve(
                conn,
                payload.question,
                request.app.state.embedder,
                settings,
                filters=payload.filters,
            )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="knowledge base unavailable") from exc


def _complete_or_raise(chat_client, messages: list[dict]) -> str:
    try:
        return chat_client.complete(messages)
    except Exception as exc:
        _raise_llm_http_error(exc)
        raise


def _raise_llm_http_error(exc: Exception) -> None:
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError
    except ImportError:
        raise HTTPException(status_code=502, detail="LLM upstream error") from exc

    if isinstance(exc, APITimeoutError):
        raise HTTPException(status_code=504, detail="LLM upstream timeout") from exc
    if isinstance(exc, (APIConnectionError, APIStatusError)):
        raise HTTPException(status_code=502, detail="LLM upstream error") from exc
    raise exc


def _llm_stream_error_payload(exc: Exception) -> dict[str, str]:
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError
    except ImportError:
        return {"detail": "LLM upstream error"}

    if isinstance(exc, APITimeoutError):
        return {"detail": "LLM upstream timeout"}
    if isinstance(exc, (APIConnectionError, APIStatusError)):
        return {"detail": "LLM upstream error"}
    return {"detail": "LLM upstream error"}


def _chat_response(answer: str, sources: list[Source], cited: set[int]) -> ChatResponse:
    for source in sources:
        source.cited = source.index in cited

    return ChatResponse(
        answer=answer,
        sources=[_source_response(source) for source in sources],
        abstained=len(sources) == 0,
        grounded=bool(sources) and bool(cited),
    )


def _source_response(source: Source) -> SourceResponse:
    return SourceResponse(
        index=source.index,
        external_id=source.external_id,
        platform=source.platform,
        content_type=source.content_type,
        url=source.url,
        author_handle=source.author_handle,
        date=source.created_at.date().isoformat() if source.created_at else None,
        snippet=source.snippet,
        score=round(source.score, 4),
        cited=source.cited,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


app = create_app()
