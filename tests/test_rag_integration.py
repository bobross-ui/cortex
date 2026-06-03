import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from cortex.api.main import app
from cortex.config import settings
from cortex.embedding.embedder import SentenceTransformerEmbedder
from cortex.pipeline.index import index_export
from cortex.rag.chat import ChatClient
from cortex.rag.retriever import retrieve
from cortex.store.db import connect


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("DATABASE_URL") and not os.getenv("CI"),
        reason="DATABASE_URL is required for pgvector integration tests",
    ),
]

FIX = Path(__file__).parent / "fixtures" / "twitter"


@pytest.fixture(scope="module")
def database_url():
    return os.environ.get("DATABASE_URL", settings.database_url)


@pytest.fixture(scope="module")
def conn(database_url):
    connection = connect(database_url)
    try:
        yield connection
    finally:
        try:
            connection.rollback()
            _reset_db(connection)
        except Exception:
            pass
        connection.close()


@pytest.fixture(scope="module")
def real_embedder():
    return SentenceTransformerEmbedder(
        settings.embed_model,
        settings.embed_dim,
        settings.embed_batch_size,
    )


@pytest.fixture(scope="module")
def seeded_db(conn, real_embedder):
    _reset_db(conn)
    index_export(FIX, conn, embedder=real_embedder, cfg=settings)
    yield
    _reset_db(conn)


def test_retrieve_joins_documents_and_returns_remote_work_source_near_top(
    conn,
    real_embedder,
    seeded_db,
):
    sources = retrieve(
        conn,
        "What does this person think about remote work?",
        real_embedder,
        settings,
    )

    assert sources
    source = _source_for_external_id(sources, "1001")
    assert source.index <= 3
    assert source.url == "https://twitter.com/cortex_demo/status/1001"
    assert source.author_handle == "cortex_demo"
    assert source.context_text == _document_text(conn, "1001")


def test_retrieve_dedups_multiple_chunks_from_same_document(
    conn,
    real_embedder,
    seeded_db,
):
    sources = retrieve(
        conn,
        "is maintaining open-source projects sustainable?",
        real_embedder,
        settings,
    )
    document_ids = [source.document_id for source in sources]

    assert len(document_ids) == len(set(document_ids))


def test_hybrid_retrieval_wires_fts_and_rrf_for_exact_token_query(
    conn,
    real_embedder,
    seeded_db,
):
    cfg = settings.model_copy(update={"retrieval_hybrid": True})

    sources = retrieve(
        conn,
        "Conway's Law",
        real_embedder,
        cfg,
    )

    assert _source_for_external_id(sources, "2022").index <= 3


def test_api_chat_uses_real_retrieval_with_fake_llm(seeded_db):
    with TestClient(app) as client:
        app.state.chat = FakeChatClient("They favor async work for deep focus [2].")
        response = client.post(
            "/chat",
            json={"question": "What does this person think about remote work?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "They favor async work for deep focus [2]."
    assert body["sources"]
    source = _response_source_for_external_id(body["sources"], "1001")
    assert source["index"] <= 3
    assert source["cited"] is True
    assert body["abstained"] is False
    assert body["grounded"] is True


def test_api_chat_stream_uses_real_retrieval_with_fake_streaming_llm(seeded_db):
    chat = FakeChatClient(["They favor ", "async work for deep focus [2]."], streaming=True)

    with TestClient(app) as client:
        app.state.chat = chat
        with client.stream(
            "POST",
            "/chat/stream",
            json={"question": "What does this person think about remote work?"},
        ) as response:
            text = "".join(response.iter_text())

    assert response.status_code == 200
    assert 'event: token\ndata: {"text":"They favor "}' in text
    assert 'event: token\ndata: {"text":"async work for deep focus [2]."}' in text
    assert '"answer":"They favor async work for deep focus [2]."' in text
    assert '"external_id":"1001"' in text
    assert '"cited":true' in text
    assert '"grounded":true' in text
    assert "event: done" in text


def test_api_chat_retries_uncited_answer_with_fake_llm(seeded_db):
    chat = FakeChatClient(["uncited answer", "fixed answer [1]"])

    with TestClient(app) as client:
        app.state.chat = chat
        response = client.post(
            "/chat",
            json={"question": "What does this person think about remote work?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fixed answer [1]"
    assert body["sources"][0]["cited"] is True
    assert body["grounded"] is True
    assert chat.calls == 2


def test_api_chat_keeps_sources_when_retry_still_uncited(seeded_db):
    chat = FakeChatClient(["uncited answer", "still uncited"])

    with TestClient(app) as client:
        app.state.chat = chat
        response = client.post(
            "/chat",
            json={"question": "What does this person think about remote work?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "still uncited"
    assert body["sources"]
    assert body["sources"][0]["cited"] is False
    assert body["grounded"] is False
    assert chat.calls == 2


def test_api_chat_abstains_when_filters_match_no_sources(seeded_db):
    chat = FakeChatClient("I don't have enough information in their posts to answer this.")

    with TestClient(app) as client:
        app.state.chat = chat
        response = client.post(
            "/chat",
            json={
                "question": "What does this person think about remote work?",
                "filters": {"content_type": "does-not-exist"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["abstained"] is True
    assert body["grounded"] is False
    assert "No passages were retrieved" in chat.last_messages[1]["content"]


def test_api_health_uses_live_database(seeded_db):
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class FakeChatClient(ChatClient):
    def __init__(self, replies, *, streaming=False):
        self.replies = [replies] if isinstance(replies, str) else list(replies)
        self.streaming = streaming
        self.calls = 0
        self.last_messages = None

    def complete(self, messages):
        self.last_messages = messages
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply

    def stream(self, messages):
        self.last_messages = messages
        self.calls += 1
        if self.streaming:
            yield from self.replies
        else:
            yield self.replies[-1]


def _reset_db(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
    conn.commit()


def _document_text(conn, external_id: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT text FROM documents WHERE source_platform = 'twitter' AND external_id = %s",
            (external_id,),
        )
        return cur.fetchone()[0]


def _source_for_external_id(sources, external_id: str):
    for source in sources:
        if source.external_id == external_id:
            return source
    raise AssertionError(f"source {external_id} not found")


def _response_source_for_external_id(sources, external_id: str):
    for source in sources:
        if source["external_id"] == external_id:
            return source
    raise AssertionError(f"source {external_id} not found")
