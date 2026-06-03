from datetime import datetime, timezone

from fastapi.testclient import TestClient

from cortex.api import main
from cortex.api.main import create_app
from cortex.rag.retriever import Source


def test_health_returns_ok_with_pool():
    app = create_app(lifespan_context=None)
    app.state.pool = FakePool()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_503_when_pool_fails():
    app = create_app(lifespan_context=None)
    app.state.pool = FakePool(fail=True)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "knowledge base unavailable"}


def test_chat_returns_cited_sources_and_grounded_true(monkeypatch):
    app = _app_with_state(FakeChatClient("They favor async work for focus [1]."))
    captured = {}

    def fake_retrieve(conn, question, embedder, cfg, *, filters=None):
        captured["conn"] = conn
        captured["question"] = question
        captured["embedder"] = embedder
        captured["filters"] = filters
        return [_source()]

    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "  What does this person think about remote work?  ",
                "filters": {"source_platform": "twitter", "content_type": "thread"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "They favor async work for focus [1]."
    assert body["abstained"] is False
    assert body["grounded"] is True
    assert body["sources"] == [
        {
            "index": 1,
            "external_id": "1001",
            "platform": "twitter",
            "content_type": "thread",
            "url": "https://twitter.com/cortex_demo/status/1001",
            "author_handle": "cortex_demo",
            "date": "2024-05-15",
            "snippet": "best matched chunk",
            "score": 0.7235,
            "cited": True,
        }
    ]
    assert captured["question"] == "What does this person think about remote work?"
    assert captured["conn"] == "conn"
    assert captured["embedder"] == "embedder"
    assert captured["filters"].source_platform == "twitter"
    assert captured["filters"].content_type == "thread"


def test_chat_retries_once_when_answer_has_no_citation(monkeypatch):
    chat = FakeChatClient(["uncited answer", "fixed answer [1]"])
    app = _app_with_state(chat)
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: [_source()])

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "remote work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fixed answer [1]"
    assert body["grounded"] is True
    assert body["sources"][0]["cited"] is True
    assert chat.calls == 2
    assert [message["role"] for message in chat.last_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "Your answer cited no sources" in chat.last_messages[-1]["content"]


def test_chat_returns_grounded_false_when_retry_still_has_no_citation(monkeypatch):
    chat = FakeChatClient(["uncited answer", "still uncited"])
    app = _app_with_state(chat)
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: [_source()])

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "remote work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "still uncited"
    assert body["grounded"] is False
    assert body["sources"][0]["cited"] is False
    assert chat.calls == 2


def test_chat_abstains_when_no_sources(monkeypatch):
    chat = FakeChatClient("I don't have enough information in their posts to answer this.")
    app = _app_with_state(chat)
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: [])

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "unknown topic?"})

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["abstained"] is True
    assert body["grounded"] is False
    assert "No passages were retrieved" in chat.last_messages[1]["content"]


def test_chat_returns_503_without_chat_client():
    app = _app_with_state(chat=None)

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "remote work?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "DEEPSEEK_API_KEY not set"}


def test_chat_stream_returns_token_events_and_final_sources(monkeypatch):
    app = _app_with_state(FakeChatClient(["They favor ", "async work [1]."], streaming=True))
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: [_source()])

    with TestClient(app) as client:
        with client.stream("POST", "/chat/stream", json={"question": "remote work?"}) as response:
            text = "".join(response.iter_text())

    assert response.status_code == 200
    assert 'event: token\ndata: {"text":"They favor "}' in text
    assert 'event: token\ndata: {"text":"async work [1]."}' in text
    assert '"answer":"They favor async work [1]."' in text
    assert '"external_id":"1001"' in text
    assert '"cited":true' in text
    assert '"grounded":true' in text
    assert "event: done" in text


def test_chat_stream_returns_503_without_chat_client():
    app = _app_with_state(chat=None)

    with TestClient(app) as client:
        response = client.post("/chat/stream", json={"question": "remote work?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "DEEPSEEK_API_KEY not set"}


def test_chat_rejects_blank_question():
    app = _app_with_state(FakeChatClient("unused"))

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_returns_503_when_retrieval_fails(monkeypatch):
    app = _app_with_state(FakeChatClient("unused"))

    def fail_retrieve(*args, **kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(main, "retrieve", fail_retrieve)

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "remote work?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "knowledge base unavailable"}


def _app_with_state(chat):
    app = create_app(lifespan_context=None)
    app.state.pool = FakePool(conn="conn")
    app.state.embedder = "embedder"
    app.state.chat = chat
    return app


def _source():
    return Source(
        index=1,
        document_id=1,
        external_id="1001",
        platform="twitter",
        content_type="thread",
        url="https://twitter.com/cortex_demo/status/1001",
        author_handle="cortex_demo",
        created_at=datetime(2024, 5, 15, tzinfo=timezone.utc),
        snippet="best matched chunk",
        context_text="full thread text",
        score=0.723456,
    )


class FakeChatClient:
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


class FakePool:
    def __init__(self, fail=False, conn=None):
        self.fail = fail
        self.conn = conn if conn is not None else FakeConnection()

    def connection(self):
        if self.fail:
            raise RuntimeError("pool unavailable")
        return FakeConnectionContext(self.conn)


class FakeConnectionContext:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def cursor(self):
        return FakeCursor()


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql):
        self.sql = sql

    def fetchone(self):
        return (1,)
