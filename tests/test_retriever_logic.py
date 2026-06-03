from datetime import datetime, timezone
from types import SimpleNamespace

from cortex.rag import retriever
from cortex.rag.retriever import group_by_document, retrieve, rrf_fuse


def test_group_by_document_keeps_best_scoring_chunk_and_orders_documents():
    hits = [
        {"id": 1, "document_id": 10, "text": "lower doc 10", "score": 0.2},
        {"id": 2, "document_id": 20, "text": "best doc 20", "score": 0.8},
        {"id": 3, "document_id": 10, "text": "best doc 10", "score": 0.7},
        {"id": 4, "document_id": 30, "text": "doc 30", "score": 0.4},
    ]

    grouped = group_by_document(hits)

    assert [hit["document_id"] for hit in grouped] == [20, 10, 30]
    assert grouped[1]["id"] == 3
    assert grouped[1]["text"] == "best doc 10"


def test_rrf_fuse_orders_ids_by_reciprocal_rank_score():
    assert rrf_fuse(["a", "b", "c"], ["c", "a", "d"], k=60) == ["a", "c", "b", "d"]


def test_retrieve_embeds_query_with_instruction_joins_documents_and_bounds_context(monkeypatch):
    cfg = SimpleNamespace(
        bge_query_instruction="Represent this sentence for searching relevant passages:",
        retrieval_candidates=20,
        retrieval_k=2,
        retrieval_hybrid=False,
        chat_context_char_cap=24,
    )
    embedder = FakeEmbedder([[0.1, 0.2, 0.3]])
    captured = {}

    def fake_search(conn, query_embedding, k, **filters):
        captured["conn"] = conn
        captured["query_embedding"] = query_embedding
        captured["k"] = k
        captured["filters"] = filters
        return [
            {"id": 1, "document_id": 101, "text": "older chunk", "score": 0.4},
            {"id": 2, "document_id": 101, "text": "best chunk", "score": 0.9},
            {"id": 3, "document_id": 202, "text": "fallback chunk", "score": 0.8},
            {"id": 4, "document_id": 303, "text": "outside k", "score": 0.7},
        ]

    def fake_fetch_documents(conn, document_ids):
        captured["document_ids"] = document_ids
        return {
            101: {
                "source_platform": "twitter",
                "external_id": "1001",
                "content_type": "thread",
                "author_handle": "cortex_demo",
                "url": "https://twitter.com/cortex_demo/status/1001",
                "text": "short full thread text",
                "created_at": datetime(2024, 5, 15, tzinfo=timezone.utc),
                "metadata": {},
            },
            202: {
                "source_platform": "twitter",
                "external_id": "2002",
                "content_type": "post",
                "author_handle": "cortex_demo",
                "url": "https://twitter.com/cortex_demo/status/2002",
                "text": "this full document is intentionally longer than the cap",
                "created_at": None,
                "metadata": {},
            },
        }

    monkeypatch.setattr(retriever.repository, "search", fake_search)
    monkeypatch.setattr(retriever.repository, "fetch_documents", fake_fetch_documents)

    sources = retrieve(
        "conn",
        "remote work",
        embedder,
        cfg,
        filters={"source_platform": "twitter", "content_type": "thread", "ignored": "x"},
    )

    assert embedder.seen == ["Represent this sentence for searching relevant passages: remote work"]
    assert captured["query_embedding"] == [0.1, 0.2, 0.3]
    assert captured["k"] == 20
    assert captured["filters"] == {"source_platform": "twitter", "content_type": "thread"}
    assert captured["document_ids"] == [101, 202]

    assert [source.external_id for source in sources] == ["1001", "2002"]
    assert [source.index for source in sources] == [1, 2]
    assert sources[0].snippet == "best chunk"
    assert sources[0].context_text == "short full thread text"
    assert sources[0].score == 0.9
    assert sources[0].cited is False
    assert sources[1].snippet == "fallback chunk"
    assert sources[1].context_text == "fallback chunk"


def test_retrieve_hybrid_fuses_semantic_and_lexical_candidates(monkeypatch):
    cfg = SimpleNamespace(
        bge_query_instruction="Represent this sentence for searching relevant passages:",
        retrieval_candidates=20,
        retrieval_k=3,
        retrieval_hybrid=True,
        rrf_candidates=50,
        rrf_k=60,
        chat_context_char_cap=200,
    )
    embedder = FakeEmbedder([[0.1, 0.2, 0.3]])
    captured = {}

    def fake_search(conn, query_embedding, k, **filters):
        captured["semantic"] = {
            "conn": conn,
            "query_embedding": query_embedding,
            "k": k,
            "filters": filters,
        }
        return [
            {"id": 1, "document_id": 101, "text": "semantic top", "score": 0.9},
            {"id": 2, "document_id": 202, "text": "semantic second", "score": 0.8},
        ]

    def fake_search_fts(conn, query_text, k, **filters):
        captured["lexical"] = {
            "conn": conn,
            "query_text": query_text,
            "k": k,
            "filters": filters,
        }
        return [
            {"id": 3, "document_id": 303, "text": "lexical top", "rank": 0.5},
            {"id": 1, "document_id": 101, "text": "semantic top", "rank": 0.4},
        ]

    def fake_fetch_documents(conn, document_ids):
        captured["document_ids"] = document_ids
        return {
            101: {
                "source_platform": "twitter",
                "external_id": "1001",
                "content_type": "post",
                "author_handle": "cortex_demo",
                "url": "https://twitter.com/cortex_demo/status/1001",
                "text": "semantic doc",
                "created_at": datetime(2024, 5, 15, tzinfo=timezone.utc),
                "metadata": {},
            },
            303: {
                "source_platform": "twitter",
                "external_id": "2022",
                "content_type": "post",
                "author_handle": "cortex_demo",
                "url": "https://twitter.com/cortex_demo/status/2022",
                "text": "lexical doc",
                "created_at": None,
                "metadata": {},
            },
            202: {
                "source_platform": "twitter",
                "external_id": "2002",
                "content_type": "post",
                "author_handle": "cortex_demo",
                "url": "https://twitter.com/cortex_demo/status/2002",
                "text": "semantic second doc",
                "created_at": None,
                "metadata": {},
            },
        }

    monkeypatch.setattr(retriever.repository, "search", fake_search)
    monkeypatch.setattr(retriever.repository, "search_fts", fake_search_fts)
    monkeypatch.setattr(retriever.repository, "fetch_documents", fake_fetch_documents)

    sources = retrieve(
        "conn",
        "Conway's Law",
        embedder,
        cfg,
        filters={"source_platform": "twitter"},
    )

    assert captured["semantic"]["k"] == 50
    assert captured["semantic"]["filters"] == {"source_platform": "twitter"}
    assert captured["lexical"] == {
        "conn": "conn",
        "query_text": "Conway's Law",
        "k": 50,
        "filters": {"source_platform": "twitter"},
    }
    assert captured["document_ids"] == [101, 303, 202]
    assert [source.external_id for source in sources] == ["1001", "2022", "2002"]
    assert sources[0].score > sources[1].score > sources[2].score


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.seen = []

    def embed(self, texts):
        self.seen.extend(texts)
        return self.vectors
