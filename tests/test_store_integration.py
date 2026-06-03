import json
import math
import os
from pathlib import Path
import shutil

import pytest

from cortex.chunking.chunker import chunk_text
from cortex.chunking.enrich import author_blurb, build_embed_text
from cortex.config import settings
from cortex.embedding.embedder import Embedder, SentenceTransformerEmbedder
from cortex.ingestion.twitter import TwitterParser
from cortex.pipeline.index import index_export, sha256_text
from cortex.store.db import connect
from cortex.store.repository import get_reusable_vectors, search


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("DATABASE_URL") and not os.getenv("CI"),
        reason="DATABASE_URL is required for pgvector integration tests",
    ),
]

FIX = Path(__file__).parent / "fixtures" / "twitter"


class CountingEmbedder(Embedder):
    def __init__(self, inner: Embedder):
        self.inner = inner
        self.calls = 0
        self.seen = []
        self.model_id = inner.model_id
        self.dim = inner.dim

    def embed(self, texts):
        self.calls += 1
        self.seen.extend(texts)
        return self.inner.embed(texts)


@pytest.fixture(scope="module")
def database_url():
    return os.environ["DATABASE_URL"]


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


def test_connect_applies_schema_idempotently(database_url):
    first = connect(database_url)
    first.close()
    second = connect(database_url)
    second.close()


def test_real_model_outputs_384_dim_unit_vectors(real_embedder):
    vector = real_embedder.embed(["remote work and async communication"])[0]
    norm = math.sqrt(sum(value * value for value in vector))

    assert real_embedder.dim == 384
    assert len(vector) == 384
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_index_pipeline_idempotency_update_dedup_and_semantic_search(
    tmp_path,
    conn,
    real_embedder,
):
    _reset_db(conn)

    counting = CountingEmbedder(real_embedder)
    first = index_export(FIX, conn, embedder=counting, cfg=settings)

    assert first.documents_new == 58
    assert first.documents_updated == 0
    assert first.documents_unchanged_skipped == 0
    assert first.chunks_inserted > first.documents_new
    assert first.chunks_embedded == len(set(counting.seen))
    assert first.chunks_reused_cross_run == 0
    assert len(counting.seen) == len(set(counting.seen))
    assert first.chunks_dedup_within_run == first.chunks_inserted - first.chunks_embedded

    assert _table_count(conn, "documents") == 58
    assert _table_count(conn, "chunks") == first.chunks_inserted
    assert _split_document_count(conn) >= 1

    query = f"{settings.bge_query_instruction} remote work"
    query_embedding = real_embedder.embed([query])[0]
    hits = search(conn, query_embedding, k=3)
    assert "1001" in _external_ids_for_hits(conn, hits)

    calls_after_first = counting.calls
    second = index_export(FIX, conn, embedder=counting, cfg=settings)

    assert second.documents_new == 0
    assert second.documents_updated == 0
    assert second.documents_unchanged_skipped == 58
    assert second.chunks_inserted == 0
    assert second.chunks_embedded == 0
    assert second.chunks_reused_cross_run == 0
    assert second.embed_batches == 0
    assert counting.calls == calls_after_first

    mutated_root = _mutated_fixture(tmp_path)
    updated = index_export(mutated_root, conn, embedder=counting, cfg=settings)
    doc_id = _document_id(conn, "1004")

    assert updated.documents_new == 0
    assert updated.documents_updated == 1
    assert updated.documents_unchanged_skipped == 57
    assert updated.chunks_inserted == 1
    assert updated.chunks_embedded == 1
    assert updated.chunks_reused_cross_run == 0
    assert _chunk_count_for_document(conn, doc_id) == 1
    assert _table_count(conn, "chunks") == first.chunks_inserted

    _reset_db(conn)
    dedup_counting = CountingEmbedder(real_embedder)
    dedup = index_export(
        FIX,
        conn,
        enricher=lambda item, chunk_text, blurb: "constant embedding input",
        embedder=dedup_counting,
        cfg=settings,
    )

    assert dedup.chunks_inserted == first.chunks_inserted
    assert dedup.chunks_embedded == 1
    assert dedup.chunks_reused_cross_run == 0
    assert dedup.chunks_dedup_within_run == first.chunks_inserted - 1
    assert dedup_counting.seen == ["constant embedding input"]


def test_changed_multichunk_document_reuses_unchanged_vectors(
    tmp_path,
    conn,
    real_embedder,
):
    _reset_db(conn)
    first = index_export(FIX, conn, embedder=real_embedder, cfg=settings)
    before = _chunk_vectors_by_hash(conn, "2300")
    with conn.transaction():
        wrong_model_reuse = get_reusable_vectors(conn, list(before), "different-model")

    mutated_root = _mutated_thread_fixture(tmp_path)
    run_two_hashes = _computed_chunk_hashes(mutated_root, "2300")
    reusable_hashes = run_two_hashes & before.keys()
    new_hashes = run_two_hashes - before.keys()

    assert reusable_hashes
    assert new_hashes

    counting = CountingEmbedder(real_embedder)
    updated = index_export(mutated_root, conn, embedder=counting, cfg=settings)
    after = _chunk_vectors_by_hash(conn, "2300")

    assert first.chunks_reused_cross_run == 0
    assert wrong_model_reuse == {}
    assert updated.documents_updated == 1
    assert updated.documents_unchanged_skipped == 57
    assert updated.chunks_embedded == len(new_hashes)
    assert updated.chunks_reused_cross_run == len(reusable_hashes)
    assert updated.chunks_embedded + updated.chunks_reused_cross_run == len(run_two_hashes)
    assert set(counting.seen) == _embed_texts_for_hashes(mutated_root, "2300", new_hashes)
    for content_hash in reusable_hashes:
        assert list(after[content_hash]) == list(before[content_hash])


def test_enrichment_context_change_invalidates_dependent_documents(
    tmp_path,
    conn,
    real_embedder,
):
    _reset_db(conn)
    first = index_export(FIX, conn, embedder=real_embedder, cfg=settings)

    bio_root = _mutated_bio_fixture(tmp_path)
    bio_updated = index_export(bio_root, conn, embedder=real_embedder, cfg=settings)

    assert bio_updated.documents_updated == first.documents_new
    assert bio_updated.documents_unchanged_skipped == 0
    assert bio_updated.chunks_inserted == first.chunks_inserted
    assert bio_updated.chunks_embedded == first.chunks_embedded
    assert bio_updated.chunks_reused_cross_run == 0

    _reset_db(conn)
    index_export(FIX, conn, embedder=real_embedder, cfg=settings)

    post_root = _mutated_fixture(tmp_path)
    post_updated = index_export(post_root, conn, embedder=real_embedder, cfg=settings)

    assert post_updated.documents_updated == 1
    assert post_updated.documents_unchanged_skipped == 57
    assert post_updated.chunks_inserted == 1
    assert post_updated.chunks_embedded == 1
    assert post_updated.chunks_reused_cross_run == 0


def _reset_db(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
    conn.commit()


def _table_count(conn, table: str) -> int:
    if table not in {"documents", "chunks"}:
        raise ValueError(f"unexpected table {table}")
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        return cur.fetchone()[0]


def _split_document_count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*)
            FROM (
              SELECT document_id
              FROM chunks
              GROUP BY document_id
              HAVING count(*) > 1
            ) split_docs
            """
        )
        return cur.fetchone()[0]


def _external_ids_for_hits(conn, hits) -> set[str]:
    document_ids = [hit["document_id"] for hit in hits]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT external_id FROM documents WHERE id = ANY(%s)",
            (document_ids,),
        )
        return {external_id for (external_id,) in cur.fetchall()}


def _document_id(conn, external_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE source_platform = 'twitter' AND external_id = %s",
            (external_id,),
        )
        return cur.fetchone()[0]


def _chunk_count_for_document(conn, document_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        return cur.fetchone()[0]


def _chunk_vectors_by_hash(conn, external_id: str) -> dict[str, object]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunks.content_hash, chunks.embedding
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.source_platform = 'twitter'
              AND documents.external_id = %s
            """,
            (external_id,),
        )
        return {content_hash: embedding for content_hash, embedding in cur.fetchall()}


def _embed_texts_by_hash(root: Path, external_id: str) -> dict[str, str]:
    items = list(TwitterParser().parse(root))
    blurb = author_blurb(items)
    item = next(item for item in items if item.external_id == external_id)
    by_hash = {}
    for chunk in chunk_text(item.text, settings):
        embed_text = build_embed_text(item, chunk.text, blurb)
        by_hash[sha256_text(embed_text)] = embed_text
    return by_hash


def _computed_chunk_hashes(root: Path, external_id: str) -> set[str]:
    return set(_embed_texts_by_hash(root, external_id))


def _embed_texts_for_hashes(root: Path, external_id: str, hashes: set[str]) -> set[str]:
    by_hash = _embed_texts_by_hash(root, external_id)
    return {by_hash[content_hash] for content_hash in hashes}


def _mutated_fixture(tmp_path) -> Path:
    root = tmp_path / "twitter"
    shutil.copytree(FIX, root)
    tweets_path = root / "data" / "tweets.js"
    prefix = "window.YTD.tweets.part0 = "
    rows = json.loads(tweets_path.read_text(encoding="utf-8").removeprefix(prefix))
    for row in rows:
        tweet = row.get("tweet", {})
        if tweet.get("id_str") == "1004":
            tweet["full_text"] = "Updated standalone post about focused async work."
            break
    else:
        raise AssertionError("fixture tweet 1004 not found")

    tweets_path.write_text(prefix + json.dumps(rows, indent=2), encoding="utf-8")
    return root


def _mutated_thread_fixture(tmp_path) -> Path:
    root = tmp_path / "twitter-thread"
    shutil.copytree(FIX, root)
    tweets_path = root / "data" / "tweets.js"
    prefix = "window.YTD.tweets.part0 = "
    rows = json.loads(tweets_path.read_text(encoding="utf-8").removeprefix(prefix))
    for row in rows:
        tweet = row.get("tweet", {})
        if tweet.get("id_str") == "2301":
            tweet["full_text"] = tweet["full_text"].replace("trillions", "billions")
            break
    else:
        raise AssertionError("fixture thread member 2301 not found")

    tweets_path.write_text(prefix + json.dumps(rows, indent=2), encoding="utf-8")
    return root


def _mutated_bio_fixture(tmp_path) -> Path:
    root = tmp_path / "twitter-bio"
    shutil.copytree(FIX, root)
    profile_path = root / "data" / "profile.js"
    prefix = "window.YTD.profile.part0 = "
    rows = json.loads(profile_path.read_text(encoding="utf-8").removeprefix(prefix))
    rows[0]["profile"]["description"]["bio"] = (
        "Engineer. Writing about durable systems and focused work. Opinions my own."
    )
    profile_path.write_text(prefix + json.dumps(rows, indent=2), encoding="utf-8")
    return root
