import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

import psycopg

from cortex.chunking.chunker import chunk_text, estimate_tokens
from cortex.chunking.enrich import author_blurb, build_embed_text
from cortex.config import settings
from cortex.embedding.embedder import SentenceTransformerEmbedder
from cortex.ingestion import resolve
from cortex.models import IndexReport
from cortex.store.db import connect
from cortex.store.repository import get_existing_hashes, replace_chunks, upsert_document


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_embed_texts(chunk_rows: list[dict]) -> list[str]:
    seen = set()
    unique = []
    for row in chunk_rows:
        embed_text = row["embed_text"]
        if embed_text in seen:
            continue
        seen.add(embed_text)
        unique.append(embed_text)
    return unique


def embed_unique_texts(
    embedder,
    texts: list[str],
    batch_size: int,
) -> tuple[dict[str, list[float]], int, float]:
    vectors: dict[str, list[float]] = {}
    if not texts:
        return vectors, 0, 0.0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    t0 = time.perf_counter()
    batches = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_vectors = embedder.embed(batch)
        if len(batch_vectors) != len(batch):
            raise ValueError(
                f"embedder returned {len(batch_vectors)} vectors for {len(batch)} texts"
            )
        vectors.update(zip(batch, batch_vectors))
        batches += 1
    return vectors, batches, time.perf_counter() - t0


def attach_embeddings(chunk_rows: list[dict], vector_by_text: dict[str, list[float]]) -> None:
    for row in chunk_rows:
        row["embedding"] = vector_by_text[row["embed_text"]]


def index_export(
    root,
    conn,
    chunker=chunk_text,
    enricher=build_embed_text,
    embedder=None,
    cfg=settings,
) -> IndexReport:
    if embedder is None:
        raise ValueError("index_export requires an embedder")
    _assert_embedder_config(embedder, cfg)

    root = Path(root)
    t0 = time.perf_counter()

    parser = resolve(root)
    items = list(parser.parse(root))
    report = IndexReport(platform=parser.platform)
    blurb = author_blurb(items)

    external_ids = [item.external_id for item in items]
    with conn.transaction():
        existing = get_existing_hashes(conn, parser.platform, external_ids)

    pending = []
    all_chunk_rows = []
    for item in items:
        doc_hash = sha256_text(item.text)
        old_hash = existing.get(item.external_id)
        if old_hash == doc_hash:
            report.documents_unchanged_skipped += 1
            continue

        if old_hash is None:
            report.documents_new += 1
        else:
            report.documents_updated += 1

        chunk_rows = []
        for chunk in chunker(item.text, cfg):
            embed_text = enricher(item, chunk.text, blurb)
            chunk_row = {
                "chunk_index": chunk.index,
                "text": chunk.text,
                "embed_text": embed_text,
                "token_count": estimate_tokens(embed_text),
                "content_hash": sha256_text(embed_text),
                "embed_model": embedder.model_id,
                "embedding": None,
                "source_platform": item.source_platform,
                "content_type": item.content_type,
                "created_at": item.created_at,
                "metadata": dict(item.metadata),
            }
            chunk_rows.append(chunk_row)
            all_chunk_rows.append(chunk_row)
        pending.append((item, doc_hash, chunk_rows))

    unique_texts = unique_embed_texts(all_chunk_rows)
    vector_by_text, embed_batches, embed_duration_s = embed_unique_texts(
        embedder,
        unique_texts,
        cfg.embed_batch_size,
    )
    report.chunks_embedded = len(unique_texts)
    report.chunks_dedup_within_run = len(all_chunk_rows) - len(unique_texts)
    report.embed_batches = embed_batches
    report.embed_duration_s = embed_duration_s

    for item, doc_hash, chunk_rows in pending:
        attach_embeddings(chunk_rows, vector_by_text)

        with conn.transaction():
            document_id = upsert_document(conn, item, doc_hash)
            report.chunks_inserted += replace_chunks(conn, document_id, chunk_rows)

    report.duration_s = time.perf_counter() - t0
    return report


def chunk_count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        return int(cur.fetchone()[0])


def seed_if_empty(conn, embedder, cfg=settings) -> IndexReport | None:
    """Index the bundled export once if the chunks table is empty.

    Keeps `docker compose up` a single command: the first boot seeds the
    knowledge base, later boots find persisted rows and skip embedding entirely.
    Returns the IndexReport when seeding ran, or None when it was skipped.
    """
    if chunk_count(conn) > 0:
        return None
    export_dir = Path(cfg.seed_export_dir)
    if not export_dir.exists():
        return None
    return index_export(export_dir, conn, embedder=embedder, cfg=cfg)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m cortex.pipeline.index <export_dir>", file=sys.stderr)
        return 2

    cfg = settings
    try:
        conn = connect(cfg.database_url)
    except psycopg.Error as exc:
        print(f"could not connect to database: {exc}", file=sys.stderr)
        return 1

    try:
        embedder = SentenceTransformerEmbedder(
            cfg.embed_model,
            cfg.embed_dim,
            cfg.embed_batch_size,
        )
        report = index_export(Path(argv[0]), conn, embedder=embedder, cfg=cfg)
    finally:
        conn.close()

    print(report.human_summary())
    print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    return 0


def _assert_embedder_config(embedder, cfg) -> None:
    if embedder.dim != cfg.embed_dim or cfg.embed_dim != 384:
        raise ValueError(
            f"embedding dimension mismatch: embedder={embedder.dim}, "
            f"settings={cfg.embed_dim}, schema=384"
        )


if __name__ == "__main__":
    raise SystemExit(main())
