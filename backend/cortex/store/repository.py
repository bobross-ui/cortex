from collections.abc import Sequence

from pgvector import Vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def get_existing_hashes(conn, source_platform: str, external_ids: Sequence[str]) -> dict[str, str]:
    """Map external_id to documents.content_hash for already stored documents."""
    if not external_ids:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT external_id, content_hash
            FROM documents
            WHERE source_platform = %s AND external_id = ANY(%s)
            """,
            (source_platform, list(external_ids)),
        )
        return {external_id: content_hash for external_id, content_hash in cur.fetchall()}


def upsert_document(conn, item, content_hash: str) -> int:
    """Insert or update one document row and return its deterministic database id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
              source_platform,
              external_id,
              content_type,
              author_handle,
              text,
              url,
              created_at,
              content_hash,
              metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_platform, external_id) DO UPDATE
            SET content_type = EXCLUDED.content_type,
                author_handle = EXCLUDED.author_handle,
                text = EXCLUDED.text,
                url = EXCLUDED.url,
                created_at = EXCLUDED.created_at,
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata,
                ingested_at = now()
            RETURNING id
            """,
            (
                item.source_platform,
                item.external_id,
                item.content_type,
                item.author_handle,
                item.text,
                item.url,
                item.created_at,
                content_hash,
                Jsonb(item.metadata),
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("document upsert returned no id")
        return row[0]


def replace_chunks(conn, document_id: int, chunk_rows: Sequence[dict]) -> int:
    """Replace all chunks for one document and return the inserted row count."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        if not chunk_rows:
            return 0

        values = [
            (
                document_id,
                row["chunk_index"],
                row["text"],
                row["embed_text"],
                row.get("token_count"),
                row["content_hash"],
                row["embed_model"],
                _adapt_embedding(row["embedding"]),
                row["source_platform"],
                row["content_type"],
                row.get("created_at"),
                Jsonb(row.get("metadata") or {}),
            )
            for row in chunk_rows
        ]
        cur.executemany(
            """
            INSERT INTO chunks (
              document_id,
              chunk_index,
              text,
              embed_text,
              token_count,
              content_hash,
              embed_model,
              embedding,
              source_platform,
              content_type,
              created_at,
              metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            values,
        )
        return len(values)


def search(
    conn,
    query_embedding,
    k: int = 8,
    *,
    source_platform: str | None = None,
    content_type: str | None = None,
) -> list[dict]:
    """Small cosine-KNN helper; full retrieval/ranking belongs to Layer 3."""
    query_vector = _adapt_embedding(query_embedding)
    clauses = ["embedding IS NOT NULL"]
    params = [query_vector]

    if source_platform is not None:
        clauses.append("source_platform = %s")
        params.append(source_platform)
    if content_type is not None:
        clauses.append("content_type = %s")
        params.append(content_type)

    params.extend([query_vector, k])
    where = " AND ".join(clauses)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT
              id,
              document_id,
              chunk_index,
              text,
              source_platform,
              content_type,
              created_at,
              metadata,
              1 - (embedding <=> %s) AS score
            FROM chunks
            WHERE {where}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            params,
        )
        return list(cur.fetchall())


def _adapt_embedding(embedding):
    if embedding is None or isinstance(embedding, Vector):
        return embedding
    return Vector(embedding)
