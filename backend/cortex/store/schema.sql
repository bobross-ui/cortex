CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id              BIGSERIAL PRIMARY KEY,
  source_platform TEXT        NOT NULL,
  external_id     TEXT        NOT NULL,
  content_type    TEXT        NOT NULL,
  author_handle   TEXT,
  text            TEXT        NOT NULL,
  url             TEXT,
  created_at      TIMESTAMPTZ,
  ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  content_hash    TEXT        NOT NULL,
  metadata        JSONB       NOT NULL DEFAULT '{}',
  UNIQUE (source_platform, external_id)
);

CREATE TABLE IF NOT EXISTS chunks (
  id              BIGSERIAL PRIMARY KEY,
  document_id     BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index     INT         NOT NULL,
  text            TEXT        NOT NULL,
  embed_text      TEXT        NOT NULL,
  token_count     INT,
  content_hash    TEXT        NOT NULL,
  embed_model     TEXT        NOT NULL,
  embedding       VECTOR(384),
  source_platform TEXT        NOT NULL,
  content_type    TEXT        NOT NULL,
  created_at      TIMESTAMPTZ,
  fts             TSVECTOR    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  metadata        JSONB       NOT NULL DEFAULT '{}',
  UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_fts_gin      ON chunks USING gin (fts);
CREATE INDEX IF NOT EXISTS chunks_hash_idx     ON chunks (content_hash);
CREATE INDEX IF NOT EXISTS chunks_filter_idx   ON chunks (source_platform, content_type);
CREATE INDEX IF NOT EXISTS chunks_created_idx  ON chunks (created_at);
