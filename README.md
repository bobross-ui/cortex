# Cortex

> **Social Data → Vector Knowledge Base → Grounded Chat.** A personal RAG system that turns your
> social-media exports into a searchable, citable knowledge base you can ask questions of.

![status](https://img.shields.io/badge/status-Layer%202%20shipped-brightgreen)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![tests](https://img.shields.io/badge/tests-pytest-green)

Cortex ingests your data exports from LinkedIn, Twitter/X, and Instagram, extracts the content
that actually *represents you* (and discards the noise), builds a vector knowledge base, and lets
you ask questions like *"What do I think about remote work?"* — answered with cited sources.
**Efficiency is a first-class requirement, not a bonus.**

---

## The brief

An end-to-end system built across four layers, each of which matters:

1. **Multi-source ingestion** — parse LinkedIn (CSV), Twitter/X (JSON), Instagram (JSON/HTML).
   Keep authored content, discard noise. Adding a 4th source should take **under an hour**.
2. **Vector knowledge base** — chunk intelligently (not by character count), embed, and store with
   a self-designed schema that's extensible to new content types without a rewrite.
3. **Chat (RAG)** — a minimal UI that retrieves relevant chunks and returns a grounded answer with
   **cited sources**.
4. **Efficiency** — survive 50MB+ exports: batching, deduplication, incremental upserts, with
   **named tradeoffs**.

---

## Roadmap

This README grows one layer at a time. **Layers 1–2** are detailed below; later sections are added
as each layer lands.

| Layer | Scope | Status |
|---|---|---|
| **1 — Ingestion** | Pluggable parsers; authored-content extraction; Twitter first | ✅ **Shipped** |
| **2 — Vector KB** | Structure-aware chunking, local embeddings, pgvector schema | ✅ **Shipped** |
| 3 — Chat (RAG) | Retrieval, grounded prompt, cited answers, UI | ⚪ Planned |
| 4 — Efficiency | Streaming parse, multiprocessing/stress numbers, RSS measurement | ⚪ Planned |

---

## Layer 1 — Multi-source ingestion

The job of Layer 1 is to turn a raw platform export (a folder of files) into a clean stream of
canonical `ContentItem`s — keeping what represents the person, dropping the noise, and **reporting
what it dropped** so the editorial decision is provable, not just claimed.

### Key decisions

| Decision | What & why |
|---|---|
| **Pluggable parser seam** | One canonical model (`ContentItem`) + a `SourceParser` ABC + a registry. Adding a source = one new file + one `register()` line, nothing downstream changes. This is what makes *"4th source < 1 hour"* real. |
| **Directory-root dispatch** | Real exports are **folders** (often split across part-files), not single files. Parsers operate on an extracted export directory and read account-level identity files first. |
| **KEEP vs DROP is explicit** | An editorial policy (authored posts/threads/quotes/bio **keep**; retweets, replies-to-others, ads, likes, DMs **drop**) — recorded, and backed by an **ingestion report** (kept / dropped / skipped counts + reasons) as evidence. |
| **Stitch threads at parse time** | A self-reply thread becomes **one** `ContentItem` (root id as `external_id`), so a thought stays whole and yields one coherent citation. |
| **Parser-owned normalization** | The parser emits clean canonical text (unescape entities, expand/strip links, collapse whitespace, drop empty/media-only items) so downstream hashing/embedding is stable. |
| **Stable `external_id`** | Native platform id (root id for threads; `profile` for bio) → re-ingesting the same export updates in place instead of duplicating. |
| **`metadata` convention** | Per-`content_type` JSONB extras follow a documented key convention → extensible to new content types **without a migration**. |
| **Twitter first, sources verified** | The Twitter archive format was verified against the canonical community parser. Two discriminators (retweet & quote-tweet shape) are **flagged unverified**, pending validation against a real export. |
| **Defensive parsing** | Tolerate missing fields; skip malformed/empty items and record them; never crash a run on one bad item. |

### The canonical model

Every parser emits the same shape — downstream code knows only this:

```python
@dataclass
class ContentItem:
    source_platform: str          # 'twitter' | 'linkedin' | 'instagram' | ...
    external_id: str              # stable id (root id for threads; 'profile' for bio)
    content_type: str             # 'post' | 'thread' | 'bio' | ...
    text: str                     # normalized authored text; never empty
    author_handle: str | None
    created_at: datetime | None   # tz-aware UTC
    url: str | None
    metadata: dict                # per-content_type extras → JSONB, extensible without migration
```

### Twitter — KEEP vs DROP

| KEEP (represents the person) | DROP (noise) |
|---|---|
| original tweets, stitched self-reply threads, quote-tweets with commentary, profile bio | retweets, replies to others, likes, DMs, blocks, ad/impression data |

### Tech (Layer 1)

Python **3.11+**, **standard library only** (`json`, `html`, `re`, `datetime`, `pathlib`,
`dataclasses`), with **pytest** for tests. No database, embeddings, or network calls at this layer.

---

## Layer 2 — Vector knowledge base

Layer 2 takes Layer 1's canonical `ContentItem`s and builds the vector knowledge base:
structure-aware chunking → deterministic embedding-input enrichment → local BGE embeddings →
Postgres + pgvector storage with incremental, deduplicated upserts.

Retrieval, RAG prompts, chat UI, reranking, and hybrid search logic are **not** in this layer.
Layer 2 only provisions the storage/search seam that makes those cheap to add later.

### Key decisions

| Decision | What & why |
|---|---|
| **Postgres + pgvector** | One durable relational store for source documents, retrievable chunks, JSONB metadata, FTS, and vectors. `pgvector` gives a production-shaped path without adding a separate vector database. |
| **Local embeddings** | `BAAI/bge-small-en-v1.5` is the default model. It is 384-dimensional, deterministic enough for repeatable indexing, offline after the first download, and avoids hosted embedding cost/rate limits. |
| **Raw text ≠ embed text** | `documents.text` and `chunks.text` stay raw for source display, citations, and FTS. `chunks.embed_text` adds deterministic context only for embedding. Hashes are separate for documents vs chunks. |
| **Structure-aware chunking** | Social records stay atomic by default. Long stitched threads split on paragraphs/sentences with token budgets and overlap; no character-window shredding. |
| **Templated enrichment seam** | A single `build_embed_text()` function adds `[platform · type · date]` plus the profile bio as standing context. Future LLM enrichment can swap in here without a schema rewrite. |
| **Incremental document skip** | `sha256(documents.text)` detects unchanged documents. Re-indexing an unchanged export skips chunking and embedding entirely. |
| **Within-run embed dedup** | Identical `embed_text` strings are embedded once per run, then reused for all matching chunk rows. Cross-run vector reuse is deliberately deferred. |
| **Extensible content types** | `content_type` is `TEXT`, not a Postgres enum. New types are new strings + JSONB metadata keys; no `ALTER TYPE` migration. |
| **Hybrid-ready, not hybrid yet** | Chunks have generated `fts` and a GIN index for Layer 3 hybrid retrieval, but Layer 2 only ships a small cosine search helper. |

### Text and hash mapping

| Column | Contains | Purpose |
|---|---|---|
| `documents.text` | full original authored text | citation/source display |
| `documents.content_hash` | `sha256(documents.text)` | document-level change detection |
| `chunks.text` | raw chunk slice | matched passage display + FTS source |
| `chunks.embed_text` | deterministic context prefix + raw chunk | passage embedding input |
| `chunks.content_hash` | `sha256(chunks.embed_text)` | within-run embed dedup key |
| `chunks.fts` | generated `tsvector` from raw chunk text | Layer 3 lexical/hybrid search |
| `chunks.embedding` | `vector(384)` from BGE | semantic search |

### Schema

Layer 2 creates two tables:

| Table | Meaning |
|---|---|
| `documents` | one row per `ContentItem`; unique on `(source_platform, external_id)` for idempotent upsert |
| `chunks` | one row per retrievable unit; foreign key to `documents`, `vector(384)`, generated FTS, JSONB metadata, filter columns |

Indexes:

- `chunks_embedding_hnsw` — HNSW cosine index over `embedding vector_cosine_ops`
- `chunks_fts_gin` — GIN index over generated `fts`
- `chunks_hash_idx` — chunk embed-text hash lookup
- `chunks_filter_idx` — source/content-type filters
- `chunks_created_idx` — time filters
- `documents_source_platform_external_id_key` — deterministic document identity

### Index report

The Layer 2 CLI returns an `IndexReport`:

```python
@dataclass
class IndexReport:
    platform: str
    documents_new: int
    documents_updated: int
    documents_unchanged_skipped: int
    chunks_inserted: int
    chunks_embedded: int
    chunks_dedup_within_run: int
    embed_batches: int
    duration_s: float
    embed_duration_s: float
```

On the bundled Twitter fixture, a clean index run currently inserts **58 documents** and **60
chunks**. The two extra chunks come from long stitched threads `2300` and `2400`.

Re-running the same export reports **58 unchanged skipped** and **0 chunks embedded**.

### Tech (Layer 2)

Python **3.11+**, `sentence-transformers`, `BAAI/bge-small-en-v1.5`, `psycopg 3`,
`pgvector`, `Postgres 16 + pgvector`, `numpy`, `pydantic-settings`, and `pytest`.

Named tradeoffs:

- Local embeddings pull a large `torch` stack, but make embedding $0, offline-after-cache, and
  rate-limit-free.
- Token counting uses a deterministic `words × 1.3` estimate instead of the model tokenizer, so
  pure tests do not download the model and chunk boundaries stay stable.
- Layer 2 holds pending chunks in memory. At 10k chunks, 384-dim vectors are small enough for this
  layer; streaming parse and stress/RSS measurements are Layer 4.

---

## Getting started

> **Prerequisite:** Python **3.11+**. (A fresh virtualenv is recommended: `python3.11 -m venv .venv && source .venv/bin/activate`.)

```bash
# install (editable, with dev/test extras)
pip install -e ".[dev]"

# ingest the bundled Twitter fixture and print the ingestion report
python -m cortex.pipeline.ingest tests/fixtures/twitter

# start Postgres + pgvector
docker compose up -d postgres

# index the bundled Twitter fixture into the vector knowledge base
DATABASE_URL=postgresql://cortex:cortex@localhost:5432/cortex \
  python -m cortex.pipeline.index tests/fixtures/twitter

# re-run to verify incremental skip
DATABASE_URL=postgresql://cortex:cortex@localhost:5432/cortex \
  python -m cortex.pipeline.index tests/fixtures/twitter

# run the pure test tier (no DB, no model download)
pytest -m "not integration"

# run the integration tier (real pgvector + real BGE model)
DATABASE_URL=postgresql://cortex:cortex@localhost:5432/cortex pytest -m integration

# stop Postgres when done
docker compose down
```

The first index or integration run may download and cache the BGE model from Hugging Face. Later
runs reuse the local cache.

Ingesting the fixture prints a one-line summary plus a JSON report, e.g.:

```
twitter: 58 kept (bio 1, post 51, thread 6), 24 dropped (reply_to_other 13, retweet 11), 7 skipped (malformed 2, empty 5) in 0.002s
```

Indexing the fixture prints a one-line summary plus a JSON report, e.g.:

```
twitter: 58 changed (new 58, updated 0), 0 unchanged skipped, 60 chunks inserted, 60 chunks embedded (0 deduped), 1 embed batches in 1.518s
```

Re-indexing the unchanged fixture prints:

```
twitter: 0 changed (new 0, updated 0), 58 unchanged skipped, 0 chunks inserted, 0 chunks embedded (0 deduped), 0 embed batches in 0.006s
```

---

## Project structure

```
cortex/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml
├── backend/cortex/
│   ├── config.py              # settings from env / .env
│   ├── models.py              # ContentItem, IngestionReport, IndexReport
│   ├── chunking/              # structure-aware chunker + embed_text enrichment
│   ├── embedding/             # Embedder ABC + SentenceTransformerEmbedder
│   ├── ingestion/             # base ABC, registry, normalize, twitter parser
│   ├── store/                 # schema.sql, db connection, repository helpers
│   └── pipeline/              # ingest.py (Layer 1), index.py (Layer 2)
└── tests/
    ├── fixtures/twitter/      # synthetic test archive
    ├── test_chunker.py
    ├── test_enrich.py
    ├── test_hashing_dedup.py
    ├── test_store_integration.py
    └── test_*.py
```

---

## Status

**Layer 1 (ingestion) is shipped** — the pluggable parser seam, the Twitter parser, and the
ingestion report are implemented and covered by tests against a high-fidelity synthetic Twitter
fixture.

**Layer 2 (vector knowledge base) is shipped** — structure-aware chunking, deterministic
enrichment, local BGE embeddings, Postgres + pgvector schema, incremental upsert, within-run
embedding dedup, semantic search helper, and two-tier tests are implemented.

LinkedIn and Instagram parsers follow on the Layer 1 seam. Chat/RAG retrieval and UI follow on
the Layer 2 store/search seam. This README will be extended as Layers 3–4 land.

> Known limitation: two Twitter classification rules (retweet & quote-tweet shape) are derived
> from secondary sources and flagged for validation against a real export.

> Known Layer 2 limitation: templated enrichment is the shipped zero-cost baseline. LLM enrichment
> for short, context-poor posts is intentionally deferred; it is the likely answer-quality knob to
> re-measure on a real corpus.
