# Cortex

> **Social Data → Vector Knowledge Base → Grounded Chat.** A personal RAG system that turns your
> social-media exports into a searchable, citable knowledge base you can ask questions of.

![status](https://img.shields.io/badge/status-Layer%203%20shipped-brightgreen)
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

**Layers 1–3 are shipped.** Layer 4's implementation mechanisms are present, while its controlled
large-scale proof remains deferred.

| Layer | Scope | Status |
|---|---|---|
| **1 — Ingestion** | Pluggable parsers; authored-content extraction; Twitter first | ✅ **Shipped** |
| **2 — Vector KB** | Structure-aware chunking, local embeddings, pgvector schema | ✅ **Shipped** |
| **3 — Chat (RAG)** | Retrieval, grounded prompt, cited answers, UI | ✅ **Shipped** |
| **4 — Efficiency** | Streaming parse, batched writes, cross-run reuse, correctness invalidation | 🟡 **Mechanisms shipped; stress proof deferred** |

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
| **Twitter first, sources verified** | The Twitter archive format was verified against the canonical community parser and a real June 2026 export. Quote-status link handling is validated; retweet-shape handling remains unverified because the real export contained no retweets. |
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

Python **3.11+**, the standard library (`json`, `html`, `re`, `datetime`, `pathlib`, `dataclasses`),
and `ijson` for streaming export arrays, with **pytest** for tests. No database, embeddings, or
network calls at this layer.

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
| **Incremental document skip** | A change hash over raw text plus deterministic embedding context detects unchanged documents. Re-indexing an unchanged export skips chunking and embedding entirely. |
| **Embed dedup and reuse** | Identical `embed_text` strings are embedded once per run; stored vectors are reused across runs by `(content_hash, embed_model)` when part of a changed document remains unchanged. |
| **Extensible content types** | `content_type` is `TEXT`, not a Postgres enum. New types are new strings + JSONB metadata keys; no `ALTER TYPE` migration. |
| **Hybrid-ready, not hybrid yet** | Chunks have generated `fts` and a GIN index for Layer 3 hybrid retrieval, but Layer 2 only ships a small cosine search helper. |

### Text and hash mapping

| Column | Contains | Purpose |
|---|---|---|
| `documents.text` | full original authored text | citation/source display |
| `documents.content_hash` | `sha256(text + platform + type + date + enrichment context)` | document-level change detection without stale embedding context |
| `chunks.text` | raw chunk slice | matched passage display + FTS source |
| `chunks.embed_text` | deterministic context prefix + raw chunk | passage embedding input |
| `chunks.content_hash` | `sha256(chunks.embed_text)` | within-run dedup and same-model cross-run vector reuse key |
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
    chunks_reused_cross_run: int
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
- The pipeline holds pending chunks and Python `list[float]` vectors in memory to preserve batch
  efficiency. Controlled 10k-chunk RSS measurement remains deferred.

---

## Layer 3 — Chat (RAG)

Layer 3 turns the Layer 2 store into a single-turn grounded chat path:
query embedding → retrieval → citation join → grounded prompt → DeepSeek → cited answer → React UI.

### Key decisions

| Decision | What & why |
|---|---|
| **Single-turn chat** | `/chat` answers one question at a time. Conversation history, query rewriting, reranking, and MMR are deferred to keep the vertical slice reliable. |
| **BGE query instruction only on queries** | Passage embeddings stay exactly as Layer 2 wrote them. Query embeddings prepend `Represent this sentence for searching relevant passages:` to match BGE's retrieval guidance. |
| **Citation join is document-level** | Retrieval starts from chunks, then joins back to `documents` for `url`, `external_id`, `author_handle`, and full source text. Multiple chunks from one document collapse to one source. |
| **Bounded document context** | The LLM gets full document text when it is under `chat_context_char_cap` and falls back to the matched chunk for long documents. The UI shows the matched chunk as the snippet. |
| **DeepSeek via OpenAI SDK** | `DeepSeekChatClient` uses the OpenAI-compatible API at `https://api.deepseek.com`, model `deepseek-v4-flash`, with thinking disabled for grounded low-temperature RAG. |
| **Citations are enforced** | If sources exist and the first answer has no valid `[n]` markers, the API retries once with an explicit citation nudge. The response includes `grounded` and per-source `cited` flags. |
| **Streaming UI** | `/chat/stream` streams answer tokens with server-sent events, then sends final cited sources after the complete answer is available for citation parsing. |
| **Hybrid default** | Semantic KNN and Postgres FTS are fused with Reciprocal Rank Fusion. Hybrid is enabled by default because it did not regress on the fixture and should provide better lexical coverage on real exports. |

### API contract

`POST /chat` returns a complete JSON response. `POST /chat/stream` accepts the same request body and
returns `text/event-stream`: `token` events while the model writes, then a final `sources` event with
the same response schema fields, followed by `done`. The non-streaming endpoint retries once if the
model cites nothing; the streaming endpoint cannot retract already-sent text, so it reports the final
`grounded` flag honestly.

```json
{
  "question": "What does this person think about remote work?",
  "filters": { "source_platform": "twitter", "content_type": "thread" }
}
```

Streaming event example:

```text
event: token
data: {"text":"They value async communication"}

event: sources
data: {"answer":"They value async communication [2].","sources":[...],"abstained":false,"grounded":true}

event: done
data: {}
```

Response:

```json
{
  "answer": "They value async communication for deep work [2].",
  "sources": [
    {
      "index": 2,
      "external_id": "1001",
      "platform": "twitter",
      "content_type": "thread",
      "url": "https://twitter.com/cortex_demo/status/1001",
      "author_handle": "cortex_demo",
      "date": "2024-05-15",
      "snippet": "Some hard-won thoughts on remote work, a thread...",
      "score": 0.7309,
      "cited": true
    }
  ],
  "abstained": false,
  "grounded": true
}
```

### Retrieval eval

The bundled eval in `eval/gold_set.py` / `eval/retrieval_eval.py` is a **sanity check**, not a
benchmark. It verifies that semantic-only and hybrid retrieval both find the expected documents. The
fixture is too small and too clean to demonstrate a hybrid quality lift, but hybrid did not regress
the measured fixture, so `retrieval_hybrid=True` is the default for better lexical coverage on real
exports.

| Query class | What happened | Implication |
|---|---|---|
| Paraphrase questions | FTS usually returned no rows because `websearch_to_tsquery` over the full question is strict on unmatched terms. | Hybrid mostly collapsed to semantic-only. |
| Exact-token questions | Semantic already ranked every expected lexical target at `#1`. | Hybrid had no measurable room to improve on this fixture. |
| Hardest semantic query | `how do interruptions affect deep focus?` ranked expected source `2400` at `#4` in both modes. | Hybrid did not fix semantic misses without useful lexical overlap. |

A stronger real-export eval should include lexical-hard queries where semantic does not already rank
the target first, rare handles/product names/hashtags, and query preprocessing for FTS before making
quality claims from the numbers.

Named tradeoff: low `chat_temperature` favors grounded fidelity over creative phrasing.

---

## Layer 4 — Efficiency

Most of the main efficiency levers were already paid for in Layers 1–2: batched local embedding,
within-run embedding deduplication, unchanged-document skips, and idempotent upserts. Layer 4 adds the
remaining implementation mechanisms while deliberately deferring the controlled stress harness and
10k-chunk benchmark.

| Mechanism | Status | Effect |
|---|---|---|
| Batch embedding | ✅ Shipped earlier | Embeds unique passage inputs in batches of 64. |
| Within-run embedding deduplication | ✅ Shipped earlier | Embeds identical `embed_text` once per run. |
| Document-level incremental skip | ✅ Shipped earlier, corrected in Layer 4 | Unchanged documents skip chunking and embedding; the change hash now includes enrichment context. |
| Streaming Twitter row parsing | ✅ Layer 4 | Avoids materializing the raw tweet file and its full decoded row list simultaneously. |
| Single-transaction writes | ✅ Layer 4 | Removes one database commit per changed document while keeping `executemany` chunk inserts. |
| Same-model cross-run vector reuse | ✅ Layer 4 | Reuses unchanged chunk vectors from changed documents by `(content_hash, embed_model)`. |
| Controlled 5/25/50 MB and ~10k-chunk proof | ⏸ Deferred | No large-scale throughput or flat-RSS claim is made yet. |

`IndexReport.chunks_embedded` and `chunks_reused_cross_run` count distinct embedding inputs;
`chunks_inserted` counts stored chunk rows.

### Memory model

Streaming removes the raw tweet file and full parsed-row list from the parser's retained state.
Memory is still `O(unique valid tweet ids) + O(authored content)`, because the parser keeps the
deduplication id set and authored records needed for thread stitching. It is therefore **not**
constant-memory, but noise content is processed and discarded instead of accumulated.

The embedding/write phase still holds pending chunks and vectors in memory to preserve batch
efficiency. `SentenceTransformerEmbedder` returns Python `list[float]` vectors, so their in-memory
cost is materially larger than the compact `float32` wire/disk representation. The projected
10k-chunk RSS cost and flat-memory behavior across controlled large tweet files remain unmeasured
until the deferred stress harness is built.

A real June 2026 Twitter export was audited as compatibility evidence: the complete archive was
48.7 MB, but the parser-relevant account/profile/tweet files totaled only 72,964 bytes. It parsed in
about 0.03 seconds with approximately 22.5 MB peak process RSS and produced 21 kept items. This
validates real-export compatibility, **not** 50 MB tweet-file scalability.

### Concurrency decisions

Concurrency was considered and intentionally not added:

- **Parallel embedding via multiprocessing: deferred.** Torch already uses intra-op CPU parallelism
  during model inference. Extra processes would each load another model copy, trading substantial
  memory and operational complexity for an unmeasured throughput gain.
- **Embedding/write pipeline overlap: deferred.** A producer/consumer pipeline could overlap CPU
  embedding with database I/O, but it complicates failure handling and transaction boundaries. Build
  it only if the deferred benchmark shows database writes are material.
- **Parallel file reads: not added.** The audited real export used one tweet file plus small
  account/profile files; split tweet parts must preserve deterministic ordering, and there is no
  evidence file I/O is the bottleneck.
- **HNSW drop/rebuild during bulk seed: deferred.** It can improve very large initial loads, but
  temporarily disables indexed search for concurrent readers and has not been shown to matter at the
  current scale.

### Named tradeoffs

- **Local embeddings choose cost and operational control over maximum quality.**
  `BAAI/bge-small-en-v1.5` is offline-after-download, rate-limit-free, and costs $0 per chunk, at a
  likely quality cost versus larger hosted embedding models.
- **Batch size 64 balances speed and memory.** Larger batches may improve throughput but increase
  peak RAM; the shipped default is a middle ground.
- **Streaming targets the unbounded raw-file buffer, not every in-memory structure.** Keeping ids
  and authored content enables deterministic deduplication and thread stitching.
- **Holding vectors until the write phase favors batch efficiency over minimum RAM.** Switching to
  `float32` arrays plus batched writes is the next lever if measured RSS becomes a problem.
- **One transaction plus `executemany` favors simplicity and atomicity over COPY throughput.** It
  eliminates per-document commits without introducing pgvector COPY-format handling. The tradeoff is
  a longer write transaction.
- **Cross-run reuse is model-specific.** Reuse is valid only for the same `embed_model`; changing
  models correctly forces re-embedding.
- **Bio changes favor correctness over a cheap stale skip.** Because the bio enriches non-bio
  passage embeddings, changing it intentionally reprocesses every dependent document.
- **Concurrency favors clarity and memory over speculative speedups.** Multiprocessing, pipeline
  overlap, and index rebuild optimizations remain gated on measured need.

Layer 4's implementation mechanisms are shipped on the Twitter path, but the controlled stress proof
and the full multi-source brief remain incomplete.

---

## Getting started

> **Prerequisites:** **Docker** (runs the database and backend) and **Node 18+** (frontend).
> Python **3.11+** is only needed to run the test suite on the host.

```bash
# optional: add a DeepSeek API key for generated answers
# (retrieval + cited sources work without it; chat returns 503 until a key is set)
cp .env.example .env        # then edit DEEPSEEK_API_KEY=...

# start Postgres + the API together
docker compose up

# start the React UI in another shell
cd frontend
npm install
npm run dev

# stop everything when done (indexed data persists in the pgdata volume)
docker compose down
```

Then open the URL Vite prints (typically <http://localhost:5173>); it proxies `/chat` and `/health`
to the API on `http://localhost:8000`.

The **first** `docker compose up` is the only slow run: it builds the backend image, downloads the
BGE model into the `hfcache` volume, and indexes the bundled Twitter fixture once (auto-seed). Every
later run reuses the cached image, the cached model, and the persisted vectors — embedding does
**not** run again. Changed dependencies need a one-time `docker compose up --build`; to wipe the
indexed data and re-seed from scratch, use `docker compose down -v`.

### Testing

Tests run on the host against the Postgres started by `docker compose up`.

> **Heads-up:** the integration tier `TRUNCATE`s the `chunks`/`documents` tables, so it wipes the
> auto-seeded demo data. Run `docker compose restart api` afterward to re-seed.

```bash
pip install -e ".[dev]"     # package + test extras (downloads the BGE model on first integration run)

# pure tier — no DB, no model download
pytest -m "not integration and not live"

# integration tier — real pgvector + real BGE model (LLM faked)
pytest -m "integration and not live"

# optional: live DeepSeek smoke test (requires DEEPSEEK_API_KEY in .env)
pytest -m live

# retrieval eval that decides the hybrid default
python eval/retrieval_eval.py

# optional: inspect Layer 1 ingestion only (no DB, prints the ingestion report)
python -m cortex.pipeline.ingest tests/fixtures/twitter
```

The optional ingest command prints a one-line summary plus a JSON report, e.g.:

```
twitter: 58 kept (bio 1, post 51, thread 6), 24 dropped (reply_to_other 13, retweet 11), 7 skipped (malformed 2, empty 5) in 0.002s
```

On the **first** `docker compose up`, auto-seed indexes the fixture and the API logs the same kind of
index report:

```
twitter: 58 changed (new 58, updated 0), 0 unchanged skipped, 60 chunks inserted, 60 chunks embedded, 0 reused cross-run (0 deduped), 1 embed batches in 1.518s
```

On every later `docker compose up`, the knowledge base already has rows, so seeding is skipped
entirely — no model embedding work runs.

---

## Project structure

```
cortex/
├── README.md
├── pyproject.toml
├── Dockerfile                 # backend image for `docker compose up`
├── docker-compose.yml         # Postgres + API, with persisted volumes
├── .dockerignore
├── .env.example
├── .github/workflows/ci.yml
├── backend/cortex/
│   ├── config.py              # settings from env / .env
│   ├── models.py              # ContentItem, IngestionReport, IndexReport
│   ├── chunking/              # structure-aware chunker + embed_text enrichment
│   ├── embedding/             # Embedder ABC + SentenceTransformerEmbedder
│   ├── ingestion/             # base ABC, registry, normalize, twitter parser
│   ├── rag/                   # retriever, grounded prompt, DeepSeek chat client
│   ├── api/                   # FastAPI /health and /chat
│   ├── store/                 # schema.sql, db connection, repository helpers
│   └── pipeline/              # ingest.py (Layer 1), index.py (Layer 2)
├── eval/                      # retrieval gold set + semantic-vs-hybrid eval
├── frontend/                  # Vite + React chat page
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

**Layer 3 (chat/RAG) is shipped** — semantic retrieval, optional hybrid RRF retrieval, document-level
citations, grounded prompt assembly, DeepSeek chat, FastAPI `/chat`, a minimal React UI, fake-LLM
integration tests, and an optional live DeepSeek smoke test are implemented.

**Layer 4 (efficiency) is partially shipped** — streaming Twitter parsing, one-transaction writes,
same-model cross-run vector reuse, and enrichment-context invalidation are implemented and tested.
The controlled 5/25/50 MB parse-memory proof and ~10k-chunk throughput/RSS benchmark are deferred.

LinkedIn and Instagram parsers still follow on the Layer 1 seam. Layer 3 is platform-agnostic across
whatever is in the store, but the full brief's multi-source requirement is not complete until those
parsers land.

> Known limitation: the retweet-shape classification rule is derived from secondary sources and
> remains flagged for validation because the real June 2026 export used for parser auditing contained
> no retweets. Quote-status link handling was validated against that export.

> Known Layer 2 limitation: templated enrichment is the shipped zero-cost baseline. LLM enrichment
> for short, context-poor posts is intentionally deferred; it is the likely answer-quality knob to
> re-measure on a real corpus.
