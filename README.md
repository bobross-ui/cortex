# Cortex

> **Social Data → Vector Knowledge Base → Grounded Chat.** A personal RAG system that turns your
> social-media exports into a searchable, citable knowledge base you can ask questions of.

![status](https://img.shields.io/badge/status-Layer%201%20shipped-brightgreen)
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

This README grows one layer at a time. Only **Layer 1** is detailed below; later sections are added
as each layer lands.

| Layer | Scope | Status |
|---|---|---|
| **1 — Ingestion** | Pluggable parsers; authored-content extraction; Twitter first | ✅ **Shipped** |
| 2 — Vector KB | Structure-aware chunking, embeddings, schema | ⚪ Planned |
| 3 — Chat (RAG) | Retrieval, grounded prompt, cited answers, UI | ⚪ Planned |
| 4 — Efficiency | Streaming, batching, dedup, incremental upsert + measured numbers | ⚪ Planned |

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

## Getting started

> **Prerequisite:** Python **3.11+**. (A fresh virtualenv is recommended: `python3.11 -m venv .venv && source .venv/bin/activate`.)

```bash
# install (editable, with dev/test extras)
pip install -e ".[dev]"

# ingest the bundled Twitter fixture and print the ingestion report
python -m cortex.pipeline.ingest tests/fixtures/twitter

# run the test suite
pytest
```

Ingesting the fixture prints a one-line summary plus a JSON report, e.g.:

```
twitter: 5 kept (bio 1, post 3, thread 1), 2 dropped (reply_to_other 1, retweet 1), 2 skipped (malformed 1, empty 1) in 0.002s
```

---

## Project structure

```
cortex/
├── README.md
├── pyproject.toml
├── backend/cortex/
│   ├── models.py              # ContentItem, IngestionReport
│   ├── ingestion/             # base ABC, registry, normalize, twitter parser
│   └── pipeline/ingest.py     # CLI entrypoint
└── tests/
    ├── fixtures/twitter/      # synthetic test archive
    └── test_*.py
```

---

## Status

**Layer 1 (ingestion) is shipped** — the pluggable parser seam, the Twitter parser, and the
ingestion report are implemented and covered by tests against a high-fidelity synthetic Twitter
fixture. LinkedIn and Instagram parsers follow on the same seam. This README will be extended as
Layers 2–4 land.

> Known limitation: two Twitter classification rules (retweet & quote-tweet shape) are derived
> from secondary sources and flagged for validation against a real export.
