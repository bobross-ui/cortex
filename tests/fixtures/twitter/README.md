# Twitter fixture — synthetic archive (golden oracle)

A **high-fidelity synthetic** Twitter/X export built to the schema verified in
`LAYER1_PLAN.md` §6T. It mirrors the real archive: `data/*.js` files, the
`window.YTD.<name>.part0 = [ … ]` JS wrapper, the `{"tweet": {…}}` envelope, real field
names, the real `created_at` format, and `favorite_count`/`retweet_count` zeroed (as real
archives do).

> **Why synthetic + why a real archive still matters.** This fixture is our committed
> **regression oracle**. It cannot *self-validate* the two `[UNVERIFIED]` discriminators
> (retweet shape, quote-tweet shape) — parser and fixture share the same assumption, so a green
> test there only proves the mechanics, not the premise. Those two are confirmed against a real
> archive (own export, pending) and the rows below updated if reality differs.

Owner identity (`data/account.js`): handle **`cortex_demo`**, `accountId` **`1234567890`**.
Tweet objects are intentionally **not sorted** (reverse-chronological + threads scrambled) to
prove the two-pass stitch (§4) sorts correctly and is order-independent.

---

## Corpus summary (125 tweet objects total)

| Category | Object count | Parser outcome |
|---|---|---|
| Original curated tweets (ids 1001–1010) | 10 | See detailed table below |
| Thread A — async culture (5 members, ids 2100–2104) | 5 | KEEP → 1 thread, 4 folded |
| Thread B — hiring (7 members, ids 2200–2206) | 7 | KEEP → 1 thread, 6 folded |
| Thread C — OSS sustainability (12 members, ids 2300–2311) | 12 | KEEP → 1 thread, 11 folded |
| Thread D — engineering productivity (11 members, ids 2400–2410) | 11 | KEEP → 1 thread, 10 folded |
| Thread E — startup lessons (4 members, ids 2500–2503) | 4 | KEEP → 1 thread, 3 folded |
| Standalone posts (ids 2001–2040) | 40 | KEEP → 40 posts |
| Quote tweets (ids 3001–3005) | 5 | KEEP → 5 posts with `quote_of_id` |
| Retweets `RT @…` (ids 4001–4010) | 10 | DROP `retweet` |
| Replies to others (ids 5001–5012) | 12 | DROP `reply_to_other` |
| Media-only / empty-after-strip (ids 6001–6004) | 4 | SKIP `empty` |
| Orphan self-replies (ids 7001–7003) | 3 | KEEP → 3 posts with `is_reply=True` |
| Extra malformed object (no `id_str`) | 1 | SKIP `malformed` |
| Dedup duplicate (id 2001 repeated) | 1 | Silently skipped (not counted in any bucket) |

---

## Original curated edge-case rows (ids 1001–1010)

| id | created_at | what it is | Expected | Reason / exercises |
|----|------------|------------|----------|--------------------|
| `1001` | 2024-05-15 10:00 | thread root | **KEEP** → `thread` (root) | self-reply thread head |
| `1002` | 2024-05-15 10:01 | self-reply "1/…" | folded into `1001` | `in_reply_to_user_id_str == accountId` |
| `1003` | 2024-05-15 10:02 | self-reply "2/…" | folded into `1001` | thread member; `&amp;` → `&` normalization |
| `1004` | 2024-05-12 09:15 | original + shared link | **KEEP** → `post` | authored t.co → expand to `blog.example.com/new-feature` |
| `1005` | 2024-05-13 14:20 | reply to `@someone_else` | **DROP** `reply_to_other` | `in_reply_to_user_id_str != accountId` |
| `1006` | 2024-05-16 08:00 | retweet `RT @influencer:` | **DROP** `retweet` | **`[UNVERIFIED]` rule:** `full_text` starts `RT @` |
| `1007` | 2024-05-18 17:30 | quote tweet + commentary | **KEEP** → `post`, `metadata.quote_of_id="8888"` | **`[UNVERIFIED]` rule:** `expanded_url` matches `/status/<id>` + added text; strip the quote link from text |
| `1008` | 2024-05-14 12:00 | media-only (no caption) | **SKIP** `empty` | text is only a media t.co → empty after strip (§7 G7) |
| `1010` | 2024-05-20 07:45 | emoji post | **KEEP** → `post` | emoji preserved; `&amp;` → `&` |
| *(none)* | 2024-05-10 06:00 | object with **no `id_str`** | **SKIP** `malformed` | defensive skip, run must not crash |

---

## Expected `IngestionReport` (§8) — golden oracle

```
items_kept            = 58
by_content_type       = {bio: 1, thread: 6, post: 51}
items_dropped_noise   = 24   → dropped_reasons = {retweet: 11, reply_to_other: 13}
items_skipped_empty   = 5
items_skipped_malformed = 2
thread_members_folded = 36
files_seen            = 1
peak_rss_mb           = None
```

### Object conservation (125 tweet objects in `tweets.js`)

```
kept_tweet_items = items_kept − bio_count = 58 − 1 = 57
total = 57 (kept tweet-items)
      + 36 (folded)
      + 24 (dropped)
      +  5 (empty)
      +  2 (malformed)
      +  1 (silently deduped)
      = 125  ✓
```

The bio item (5th kept item, `external_id="profile"`) comes from `profile.js`, not a tweet object,
so it is excluded from the tweet-object conservation count.

---

## Thread oracle snapshots

| Thread | Root id | Members | thread_len | Long-form? | Root created_at |
|--------|---------|---------|-----------|-----------|----------------|
| 1001 (original) | `1001` | 1001, 1002, 1003 | 3 | no | 2024-05-15T10:00:00Z |
| Thread A | `2100` | 2100–2104 | 5 | no | 2024-01-08T09:00:00Z |
| Thread B | `2200` | 2200–2206 | 7 | no | 2024-02-02T15:00:00Z |
| **Thread C** | `2300` | 2300–2311 | **12** | **yes (>1500 chars)** | 2024-03-05T10:00:00Z |
| **Thread D** | `2400` | 2400–2410 | **11** | **yes (>1200 chars)** | 2024-04-10T08:00:00Z |
| Thread E | `2500` | 2500–2503 | 4 | no | 2024-05-02T12:00:00Z |

Threads C and D are deliberately long-form (10+ substantive members each, total text > 800 chars)
to exercise future Layer 2 chunk-splitting.

---

## Normalization checks embedded in the fixture

- `&amp;` → `&` (ids 1003, 1010).
- Authored t.co link → expanded URL (id 1004); media t.co link → stripped (ids 1008, 6001–6004).
- Quote-status t.co → stripped from text, captured as `quote_of_id` metadata (ids 1007, 3001–3005).
- Emoji survive round-trip (ids 1010, 2040).
- Leading `@mention` stripped on replies (ids 5001–5012); retained when not a reply.
- `full_text` intra-tweet newlines are collapsed to spaces (whitespace normalization); multi-paragraph
  long-form only comes from thread joins (`\n\n` between members).

## Orphan self-reply behavior

Tweet objects 7001–7003 are self-replies (`in_reply_to_user_id_str == "1234567890"`) whose parent
id (7000) is **not in the tweet archive**. The parser keeps them as standalone `post` items with
`metadata.is_reply = True` rather than folding them into a thread.

## Profile / bio

`data/profile.js` → `description.bio` becomes a separate `content_type='bio'` `ContentItem`
(`external_id="profile"`, stable). Website `t.co` is profile chrome, not authored prose.

## Not yet covered (add when the real archive lands)

- `tweets-part0.js` / `tweets-part1.js` split (part-file globbing) — current fixture is a single
  `tweets.js`. A `twitter_split/` variant will exercise chaining.
- Real retweet & quote-tweet shapes (the two `[UNVERIFIED]` rows above).
- BOM-prefixed file + non-UTF-8 byte (encoding robustness, §7).
