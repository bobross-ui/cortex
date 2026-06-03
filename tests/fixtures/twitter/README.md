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
Files are intentionally **reverse-chronological with the thread scrambled** to prove the
two-pass stitch (§4) sorts correctly and is order-independent.

## Expected parser outcome (assert against this)

| id | created_at | what it is | Expected | Reason / exercises |
|----|------------|------------|----------|--------------------|
| `1001` | 2024-05-15 10:00 | thread root | **KEEP** → `thread` (root) | self-reply thread head |
| `1002` | 2024-05-15 10:01 | self-reply "1/…" | folded into `1001` | `in_reply_to_user_id_str == accountId` |
| `1003` | 2024-05-15 10:02 | self-reply "2/…" | folded into `1001` | thread member; `&amp;` → `&` normalization |
| `1004` | 2024-05-12 09:15 | original + shared link | **KEEP** → `post` | authored t.co → expand to `blog.example.com/new-feature` |
| `1005` | 2024-05-13 14:20 | reply to `@someone_else` | **DROP** `reply_to_other` | `in_reply_to_user_id_str != accountId` |
| `1006` | 2024-05-16 08:00 | retweet `RT @influencer:` | **DROP** `retweet` | **`[UNVERIFIED]` rule:** `full_text` starts `RT @` |
| `1007` | 2024-05-18 17:30 | quote tweet + commentary | **KEEP** → `post`, `metadata.quote_of_id="8888"` | **`[UNVERIFIED]` rule:** `expanded_url` matches `/status/<id>` + added text; strip the quote link from text |
| `1008` | 2024-05-14 12:00 | media-only (no caption) | **DROP** `empty` | text is only a media t.co → empty after strip (§7 G7) |
| `1010` | 2024-05-20 07:45 | emoji post | **KEEP** → `post` | emoji preserved; `&amp;` → `&` |
| *(none)* | 2024-05-10 06:00 | object with **no `id_str`** | **SKIP** `malformed` | defensive skip, run must not crash |

### Expected `IngestionReport` (§8)
- `items_kept = 5` → `by_content_type = {thread: 1, post: 3, bio: 1}`  *(1001 thread; 1004, 1007, 1010 posts; bio from profile.js)*
- `items_dropped_noise = 2` → `dropped_reasons = {retweet: 1, reply_to_other: 1}`
- `items_skipped_empty = 1`  *(1008 media-only)*
- `items_skipped_malformed = 1`  *(the no-`id_str` object)*
- `thread_members_folded = 2`  *(1002, 1003 fold into 1001)*
- `files_seen = 1`, `peak_rss_mb = None`
- **Object conservation** (10 tweet objects in `tweets.js`): 4 kept tweet-items (1 thread + 3 posts) + 2 folded + 2 dropped + 1 empty + 1 malformed = 10. *(bio is the 5th kept item — it comes from `profile.js`, not a tweet object.)*
- thread `1001`: `metadata.thread_len = 3`, `member_ids = ["1001","1002","1003"]`, text = root + `1/…` + `2/…` joined chronologically with `\n\n`.

### Normalization checks embedded here
- `&amp;` → `&` (1003, 1010).
- t.co expansion of an **authored** link → real URL (1004); t.co **media** link → stripped (1008).
- quote-status t.co → stripped from text, captured as `quote_of_id` metadata (1007).
- emoji survive round-trip (1010).

## Profile / bio
`data/profile.js` → `description.bio` becomes a separate `content_type='bio'` `ContentItem`
(`external_id="profile"`, stable). Website `t.co` is profile chrome, not authored prose.

## Not yet covered (add when the real archive lands)
- `tweets-part0.js` / `tweets-part1.js` split (part-file globbing) — current fixture is a single
  `tweets.js`. A `twitter_split/` variant will exercise chaining.
- Real retweet & quote-tweet shapes (the two `[UNVERIFIED]` rows above).
- BOM-prefixed file + non-UTF-8 byte (encoding robustness, §7).
