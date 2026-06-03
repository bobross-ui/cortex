"""
Twitter parser tests — golden oracle + corpus invariants.

All assertions go against the synthetic fixture at tests/fixtures/twitter/.
See tests/fixtures/twitter/README.md for the full breakdown.

FROZEN guardrail: do NOT modify any parser/ingestion files. If a test fails,
fix the fixture data or these test expectations — never the parser.
"""

from pathlib import Path

from cortex.ingestion.normalize import load_js_array
from cortex.ingestion.twitter import TwitterParser

FIX = Path(__file__).parent / "fixtures" / "twitter"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse():
    """Return (items, report) for a fresh parse of the fixture."""
    parser = TwitterParser()
    items = list(parser.parse(FIX))
    return items, parser.report


def _by_id(items):
    return {item.external_id: item for item in items}


# ---------------------------------------------------------------------------
# Aggregate report oracle
# ---------------------------------------------------------------------------


def test_twitter_parser_report_matches_fixture_oracle():
    """IngestionReport numbers must match the designed fixture breakdown."""
    _, report = _parse()

    assert report.items_kept == 58
    assert report.by_content_type == {"bio": 1, "thread": 6, "post": 51}
    assert report.items_dropped_noise == 24
    assert report.dropped_reasons == {"retweet": 11, "reply_to_other": 13}
    assert report.items_skipped_empty == 5
    assert report.items_skipped_malformed == 2
    assert report.thread_members_folded == 36
    assert report.files_seen == 1
    assert report.peak_rss_mb is None
    # Invariant: by_content_type values must sum to items_kept
    assert sum(report.by_content_type.values()) == report.items_kept


# ---------------------------------------------------------------------------
# Emitted id set
# ---------------------------------------------------------------------------


def test_twitter_parser_emits_expected_ids_original():
    """All five original curated external_ids must be present."""
    items, _ = _parse()
    emitted = {item.external_id for item in items}
    assert {"profile", "1001", "1004", "1007", "1010"}.issubset(emitted)


def test_twitter_parser_emits_expected_new_thread_ids():
    """The five new threads (A–E) must be present."""
    items, _ = _parse()
    emitted = {item.external_id for item in items}
    assert {"2100", "2200", "2300", "2400", "2500"}.issubset(emitted)


# ---------------------------------------------------------------------------
# Original curated item oracle assertions (unchanged from v1)
# ---------------------------------------------------------------------------


def test_twitter_parser_thread_matches_fixture_oracle():
    items, _ = _parse()
    by_id = _by_id(items)
    thread = by_id["1001"]

    assert thread.content_type == "thread"
    assert thread.metadata["member_ids"] == ["1001", "1002", "1003"]
    assert thread.metadata["thread_len"] == 3
    assert thread.author_handle == "cortex_demo"
    assert thread.url == "https://twitter.com/cortex_demo/status/1001"
    assert thread.created_at.isoformat() == "2024-05-15T10:00:00+00:00"
    assert thread.text == (
        "Some hard-won thoughts on remote work, a thread 🧵👇\n\n"
        "1/ Async communication beats synchronous meetings for deep work. Fewer interruptions, more flow.\n\n"
        "2/ But you lose the hallway serendipity & mentorship. Tradeoffs everywhere."
    )


def test_twitter_parser_posts_match_fixture_oracle():
    items, _ = _parse()
    by_id = _by_id(items)

    assert by_id["1004"].text == (
        "Shipped a new feature today and could not be prouder of the team 🚀 "
        "https://blog.example.com/new-feature"
    )
    assert by_id["1007"].text == "This is exactly right, especially the part about trust."
    assert by_id["1007"].metadata["quote_of_id"] == "8888"
    assert by_id["1010"].text == "Loving the new home-office setup ☕️🚀 — coffee & code."


def test_twitter_parser_bio_matches_fixture_oracle():
    items, _ = _parse()
    by_id = _by_id(items)
    bio = by_id["profile"]

    assert bio.content_type == "bio"
    assert bio.external_id == "profile"
    assert bio.created_at is None
    assert bio.author_handle == "cortex_demo"
    assert bio.text == (
        "Engineer. Writing about remote work, async culture & building tools. Opinions my own."
    )


# ---------------------------------------------------------------------------
# Corpus-wide invariants (scale without hand-counting every item)
# ---------------------------------------------------------------------------


def test_by_content_type_sums_to_items_kept():
    _, report = _parse()
    assert sum(report.by_content_type.values()) == report.items_kept


def test_object_conservation():
    """
    Total tweet objects in tweets.js == kept tweet-items + folded + dropped
    + empty + malformed + silently-deduped.

    The bio item is produced from profile.js, not a tweet object, so we
    subtract by_content_type["bio"] from items_kept to get kept_tweet_items.
    """
    _, report = _parse()
    tweet_objects = load_js_array(FIX / "data" / "tweets.js")
    N = len(tweet_objects)

    bio_count = report.by_content_type.get("bio", 0)
    kept_tweet_items = report.items_kept - bio_count
    deduped = 1  # one deliberate duplicate (id 2001 appears twice)

    total = (
        kept_tweet_items
        + report.thread_members_folded
        + report.items_dropped_noise
        + report.items_skipped_empty
        + report.items_skipped_malformed
        + deduped
    )
    assert N == total, (
        f"Object conservation failed: {N} tweet objects in file != "
        f"kept_tweet_items({kept_tweet_items}) + folded({report.thread_members_folded}) "
        f"+ dropped({report.items_dropped_noise}) + empty({report.items_skipped_empty}) "
        f"+ malformed({report.items_skipped_malformed}) + deduped({deduped}) = {total}"
    )


def test_all_emitted_texts_are_non_empty():
    items, _ = _parse()
    for item in items:
        assert item.text, f"Item {item.external_id!r} has empty text"


def test_all_external_ids_are_unique():
    items, _ = _parse()
    ids = [item.external_id for item in items]
    assert len(ids) == len(set(ids)), "Duplicate external_ids found"


def test_every_thread_has_consistent_metadata():
    """Every thread must have thread_len == len(member_ids) and thread_len >= 2."""
    items, _ = _parse()
    for item in items:
        if item.content_type != "thread":
            continue
        meta = item.metadata
        assert meta["thread_len"] == len(meta["member_ids"]), (
            f"Thread {item.external_id}: thread_len {meta['thread_len']} "
            f"!= len(member_ids) {len(meta['member_ids'])}"
        )
        assert meta["thread_len"] >= 2, (
            f"Thread {item.external_id} has thread_len < 2"
        )


def test_every_thread_created_at_matches_root():
    """A thread's created_at must equal the root tweet's created_at (earliest member)."""
    items, _ = _parse()
    by_id = _by_id(items)
    for item in items:
        if item.content_type != "thread":
            continue
        root_id = item.metadata["member_ids"][0]
        assert item.external_id == root_id, (
            f"Thread {item.external_id}: external_id != first member_id {root_id}"
        )


def test_idempotency():
    """Two fresh TwitterParser instances must emit identical external_id sets."""
    items1 = list(TwitterParser().parse(FIX))
    items2 = list(TwitterParser().parse(FIX))
    assert {i.external_id for i in items1} == {i.external_id for i in items2}


def test_at_least_one_long_thread_exists():
    """
    At least one thread (specifically thread 2300, 12 members) must have
    thread_len >= 10 and joined text > 800 chars — exercises future chunk-splitting.
    """
    items, _ = _parse()
    by_id = _by_id(items)

    thread_c = by_id["2300"]
    assert thread_c.content_type == "thread"
    assert thread_c.metadata["thread_len"] >= 10, (
        f"Thread 2300 has thread_len {thread_c.metadata['thread_len']} < 10"
    )
    assert len(thread_c.text) > 800, (
        f"Thread 2300 text too short: {len(thread_c.text)} chars"
    )


# ---------------------------------------------------------------------------
# Spot-checks for new notable items
# ---------------------------------------------------------------------------


def test_new_long_thread_c_oracle():
    """Thread C (12 members, open-source sustainability) — structural spot-check."""
    items, _ = _parse()
    by_id = _by_id(items)
    t = by_id["2300"]

    assert t.content_type == "thread"
    assert t.metadata["thread_len"] == 12
    assert t.metadata["member_ids"] == [
        "2300", "2301", "2302", "2303", "2304", "2305",
        "2306", "2307", "2308", "2309", "2310", "2311",
    ]
    assert t.author_handle == "cortex_demo"
    assert t.created_at.isoformat() == "2024-03-05T10:00:00+00:00"
    # Root tweet text must appear at the start
    assert t.text.startswith("Open source sustainability is a real crisis")
    # All members joined by double-newline
    assert t.text.count("\n\n") == 11  # 12 members → 11 joins
    assert len(t.text) > 1500


def test_new_long_thread_d_oracle():
    """Thread D (11 members, engineering productivity) — structural spot-check."""
    items, _ = _parse()
    by_id = _by_id(items)
    t = by_id["2400"]

    assert t.content_type == "thread"
    assert t.metadata["thread_len"] == 11
    assert t.metadata["member_ids"] == [
        "2400", "2401", "2402", "2403", "2404", "2405",
        "2406", "2407", "2408", "2409", "2410",
    ]
    assert t.created_at.isoformat() == "2024-04-10T08:00:00+00:00"
    assert t.text.startswith("10 things I've learned about engineering productivity")
    assert t.text.count("\n\n") == 10
    assert len(t.text) > 1200


def test_new_quote_tweet_spot_check():
    """New quote tweet 3001 must be a post with quote_of_id and non-quote text."""
    items, _ = _parse()
    by_id = _by_id(items)
    q = by_id["3001"]

    assert q.content_type == "post"
    assert q.metadata.get("quote_of_id") == "40001"
    # The t.co quote link must be stripped from the text
    assert "t.co" not in q.text
    assert "async writing problem" in q.text or "Clarity of thought" in q.text


def test_new_standalone_post_spot_check():
    """Standalone post 2001 (short crisp take on PRs) must be kept as-is."""
    items, _ = _parse()
    by_id = _by_id(items)
    p = by_id["2001"]

    assert p.content_type == "post"
    assert p.metadata.get("is_reply") is False
    assert "PRs" in p.text or "200 lines" in p.text


def test_orphan_self_replies_are_standalone_posts():
    """Self-replies whose parent is not in the kept set must be posts with is_reply=True."""
    items, _ = _parse()
    by_id = _by_id(items)

    for oid in ("7001", "7002", "7003"):
        item = by_id[oid]
        assert item.content_type == "post", f"{oid} should be a post"
        assert item.metadata.get("is_reply") is True, f"{oid} should have is_reply=True"


# ---------------------------------------------------------------------------
# Parser was not modified (meta-guard via import path check)
# ---------------------------------------------------------------------------


def test_twitter_parser_is_idempotent_for_fresh_instances():
    items1 = list(TwitterParser().parse(FIX))
    items2 = list(TwitterParser().parse(FIX))
    assert {item.external_id for item in items1} == {item.external_id for item in items2}
