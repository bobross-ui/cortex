from pathlib import Path
from cortex.ingestion.twitter import TwitterParser


FIX = Path(__file__).parent / "fixtures" / "twitter"


def test_twitter_parser_report_matches_fixture_oracle():
    parser = TwitterParser()
    list(parser.parse(FIX))
    report = parser.report

    assert report.items_kept == 5
    assert report.by_content_type == {"thread": 1, "post": 3, "bio": 1}
    assert report.items_dropped_noise == 2
    assert report.dropped_reasons == {"retweet": 1, "reply_to_other": 1}
    assert report.items_skipped_empty == 1
    assert report.items_skipped_malformed == 1
    assert report.thread_members_folded == 2
    assert report.files_seen == 1
    assert report.peak_rss_mb is None
    assert sum(report.by_content_type.values()) == report.items_kept


def test_twitter_parser_emits_expected_ids():
    items = list(TwitterParser().parse(FIX))

    assert {item.external_id for item in items} == {
        "profile",
        "1001",
        "1004",
        "1007",
        "1010",
    }


def test_twitter_parser_thread_matches_fixture_oracle():
    items = list(TwitterParser().parse(FIX))
    by_id = {item.external_id: item for item in items}
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
    items = list(TwitterParser().parse(FIX))
    by_id = {item.external_id: item for item in items}

    assert by_id["1004"].text == (
        "Shipped a new feature today and could not be prouder of the team 🚀 "
        "https://blog.example.com/new-feature"
    )
    assert by_id["1007"].text == "This is exactly right, especially the part about trust."
    assert by_id["1007"].metadata["quote_of_id"] == "8888"
    assert by_id["1010"].text == "Loving the new home-office setup ☕️🚀 — coffee & code."


def test_twitter_parser_bio_matches_fixture_oracle():
    items = list(TwitterParser().parse(FIX))
    by_id = {item.external_id: item for item in items}
    bio = by_id["profile"]

    assert bio.content_type == "bio"
    assert bio.external_id == "profile"
    assert bio.created_at is None
    assert bio.author_handle == "cortex_demo"
    assert bio.text == (
        "Engineer. Writing about remote work, async culture & building tools. Opinions my own."
    )


def test_twitter_parser_is_idempotent_for_fresh_instances():
    items1 = list(TwitterParser().parse(FIX))
    items2 = list(TwitterParser().parse(FIX))

    assert {item.external_id for item in items1} == {item.external_id for item in items2}
