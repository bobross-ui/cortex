from datetime import datetime, timezone

from cortex.chunking.enrich import author_blurb, build_embed_text
from cortex.models import ContentItem


def _item(
    *,
    content_type: str = "post",
    text: str = "Raw chunk text.",
    created_at: datetime | None = datetime(2024, 5, 15, 10, 0, tzinfo=timezone.utc),
) -> ContentItem:
    return ContentItem(
        source_platform="twitter",
        external_id="1001",
        content_type=content_type,
        text=text,
        author_handle="cortex_demo",
        created_at=created_at,
        url="https://twitter.com/cortex_demo/status/1001",
        metadata={},
    )


def test_build_embed_text_uses_exact_head_and_blurb_for_non_bio():
    item = _item(content_type="thread")

    embed_text = build_embed_text(
        item,
        "Async communication beats synchronous meetings.",
        "Engineer writing about remote work.",
    )

    assert embed_text == (
        "[twitter · thread · 2024-05-15] Engineer writing about remote work.\n"
        "Async communication beats synchronous meetings."
    )


def test_build_embed_text_omits_blurb_for_bio_item():
    item = _item(
        content_type="bio",
        text="Engineer writing about remote work.",
        created_at=None,
    )

    embed_text = build_embed_text(
        item,
        "Engineer writing about remote work.",
        "Engineer writing about remote work.",
    )

    assert embed_text == "[twitter · bio · undated]\nEngineer writing about remote work."


def test_build_embed_text_uses_undated_without_created_at():
    item = _item(created_at=None)

    embed_text = build_embed_text(item, "Short post.", "")

    assert embed_text == "[twitter · post · undated]\nShort post."


def test_author_blurb_picks_first_bio_item():
    items = [
        _item(content_type="post", text="First post."),
        _item(content_type="bio", text="Profile context."),
        _item(content_type="bio", text="Later profile context."),
    ]

    assert author_blurb(items) == "Profile context."


def test_author_blurb_returns_empty_string_when_absent():
    assert author_blurb([_item(content_type="post", text="Only a post.")]) == ""
