from dataclasses import replace
from datetime import datetime, timezone

from cortex.models import ContentItem
from cortex.pipeline.index import (
    attach_embeddings,
    document_change_hash,
    sha256_text,
    unique_embed_texts,
)


def test_sha256_text_is_stable():
    text = "Raw authored source text."

    assert sha256_text(text) == (
        "611bc08bff26f1936a8af2f7be3d072e1f33a612eddc1440acb6c98b3e95b17f"
    )


def test_document_change_hash_includes_embed_context():
    item = ContentItem(
        source_platform="twitter",
        external_id="1",
        content_type="post",
        text="Raw authored source text.",
        author_handle="author",
        created_at=datetime(2024, 5, 15, tzinfo=timezone.utc),
        url=None,
    )

    original = document_change_hash(item, "Original bio")

    assert original != document_change_hash(
        replace(item, text="Updated authored source text."),
        "Original bio",
    )
    assert original != document_change_hash(item, "Updated bio")
    assert original != document_change_hash(
        replace(item, source_platform="linkedin"),
        "Original bio",
    )
    assert original != document_change_hash(
        replace(item, content_type="article"),
        "Original bio",
    )
    assert original != document_change_hash(
        replace(item, created_at=datetime(2024, 5, 16, tzinfo=timezone.utc)),
        "Original bio",
    )


def test_bio_document_change_hash_does_not_self_include_blurb():
    bio = ContentItem(
        source_platform="twitter",
        external_id="profile",
        content_type="bio",
        text="The profile bio.",
        author_handle="author",
        created_at=None,
        url=None,
    )

    assert document_change_hash(bio, "Old bio") == document_change_hash(bio, "New bio")


def test_chunk_content_hash_is_sha256_of_embed_text():
    embed_text = "[twitter · post · 2024-05-15] Context\nRaw chunk text."

    assert sha256_text(embed_text) == (
        "e5d7c202472b298dd26af15ef1cc97d9d5440e29efc6d0c400c8b1bcbd0d1841"
    )


def test_unique_embed_texts_preserves_order_and_dedups_once():
    rows = [
        {"chunk_index": 0, "embed_text": "same"},
        {"chunk_index": 1, "embed_text": "different"},
        {"chunk_index": 2, "embed_text": "same"},
    ]

    assert unique_embed_texts(rows) == ["same", "different"]


def test_attach_embeddings_maps_duplicate_embed_texts_to_same_vector():
    rows = [
        {"chunk_index": 0, "embed_text": "same", "embedding": None},
        {"chunk_index": 1, "embed_text": "different", "embedding": None},
        {"chunk_index": 2, "embed_text": "same", "embedding": None},
    ]
    same_vector = [1.0, 0.0]
    different_vector = [0.0, 1.0]

    attach_embeddings(rows, {"same": same_vector, "different": different_vector})

    assert rows[0]["embedding"] is same_vector
    assert rows[1]["embedding"] is different_vector
    assert rows[2]["embedding"] is same_vector
