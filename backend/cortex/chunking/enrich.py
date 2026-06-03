from cortex.models import ContentItem


def author_blurb(items: list[ContentItem]) -> str:
    for it in items:
        if it.content_type == "bio":
            return it.text
    return ""


def build_embed_text(item: ContentItem, chunk_text: str, blurb: str) -> str:
    """Build passage embedding input; future LLM enrichment can swap in at this seam."""
    date = item.created_at.date().isoformat() if item.created_at else "undated"
    head = f"[{item.source_platform} · {item.content_type} · {date}]"
    # Do not prepend the blurb to the bio itself; it is already the blurb.
    if blurb and item.content_type != "bio":
        return f"{head} {blurb}\n{chunk_text}"
    return f"{head}\n{chunk_text}"
