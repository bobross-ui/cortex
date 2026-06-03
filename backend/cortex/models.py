from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ContentItem:
    source_platform: str
    external_id: str
    content_type: str
    text: str
    author_handle: str | None
    created_at: datetime | None
    url: str | None
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestionReport:
    platform: str
    root: str
    files_seen: int = 0
    items_kept: int = 0
    items_dropped_noise: int = 0
    items_skipped_malformed: int = 0
    items_skipped_empty: int = 0
    thread_members_folded: int = 0
    by_content_type: dict = field(default_factory=dict)
    dropped_reasons: dict = field(default_factory=dict)
    duration_s: float = 0.0
    peak_rss_mb: float | None = None

    def human_summary(self) -> str:
        by = ", ".join(f"{k} {v}" for k, v in sorted(self.by_content_type.items()))
        dr = ", ".join(f"{k} {v}" for k, v in sorted(self.dropped_reasons.items()))
        return (f"{self.platform}: {self.items_kept} kept ({by}), "
                f"{self.items_dropped_noise} dropped ({dr}), "
                f"{self.items_skipped_malformed + self.items_skipped_empty} skipped "
                f"(malformed {self.items_skipped_malformed}, empty {self.items_skipped_empty}) "
                f"in {self.duration_s:.3f}s")


@dataclass
class IndexReport:
    platform: str
    documents_new: int = 0
    documents_updated: int = 0
    documents_unchanged_skipped: int = 0
    chunks_inserted: int = 0
    chunks_embedded: int = 0
    chunks_dedup_within_run: int = 0
    embed_batches: int = 0
    duration_s: float = 0.0
    embed_duration_s: float = 0.0

    def chunks_per_s(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.chunks_inserted / self.duration_s

    def human_summary(self) -> str:
        docs_changed = self.documents_new + self.documents_updated
        return (
            f"{self.platform}: {docs_changed} changed "
            f"(new {self.documents_new}, updated {self.documents_updated}), "
            f"{self.documents_unchanged_skipped} unchanged skipped, "
            f"{self.chunks_inserted} chunks inserted, "
            f"{self.chunks_embedded} chunks embedded "
            f"({self.chunks_dedup_within_run} deduped), "
            f"{self.embed_batches} embed batches in {self.duration_s:.3f}s"
        )
