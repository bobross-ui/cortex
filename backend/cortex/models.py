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
