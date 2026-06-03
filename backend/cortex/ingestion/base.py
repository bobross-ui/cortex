from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from cortex.models import ContentItem, IngestionReport


class SourceParser(ABC):
    platform: str
    report: IngestionReport | None = None

    @abstractmethod
    def detect(self, root: Path) -> bool:
        """True iff this parser recognizes the export at `root` (an extracted directory)."""

    @abstractmethod
    def parse(self, root: Path) -> Iterator[ContentItem]:
        """Yield canonical items. MUST create a fresh self.report at entry.
        self.report is valid only after the iterator is fully consumed."""
