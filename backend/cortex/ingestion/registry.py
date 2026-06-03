from pathlib import Path
from cortex.ingestion.base import SourceParser


class UnknownExportError(Exception): ...


class AmbiguousExportError(Exception): ...


_PARSERS: list[SourceParser] = []


def register(parser: SourceParser) -> None:
    _PARSERS.append(parser)


def resolve(root: Path) -> SourceParser:
    matches = [p for p in _PARSERS if p.detect(root)]
    if not matches:
        raise UnknownExportError(f"No parser recognized export at {root}")
    if len(matches) > 1:
        raise AmbiguousExportError(
            f"{[p.platform for p in matches]} all claimed {root}")
    return matches[0]
