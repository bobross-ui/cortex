from dataclasses import dataclass
import re


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    index: int
    text: str


@dataclass
class _Unit:
    text: str
    paragraph_index: int


def estimate_tokens(text: str) -> int:
    # Conservative, deterministic estimate. Avoids loading the model tokenizer in unit tests.
    return max(1, round(len(text.split()) * 1.3))


def chunk_text(text: str, cfg) -> list[Chunk]:
    if estimate_tokens(text) <= cfg.chunk_target_tokens:
        return [Chunk(0, text)]

    units = _split_units(text, cfg.chunk_target_tokens)
    packed = _pack_units(units, cfg)
    return [Chunk(index=i, text=_join_units(chunk)) for i, chunk in enumerate(packed)]


def _split_units(text: str, target_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    for paragraph_index, paragraph in enumerate(paragraphs):
        if estimate_tokens(paragraph) <= target_tokens:
            units.append(_Unit(paragraph, paragraph_index))
            continue

        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                units.append(_Unit(sentence, paragraph_index))
    return units


def _pack_units(units: list[_Unit], cfg) -> list[list[_Unit]]:
    chunks: list[list[_Unit]] = []
    current: list[_Unit] = []

    for unit in units:
        candidate = [*current, unit]
        candidate_text = _join_units(candidate)
        unit_is_oversized = estimate_tokens(unit.text) > cfg.chunk_max_tokens

        if (
            current
            and estimate_tokens(candidate_text) > cfg.chunk_target_tokens
            and not unit_is_oversized
        ):
            chunks.append(current)
            current = _overlap_seed(current, cfg.chunk_overlap_tokens, cfg.chunk_max_tokens)
            candidate = [*current, unit]
            if estimate_tokens(_join_units(candidate)) <= cfg.chunk_max_tokens:
                current = candidate
            else:
                current = [unit]
            continue

        if current and unit_is_oversized:
            chunks.append(current)
            current = [unit]
            continue

        current = candidate

    if current:
        chunks.append(current)

    return chunks


def _overlap_seed(
    previous: list[_Unit],
    overlap_tokens: int,
    max_tokens: int,
) -> list[_Unit]:
    if overlap_tokens <= 0:
        return []

    seed: list[_Unit] = []
    for unit in reversed(previous):
        candidate = [unit, *seed]
        tokens = estimate_tokens(_join_units(candidate))
        if tokens > overlap_tokens or tokens > max_tokens:
            break
        seed = candidate
    return seed


def _join_units(units: list[_Unit]) -> str:
    if not units:
        return ""

    parts = [units[0].text]
    for previous, current in zip(units, units[1:]):
        joiner = " " if previous.paragraph_index == current.paragraph_index else "\n\n"
        parts.append(joiner)
        parts.append(current.text)
    return "".join(parts)
