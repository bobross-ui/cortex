from types import SimpleNamespace

from cortex.chunking.chunker import chunk_text, estimate_tokens


def _cfg(target=8, max_tokens=16, overlap=6):
    return SimpleNamespace(
        chunk_target_tokens=target,
        chunk_max_tokens=max_tokens,
        chunk_overlap_tokens=overlap,
    )


def _paragraph(prefix: str, words: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(words)) + "."


def test_short_text_returns_single_chunk_equal_to_input():
    text = "Async writing creates better decision records."

    chunks = chunk_text(text, _cfg(target=350, max_tokens=480, overlap=40))

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == text


def test_long_multi_paragraph_input_has_pinned_boundaries_and_overlap():
    paragraphs = [
        _paragraph("a", 4),
        _paragraph("b", 4),
        _paragraph("c", 4),
        _paragraph("d", 4),
    ]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, _cfg())

    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.text for chunk in chunks] == [
        paragraphs[0],
        f"{paragraphs[0]}\n\n{paragraphs[1]}",
        f"{paragraphs[1]}\n\n{paragraphs[2]}",
        f"{paragraphs[2]}\n\n{paragraphs[3]}",
    ]
    assert all(estimate_tokens(chunk.text) <= 16 for chunk in chunks)
    assert paragraphs[0] in chunks[1].text
    assert paragraphs[1] in chunks[2].text
    assert paragraphs[2] in chunks[3].text

    reconstructed = []
    for chunk in chunks:
        for paragraph in chunk.text.split("\n\n"):
            if not reconstructed or paragraph != reconstructed[-1]:
                reconstructed.append(paragraph)
    assert "\n\n".join(reconstructed) == text


def test_oversized_paragraph_splits_into_sentences():
    sentences = [
        "one two three four five six.",
        "seven eight nine ten eleven twelve.",
        "thirteen fourteen fifteen sixteen seventeen eighteen.",
    ]
    text = " ".join(sentences)

    chunks = chunk_text(text, _cfg(target=8, max_tokens=16, overlap=8))

    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.text for chunk in chunks] == [
        sentences[0],
        f"{sentences[0]} {sentences[1]}",
        f"{sentences[1]} {sentences[2]}",
    ]
    assert all(estimate_tokens(chunk.text) <= 16 for chunk in chunks)


def test_overlap_does_not_skip_back_past_large_trailing_unit():
    first = _paragraph("small", 4)
    large_trailing = _paragraph("large", 8)
    next_paragraph = _paragraph("next", 4)
    text = "\n\n".join([first, large_trailing, next_paragraph])

    chunks = chunk_text(text, _cfg(target=18, max_tokens=24, overlap=6))

    assert [chunk.text for chunk in chunks] == [
        f"{first}\n\n{large_trailing}",
        next_paragraph,
    ]


def test_lone_oversized_sentence_is_emitted_without_word_splitting():
    text = " ".join(f"word{i}" for i in range(10))

    chunks = chunk_text(text, _cfg(target=5, max_tokens=8, overlap=2))

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == text
    assert estimate_tokens(chunks[0].text) > 8
