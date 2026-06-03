import json
from pathlib import Path

import cortex.ingestion.twitter as twitter_module
import pytest
from cortex.ingestion.normalize import iter_js_array, load_js_array
from cortex.ingestion.twitter import TwitterParser


FIX = Path(__file__).parent / "fixtures" / "twitter"
REPORT_FIELDS = (
    "items_kept",
    "by_content_type",
    "items_dropped_noise",
    "dropped_reasons",
    "items_skipped_malformed",
    "items_skipped_empty",
    "thread_members_folded",
    "files_seen",
)
EXPECTED_REPORT = {
    "items_kept": 58,
    "by_content_type": {"bio": 1, "thread": 6, "post": 51},
    "items_dropped_noise": 24,
    "dropped_reasons": {"retweet": 11, "reply_to_other": 13},
    "items_skipped_malformed": 2,
    "items_skipped_empty": 5,
    "thread_members_folded": 36,
    "files_seen": 1,
}


class _TrackingReader:
    """Cap individual reads so the test can observe whether iteration is lazy."""

    def __init__(self, raw, read_cap: int = 256):
        self.raw = raw
        self.read_cap = read_cap
        self.bytes_read = 0

    def read(self, size=-1):
        size = self.read_cap if size < 0 else min(size, self.read_cap)
        data = self.raw.read(size)
        self.bytes_read += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def __enter__(self):
        self.raw.__enter__()
        return self

    def __exit__(self, *args):
        return self.raw.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.raw, name)


def _write_js_array(path: Path, prefix: bytes, payload: list, suffix: bytes = b"") -> None:
    path.write_bytes(prefix + json.dumps(payload).encode() + suffix)


def _report_values(report) -> dict:
    return {field: getattr(report, field) for field in REPORT_FIELDS}


def test_streaming_parser_matches_eager_parser(monkeypatch):
    streaming_parser = TwitterParser()
    streaming_items = list(streaming_parser.parse(FIX))
    streaming_report = streaming_parser.report

    monkeypatch.setattr(
        twitter_module,
        "iter_js_array",
        lambda path: iter(load_js_array(path)),
    )
    eager_parser = TwitterParser()
    eager_items = list(eager_parser.parse(FIX))
    eager_report = eager_parser.report

    assert streaming_items == eager_items
    assert _report_values(streaming_report) == _report_values(eager_report)
    assert _report_values(streaming_report) == EXPECTED_REPORT


def test_iter_js_array_reads_plain_wrapper(tmp_path):
    payload = [{"tweet": {"id_str": "1"}}, {"tweet": {"id_str": "2"}}]
    path = tmp_path / "tweets.js"
    _write_js_array(path, b"window.YTD.tweets.part0 = ", payload)

    assert list(iter_js_array(path)) == payload


def test_iter_js_array_tolerates_bom(tmp_path):
    payload = [{"tweet": {"id_str": "1"}}]
    path = tmp_path / "tweets.js"
    _write_js_array(path, b"\xef\xbb\xbfwindow.YTD.tweets.part0 = ", payload)

    assert list(iter_js_array(path)) == payload


def test_iter_js_array_scans_long_prefix(tmp_path):
    payload = [{"tweet": {"id_str": "1"}}]
    path = tmp_path / "tweets-part17.js"
    prefix = b" " * 5000 + b"window.YTD.tweets.part17 = "
    _write_js_array(path, prefix, payload)

    assert list(iter_js_array(path)) == payload


def test_iter_js_array_ignores_trailing_content(tmp_path):
    payload = [{"tweet": {"id_str": "1"}}, {"tweet": {"id_str": "2"}}]
    path = tmp_path / "tweets.js"
    _write_js_array(path, b"window.YTD.tweets.part0 = ", payload, b";\n")

    assert list(iter_js_array(path)) == payload


def test_iter_js_array_yields_before_reading_generated_file_tail(tmp_path, monkeypatch):
    payload = [
        {"tweet": {"id_str": str(index), "full_text": f"generated row {index}"}}
        for index in range(10_000)
    ]
    path = tmp_path / "tweets.js"
    _write_js_array(path, b"window.YTD.tweets.part0 = ", payload)
    original_open = Path.open
    opened = {}

    def tracking_open(self, *args, **kwargs):
        raw = original_open(self, *args, **kwargs)
        if self != path:
            return raw
        reader = _TrackingReader(raw)
        opened["reader"] = reader
        return reader

    monkeypatch.setattr(Path, "open", tracking_open)
    rows = iter_js_array(path)
    try:
        assert next(rows) == payload[0]
        assert opened["reader"].bytes_read < path.stat().st_size // 10
    finally:
        rows.close()


def test_iter_js_array_rejects_file_without_array(tmp_path):
    path = tmp_path / "tweets.js"
    path.write_bytes(b"window.YTD.tweets.part0 = {};")

    with pytest.raises(ValueError, match="no JSON array found"):
        list(iter_js_array(path))


def test_twitter_parser_streams_split_part_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    account = [{"account": {"accountId": "42", "username": "split_demo"}}]
    _write_js_array(data / "account.js", b"window.YTD.account.part0 = ", account)

    first = [{"tweet": {"id_str": "1", "full_text": "first"}}]
    second = [{"tweet": {"id_str": "2", "full_text": "second"}}]
    _write_js_array(data / "tweets-part0.js", b"window.YTD.tweets.part0 = ", first)
    _write_js_array(data / "tweets-part1.js", b"window.YTD.tweets.part1 = ", second)

    parser = TwitterParser()
    items = list(parser.parse(tmp_path))

    assert [item.external_id for item in items] == ["1", "2"]
    assert parser.report.files_seen == 2
