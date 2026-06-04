"""Seeded, deterministic Twitter archive generator for stress testing.

Two modes:
  throughput — n distinct authored tweets (varied text so no embed-dedup collapses them).
               Includes a handful of long self-reply threads so some items chunk into >1 chunk.
  memory     — fixed authored count padded with noise (retweets + replies-to-others) until
               tweets.js reaches roughly size_mb megabytes. Keeps items_kept constant while
               the file grows, proving the parser streams noise away.

Usage:
  python -m eval.stress_fixture --mode throughput --n 200 --out /tmp/x
  python -m eval.stress_fixture --mode memory --n 2000 --size-mb 50 --out /tmp/y
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


_OWN_ID = "1234567890"
_OWN_HANDLE = "cortex_demo"
_NOISE_USER_ID = "9999999999"
_NOISE_HANDLE = "noise_user"

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Topics and phrasing templates to vary tweet text sufficiently.
_TOPICS = [
    "machine learning", "distributed systems", "type theory", "compilers",
    "database indexing", "graph algorithms", "functional programming",
    "operating systems", "network protocols", "cryptography",
    "programming languages", "software architecture", "code review",
    "testing strategies", "performance optimization", "memory management",
    "concurrency patterns", "API design", "dependency injection",
    "observability", "tracing", "metrics", "logging", "deployment",
    "containerization", "kubernetes", "CI/CD pipelines", "static analysis",
    "fuzzing", "formal verification", "proof assistants", "lambda calculus",
    "category theory", "monads", "functors", "type inference",
    "garbage collection", "reference counting", "arena allocation",
    "cache coherence", "NUMA architectures", "instruction pipelining",
    "branch prediction", "speculative execution", "SIMD intrinsics",
    "GPU computing", "tensor operations", "automatic differentiation",
    "backpropagation", "attention mechanisms", "transformer architectures",
]

_VERBS = [
    "is fundamentally about", "comes down to", "requires careful thinking about",
    "depends critically on", "teaches us that", "reveals the importance of",
    "often breaks down at the seam between", "demands clarity around",
]

_ELABORATIONS = [
    "The key insight is that abstraction boundaries must be drawn with care.",
    "Without a clear mental model, debugging becomes guesswork.",
    "The tradeoffs here are non-trivial and context-dependent.",
    "Most practitioners underestimate the complexity until it bites them.",
    "The literature covers this well but practitioners often skip it.",
    "A principled approach saves hours of debugging downstream.",
    "The fundamental tension is between simplicity and expressiveness.",
    "Good tooling can compensate, but only up to a point.",
    "The right abstraction level determines everything that follows.",
    "This is an area where experience compounds quickly.",
    "Benchmarks rarely capture the full picture here.",
    "The gotchas are well-documented but still surprising in practice.",
    "Theory and practice diverge sharply in production environments.",
    "The community consensus has shifted significantly over the past decade.",
    "There is an elegant solution hiding behind the apparent complexity.",
]

_FOLLOWUP_SENTENCES = [
    "Worth spending time on the fundamentals before optimizing.",
    "The standard approach works until it doesn't.",
    "Documentation rarely covers the edge cases that matter.",
    "The mental model needs to match the actual implementation.",
    "Start with the simplest thing that could possibly work.",
    "Measure before optimizing; intuition is often wrong.",
    "The right tool depends on the constraints, not the hype.",
    "Reading the source code reveals what the docs omit.",
    "Abstractions leak, but good ones leak predictably.",
    "Composition over configuration is the pattern that scales.",
    "The real cost is usually in the transitions between states.",
    "Isolation makes testing tractable; coupling makes it fragile.",
    "Invariants are the backbone of correct concurrent code.",
    "Type systems cannot catch all bugs but they catch many of the expensive ones.",
    "The failure mode of over-engineering is subtle but real.",
]


def _fmt_date(rng: random.Random) -> str:
    """Generate a valid Twitter date string deterministically."""
    year = rng.randint(2018, 2024)
    month_idx = rng.randint(0, 11)
    month = _MONTHS[month_idx]
    # Simple day range to avoid leap-year and 30/31 edge cases
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    weekday = _WEEKDAYS[rng.randint(0, 6)]
    return f"{weekday} {month} {day:02d} {hour:02d}:{minute:02d}:{second:02d} +0000 {year}"


def _authored_tweet_text(rng: random.Random, index: int) -> str:
    """Generate distinct, non-empty standalone tweet text. Includes the index for uniqueness."""
    topic1 = _TOPICS[rng.randint(0, len(_TOPICS) - 1)]
    topic2 = _TOPICS[rng.randint(0, len(_TOPICS) - 1)]
    verb = _VERBS[rng.randint(0, len(_VERBS) - 1)]
    elab = _ELABORATIONS[rng.randint(0, len(_ELABORATIONS) - 1)]
    follow = _FOLLOWUP_SENTENCES[rng.randint(0, len(_FOLLOWUP_SENTENCES) - 1)]
    # The index ensures global uniqueness even if all other choices collide.
    return (
        f"Tweet #{index}: Understanding {topic1} {verb} {topic2}. "
        f"{elab} {follow}"
    )


def _thread_member_text(rng: random.Random, thread_idx: int, member_idx: int) -> str:
    """Generate one thread member's text — must be distinct and non-empty.

    Each member is ~40-50 words to make the full thread exceed chunk_target_tokens=350
    (estimate_tokens = round(words*1.3); 12 members * 45 words = 540 words -> ~702 tokens).
    """
    topic1 = _TOPICS[rng.randint(0, len(_TOPICS) - 1)]
    topic2 = _TOPICS[rng.randint(0, len(_TOPICS) - 1)]
    elab = _ELABORATIONS[rng.randint(0, len(_ELABORATIONS) - 1)]
    follow1 = _FOLLOWUP_SENTENCES[rng.randint(0, len(_FOLLOWUP_SENTENCES) - 1)]
    follow2 = _FOLLOWUP_SENTENCES[rng.randint(0, len(_FOLLOWUP_SENTENCES) - 1)]
    # thread_idx + member_idx ensures global uniqueness across all thread members.
    return (
        f"Thread {thread_idx} member {member_idx}: A deep dive into {topic1} "
        f"and its relationship with {topic2}. {elab} {follow1} {follow2} "
        f"This is part {member_idx + 1} of the thread."
    )


def _make_tweet(tweet_id: str, full_text: str, created_at: str, *,
                in_reply_to_status_id_str: str | None = None,
                in_reply_to_user_id_str: str | None = None,
                in_reply_to_screen_name: str | None = None) -> dict:
    """Build a tweet dict matching the parser's expected shape."""
    tweet = {
        "id": tweet_id,
        "id_str": tweet_id,
        "full_text": full_text,
        "created_at": created_at,
        "lang": "en",
        "entities": {
            "hashtags": [],
            "symbols": [],
            "user_mentions": [],
            "urls": [],
        },
    }
    if in_reply_to_status_id_str is not None:
        tweet["in_reply_to_status_id_str"] = in_reply_to_status_id_str
        tweet["in_reply_to_user_id_str"] = in_reply_to_user_id_str
        tweet["in_reply_to_screen_name"] = in_reply_to_screen_name
    return {"tweet": tweet}


def _write_account_js(data_dir: Path) -> None:
    """Write data/account.js in the Twitter archive wrapper format."""
    payload = [{"account": {"accountId": _OWN_ID, "username": _OWN_HANDLE}}]
    content = f"window.YTD.account.part0 = {json.dumps(payload, indent=2)}"
    (data_dir / "account.js").write_text(content, encoding="utf-8")


def _write_profile_js(data_dir: Path) -> None:
    """Write data/profile.js in the Twitter archive wrapper format."""
    payload = [{
        "profile": {
            "description": {
                "bio": "Engineer exploring the intersection of systems, language, and learning.",
                "location": "San Francisco, CA",
                "website": "https://cortex.example.com",
            }
        }
    }]
    content = f"window.YTD.profile.part0 = {json.dumps(payload, indent=2)}"
    (data_dir / "profile.js").write_text(content, encoding="utf-8")


def _write_tweets_js(data_dir: Path, rows: list[dict]) -> None:
    """Write data/tweets.js — the array is valid JSON inside the wrapper prefix."""
    content = f"window.YTD.tweets.part0 = {json.dumps(rows)}"
    (data_dir / "tweets.js").write_text(content, encoding="utf-8")


def _generate_authored_rows(rng: random.Random, n: int, base_id: int,
                             n_threads: int = 4, thread_members: int = 12) -> list[dict]:
    """Generate authored tweet rows: some standalone posts and some self-reply threads.

    Args:
        rng: seeded RNG (must be called in a deterministic order).
        n: total number of authored tweets to generate.
        base_id: starting numeric id for tweets (to ensure disjoint id spaces).
        n_threads: number of long multi-member threads to include.
        thread_members: number of members per thread.

    Returns:
        List of {"tweet": {...}} dicts in the order they were created.
    """
    rows: list[dict] = []
    tweet_id_counter = base_id

    # Reserve some authored budget for threads; the rest are standalone posts.
    # Each thread uses thread_members authored slots.
    thread_tweet_count = n_threads * thread_members
    standalone_count = max(0, n - thread_tweet_count)

    tweet_index = 0

    # Generate standalone posts first.
    for _ in range(standalone_count):
        tid = str(tweet_id_counter)
        tweet_id_counter += 1
        text = _authored_tweet_text(rng, tweet_index)
        tweet_index += 1
        created_at = _fmt_date(rng)
        rows.append(_make_tweet(tid, text, created_at))

    # Generate self-reply threads.
    for thread_idx in range(n_threads):
        root_id = str(tweet_id_counter)
        tweet_id_counter += 1
        root_text = _thread_member_text(rng, thread_idx, 0)
        tweet_index += 1
        created_at = _fmt_date(rng)
        rows.append(_make_tweet(root_id, root_text, created_at))

        prev_id = root_id
        for member_idx in range(1, thread_members):
            mid = str(tweet_id_counter)
            tweet_id_counter += 1
            member_text = _thread_member_text(rng, thread_idx, member_idx)
            tweet_index += 1
            created_at = _fmt_date(rng)
            rows.append(_make_tweet(
                mid, member_text, created_at,
                in_reply_to_status_id_str=prev_id,
                in_reply_to_user_id_str=_OWN_ID,
                in_reply_to_screen_name=_OWN_HANDLE,
            ))
            prev_id = mid

    return rows


def generate(
    out_dir: Path,
    *,
    mode: str,
    n: int = 10_000,
    size_mb: int | None = None,
    seed: int = 0,
) -> Path:
    """Write account.js / profile.js / data files for a synthetic Twitter archive into out_dir.
    Returns the export root (the dir containing `data/`). Deterministic for a given seed.

    throughput: n distinct authored tweets (varied text -> no embed dedup collapse).
                Includes a handful of long self-reply threads to exercise the chunker.
    memory:     fixed authored count + noise rows (retweets + replies-to-others) until
                tweets.js reaches roughly size_mb megabytes. The authored count is
                INDEPENDENT of size_mb so items_kept stays constant as file grows.
    """
    if mode not in ("throughput", "memory"):
        raise ValueError(f"mode must be 'throughput' or 'memory', got {mode!r}")

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    _write_account_js(data_dir)
    _write_profile_js(data_dir)

    if mode == "throughput":
        _generate_throughput(rng, data_dir, n)
    else:
        if size_mb is None:
            raise ValueError("size_mb is required for memory mode")
        _generate_memory(rng, data_dir, n, size_mb)

    return out_dir


def _generate_throughput(rng: random.Random, data_dir: Path, n: int) -> None:
    """Throughput mode: n distinct authored tweets including a few long threads."""
    # Authored tweet ids start at 1_000_000_000 (10 digits, well above noise range).
    authored_rows = _generate_authored_rows(rng, n, base_id=1_000_000_000)
    _write_tweets_js(data_dir, authored_rows)


def _generate_memory(rng: random.Random, data_dir: Path, n: int, size_mb: int) -> None:
    """Memory mode: fixed authored count + noise padding to target file size.

    Authored rows are generated first using a deterministic RNG slice, independent
    of size_mb. Noise rows are appended until the content reaches the size target.
    """
    target_bytes = size_mb * 1024 * 1024

    # Authored tweets: ids in range [1_000_000_000, 2_000_000_000).
    # Use fewer threads for memory mode (authored count is smaller).
    n_threads = min(4, max(0, n // 100))
    thread_members = 12
    authored_rows = _generate_authored_rows(
        rng, n, base_id=1_000_000_000,
        n_threads=n_threads, thread_members=thread_members,
    )

    # Serialize authored rows to count their contribution.
    # We'll build the full list and serialize once at the end to avoid O(n^2).
    # Track running byte size incrementally using per-row json sizes.
    # json.dumps(rows) is: "[" + ", ".join(json.dumps(row) for row in rows) + "]"
    # Incremental approximation: sum of len(json.dumps(row)) + 2 per row, plus 2 for brackets.
    rows: list[dict] = list(authored_rows)

    # Compute the running size after authored rows.
    # We will compute this accurately later — for now just start noise generation.
    # Noise tweet ids: start at 2_000_000_000 to be disjoint from authored.
    noise_id_counter = 2_000_000_000

    # Compute current approximate byte total from authored content.
    # Use the exact serialization approach matching _write_tweets_js:
    # content = "window.YTD.tweets.part0 = " + json.dumps(rows)
    # We track the inner array size incrementally.
    prefix = "window.YTD.tweets.part0 = "
    # Inner array: "[" + items joined by ", " + "]"
    # Item serialization cost: len(json.dumps(item))
    # Separator cost: 2 bytes for ", " between consecutive items (n-1 separators)
    # Brackets: 2 bytes
    running_content_bytes = len(prefix) + 2  # prefix + "[]"
    if rows:
        # First item has no leading ", "
        running_content_bytes += len(json.dumps(rows[0]))
        for row in rows[1:]:
            running_content_bytes += 2 + len(json.dumps(row))

    # Noise rows: alternate between retweets (dropped_reasons["retweet"]) and
    # replies-to-others (dropped_reasons["reply_to_other"]).
    # Both types are dropped by the parser, keeping authored count fixed.
    noise_counter = 0
    while running_content_bytes < target_bytes:
        noise_id = str(noise_id_counter)
        noise_id_counter += 1
        noise_date = _fmt_date(rng)

        if noise_counter % 2 == 0:
            # Retweet — dropped because full_text.startswith("RT @")
            text = f"RT @{_NOISE_HANDLE}: Some retweeted content about topic #{noise_counter}."
            row = _make_tweet(noise_id, text, noise_date)
        else:
            # Reply-to-other — dropped because in_reply_to_user_id_str != own_id
            text = f"Reply #{noise_counter}: Responding to someone about an interesting topic here."
            row = _make_tweet(
                noise_id, text, noise_date,
                in_reply_to_status_id_str=str(noise_id_counter - 100),
                in_reply_to_user_id_str=_NOISE_USER_ID,
                in_reply_to_screen_name=_NOISE_HANDLE,
            )

        rows.append(row)
        # Each appended row adds ", " + row_bytes to the running count.
        running_content_bytes += 2 + len(json.dumps(row))
        noise_counter += 1

    _write_tweets_js(data_dir, rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: python -m eval.stress_fixture --mode throughput --n 200 --out /tmp/x"""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Twitter archive for stress testing."
    )
    parser.add_argument("--mode", required=True, choices=["throughput", "memory"],
                        help="throughput: n distinct authored tweets; memory: fixed count + noise")
    parser.add_argument("--n", type=int, default=10_000,
                        help="number of authored tweets (throughput) or fixed authored count (memory)")
    parser.add_argument("--size-mb", type=int, default=None,
                        help="target file size in MB (memory mode only)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for deterministic output")
    parser.add_argument("--out", required=True,
                        help="output directory (will contain data/)")

    args = parser.parse_args(argv)
    root = generate(
        Path(args.out),
        mode=args.mode,
        n=args.n,
        size_mb=args.size_mb,
        seed=args.seed,
    )
    print(f"Generated archive at: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
