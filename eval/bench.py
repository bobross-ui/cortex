"""Cortex Layer 4 bench harness — illustrative, not a benchmark; numbers are machine-relative.

Three experiments:
  1. Embed throughput + end-to-end ingest + throughput-path peak RSS delta (needs Postgres + model).
  2. Re-ingest idempotency proof (needs Postgres + model, runs after exp 1).
  3. Flat parse memory — parse-only in subprocess, no DB, no torch.

CLI:
  python -m eval.bench [--database-url URL] [--n 10000] [--mem-n 2000] [--sizes 5,25,50]
                       [--seed 0]

Hidden internal flag (do not call directly):
  python -m eval.bench --parse-rss <export_root>
    Runs parse-only with an RSS sampler and prints PEAK_MB=<float> to stdout.
    This branch imports NO torch and uses NO DB.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


# ---------------------------------------------------------------------------
# RSS sampler — runs inside both the main process (exp 1) and the subprocess (exp 3)
# ---------------------------------------------------------------------------

def peak_rss_mb_while(fn):
    """Run fn(), sampling process RSS every 10 ms. Returns (result, peak_rss_mb)."""
    import psutil
    proc = psutil.Process()
    peak = proc.memory_info().rss
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(0.01)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join()
    return result, peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _bench_db_url(settings_url: str) -> str:
    """Derive a bench DB URL from settings.database_url by replacing the db name with cortex_bench."""
    parts = urlsplit(settings_url)
    # path is like "/cortex"; replace last segment with "cortex_bench"
    old_path = parts.path.rstrip("/")
    # Find last "/" to replace trailing segment
    slash_idx = old_path.rfind("/")
    if slash_idx == -1:
        new_path = "/cortex_bench"
    else:
        new_path = old_path[: slash_idx + 1] + "cortex_bench"
    new_parts = parts._replace(path=new_path)
    return urlunsplit(new_parts)


def _reset_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
    conn.commit()


# ---------------------------------------------------------------------------
# Experiment 3 — subprocess entry point (parse-only, torch-free)
# ---------------------------------------------------------------------------

def _run_parse_rss_subprocess(export_root: str) -> None:
    """Hidden flag: parse the export under an RSS sampler, print PEAK_MB=<float>, exit.

    This branch must NOT import torch, psycopg, or any heavy dependency.
    """
    from pathlib import Path as _Path

    # Import cortex.ingestion only — no pipeline, no embedder
    from cortex.ingestion.twitter import TwitterParser

    root = _Path(export_root)

    def _parse():
        return list(TwitterParser().parse(root))

    _, peak_mb = peak_rss_mb_while(_parse)
    print(f"PEAK_MB={peak_mb:.2f}", flush=True)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Experiment 1 — embed throughput + end-to-end + RSS delta
# ---------------------------------------------------------------------------

def _run_exp1(conn, root: Path, settings, n: int):
    """Index a throughput fixture (empty DB) and return (report, delta_mb)."""
    import gc
    import psutil

    # Imports that require the model/pipeline only inside this function
    from cortex.embedding.embedder import SentenceTransformerEmbedder
    from cortex.pipeline.index import index_export

    # Construct embedder first so torch loads now; then baseline is post-load
    embedder = SentenceTransformerEmbedder(
        settings.embed_model,
        settings.embed_dim,
        settings.embed_batch_size,
    )

    # Discard the fixture-generation transient garbage before the baseline snapshot
    gc.collect()
    baseline_bytes = psutil.Process().memory_info().rss

    def _ingest():
        return index_export(root, conn, embedder=embedder, cfg=settings)

    report, peak_mb = peak_rss_mb_while(_ingest)
    baseline_mb = baseline_bytes / (1024 * 1024)
    delta_mb = peak_mb - baseline_mb
    return report, delta_mb, embedder


# ---------------------------------------------------------------------------
# Experiment 2 — re-ingest idempotency
# ---------------------------------------------------------------------------

def _run_exp2(conn, root: Path, settings, embedder, total_docs: int):
    """Re-index the same export. Asserts all docs are skipped, nothing embedded."""
    from cortex.pipeline.index import index_export

    report = index_export(root, conn, embedder=embedder, cfg=settings)

    assert report.documents_unchanged_skipped == total_docs, (
        f"exp2: expected all {total_docs} docs unchanged_skipped, "
        f"got {report.documents_unchanged_skipped}"
    )
    assert report.chunks_embedded == 0, (
        f"exp2: expected 0 chunks_embedded, got {report.chunks_embedded}"
    )
    assert report.chunks_reused_cross_run == 0, (
        f"exp2: expected 0 chunks_reused_cross_run, got {report.chunks_reused_cross_run}"
    )
    assert report.embed_batches == 0, (
        f"exp2: expected 0 embed_batches, got {report.embed_batches}"
    )
    return report


# ---------------------------------------------------------------------------
# Experiment 3 — flat parse memory across file sizes
# ---------------------------------------------------------------------------

def _run_exp3(sizes: list[int], mem_n: int, seed: int) -> list[tuple[int, float]]:
    """For each size_mb, generate a memory fixture and measure peak parse RSS in a subprocess."""
    from eval.stress_fixture import generate

    results: list[tuple[int, float]] = []
    for size_mb in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            root = generate(
                Path(tmp) / f"mem_{size_mb}",
                mode="memory",
                n=mem_n,
                size_mb=size_mb,
                seed=seed,
            )
            peak_mb = _subprocess_parse_rss(root)
            results.append((size_mb, peak_mb))
    return results


def _subprocess_parse_rss(root: Path) -> float:
    """Run bench.py --parse-rss <root> in a subprocess; parse and return PEAK_MB."""
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "eval.bench", "--parse-rss", str(root)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"parse-rss subprocess failed (rc={result.returncode}):\n"
            f"  stdout: {result.stdout!r}\n"
            f"  stderr: {result.stderr!r}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("PEAK_MB="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(
        f"parse-rss subprocess did not print PEAK_MB=...\n"
        f"  stdout: {result.stdout!r}\n  stderr: {result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _machine_line() -> str:
    proc = platform.processor() or platform.machine() or "unknown CPU"
    cpus = os.cpu_count() or "?"
    mem_gb = ""
    try:
        import psutil
        mem_gb = f", {psutil.virtual_memory().total / (1024**3):.0f} GB RAM"
    except Exception:
        pass
    return f"{proc}, {cpus} logical CPUs{mem_gb}"


def _print_results(
    *,
    n: int,
    exp1_report,
    delta_mb: float,
    exp2_report,
    exp3_results: list[tuple[int, float]],
) -> None:
    e = exp1_report  # convenience alias

    embed_thr = e.chunks_embedded / e.embed_duration_s if e.embed_duration_s > 0 else 0.0
    e2e_thr = e.chunks_per_s()

    parse_rss_cells = " / ".join(f"{mb:.1f}" for _, mb in exp3_results)
    parse_rss_label = " / ".join(str(s) for s, _ in exp3_results)

    # Idempotency row
    idempotency = (
        f"{exp2_report.documents_unchanged_skipped} docs skipped, "
        f"{exp2_report.chunks_embedded} embedded, "
        f"{exp2_report.embed_batches} batches"
    )

    machine = _machine_line()

    print()
    print("## Layer 4 — Measured Numbers")
    print("_Illustrative, not a benchmark; numbers are machine-relative._")
    print()
    print("| Measurement | Value | Notes |")
    print("|---|---|---|")
    print(f"| Embed throughput (chunks/s) | {embed_thr:.0f} | "
          f"CPU, bge-small-en-v1.5, batch 64 |")
    print(f"| End-to-end ingest (~{n // 1000}k chunks) (s) | {e.duration_s:.1f} | "
          f"parse + chunk + embed + write; {e2e_thr:.0f} chunks/s |")
    print(f"| Throughput-path peak RSS delta (MB) | {delta_mb:.0f} | "
          f"delta over post-model-load baseline; the list[float] vector cost |")
    print(f"| Re-ingest same export | {idempotency} | "
          f"document-level incremental skip |")
    print(f"| Peak parse RSS @ {parse_rss_label} MB (MB) | {parse_rss_cells} | "
          f"flat = streaming drops noise |")
    print(f"| Machine | {machine} | numbers are illustrative |")
    print()
    print("Detail:")
    print(f"  chunks_embedded={e.chunks_embedded}  embed_duration_s={e.embed_duration_s:.2f}s"
          f"  embed_batches={e.embed_batches}")
    print(f"  chunks_inserted={e.chunks_inserted}  duration_s={e.duration_s:.2f}s")
    print(f"  documents_new={e.documents_new}  documents_unchanged_skipped={e.documents_unchanged_skipped}")


def _print_exp3_only(exp3_results: list[tuple[int, float]]) -> None:
    print()
    print("## Experiment 3 — Flat parse memory")
    print("_Illustrative, not a benchmark; numbers are machine-relative._")
    print()
    print("| File size (MB) | Peak parse RSS (MB) |")
    print("|---|---|")
    for size_mb, peak_mb in exp3_results:
        print(f"| {size_mb} | {peak_mb:.1f} |")
    print()
    print("Headline: peak RSS should be roughly flat as file size grows "
          "(noise is streamed and dropped by the parser).")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cortex Layer 4 bench — illustrative, not a benchmark.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "Bench DB URL. Defaults to a SEPARATE db (cortex_bench) derived from "
            "settings.database_url. NEVER the demo DB."
        ),
    )
    parser.add_argument("--n", type=int, default=10_000,
                        help="Number of authored tweets for the throughput fixture (default 10000).")
    parser.add_argument("--mem-n", type=int, default=2_000,
                        help="Fixed authored tweet count for memory-mode fixtures (default 2000).")
    parser.add_argument("--sizes", default="5,25,50",
                        help="Comma-separated list of file sizes in MB for exp 3 (default 5,25,50).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for fixture generation (default 0).")

    # Hidden internal flag — subprocess parse-only path
    parser.add_argument("--parse-rss", default=None, metavar="EXPORT_ROOT",
                        help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    # ---- Hidden subprocess branch (torch-free, no DB) ----
    if args.parse_rss is not None:
        _run_parse_rss_subprocess(args.parse_rss)
        return 0  # unreachable; _run_parse_rss_subprocess calls sys.exit

    # ---- Parse sizes ----
    try:
        sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    except ValueError as exc:
        print(f"error: --sizes must be comma-separated integers: {exc}", file=sys.stderr)
        return 1

    # ---- Determine bench DB URL ----
    from cortex.config import settings as _settings

    raw_bench_url = (
        os.environ.get("BENCH_DATABASE_URL")
        or args.database_url
        or _bench_db_url(_settings.database_url)
    )

    # Safety check: never truncate the demo DB
    if raw_bench_url == _settings.database_url:
        print(
            "error: bench DB URL matches settings.database_url (the demo DB). "
            "Pass --database-url to use a separate database.",
            file=sys.stderr,
        )
        return 1

    print(f"Bench DB: {raw_bench_url}")
    print()

    # ---- Experiments 1 & 2 (need Postgres + model) ----
    exp1_report = None
    exp2_report = None
    delta_mb = None
    embedder = None
    db_available = False

    try:
        from cortex.store.db import connect as _connect
        import psycopg

        conn = _connect(raw_bench_url)
        db_available = True
    except Exception as exc:
        print(
            f"Postgres not available at {raw_bench_url!r}: {exc}\n"
            f"Skipping experiments 1 & 2 (embed throughput + idempotency). "
            f"Running experiment 3 (parse memory) standalone.",
        )
        conn = None

    if db_available and conn is not None:
        from eval.stress_fixture import generate as _generate

        try:
            with tempfile.TemporaryDirectory() as tmp_throughput:
                root = _generate(
                    Path(tmp_throughput) / "throughput",
                    mode="throughput",
                    n=args.n,
                    seed=args.seed,
                )

                # Truncate so exp 1 starts from an empty store
                _reset_db(conn)

                print(f"[Exp 1] Indexing throughput fixture (n={args.n})...")
                try:
                    exp1_report, delta_mb, embedder = _run_exp1(
                        conn, root, _settings, args.n
                    )
                    print(f"  done: {exp1_report.chunks_inserted} chunks inserted, "
                          f"{exp1_report.chunks_embedded} embedded, "
                          f"duration={exp1_report.duration_s:.2f}s")

                    total_docs = exp1_report.documents_new
                    print(f"[Exp 2] Re-indexing (idempotency check, total_docs={total_docs})...")
                    exp2_report = _run_exp2(conn, root, _settings, embedder, total_docs)
                    print(f"  done: {exp2_report.documents_unchanged_skipped} unchanged_skipped, "
                          f"{exp2_report.chunks_embedded} embedded, "
                          f"{exp2_report.embed_batches} batches")
                finally:
                    try:
                        _reset_db(conn)
                    except Exception:
                        pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ---- Experiment 3 (no DB, no model) ----
    print(f"[Exp 3] Parse-memory experiment (mem_n={args.mem_n}, sizes={sizes})...")
    exp3_results = _run_exp3(sizes, args.mem_n, args.seed)
    for size_mb, peak_mb in exp3_results:
        print(f"  size={size_mb} MB -> peak parse RSS={peak_mb:.1f} MB")

    # ---- Print results ----
    if exp1_report is not None and exp2_report is not None and delta_mb is not None:
        _print_results(
            n=args.n,
            exp1_report=exp1_report,
            delta_mb=delta_mb,
            exp2_report=exp2_report,
            exp3_results=exp3_results,
        )
    else:
        print()
        print("(Experiments 1 & 2 skipped — no Postgres.)")
        _print_exp3_only(exp3_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
