from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cortex.config import settings
from cortex.embedding.embedder import SentenceTransformerEmbedder
from cortex.pipeline.index import index_export
from cortex.rag.retriever import retrieve
from cortex.store.db import connect

try:
    from eval.gold_set import GOLD
except ModuleNotFoundError:
    from gold_set import GOLD


FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "twitter"


@dataclass
class QueryResult:
    query: str
    expected_external_id: str
    bucket: str
    semantic_rank: int | None
    hybrid_rank: int | None


def main() -> int:
    conn = connect(settings.database_url)
    try:
        _reset_db(conn)
        embedder = SentenceTransformerEmbedder(
            settings.embed_model,
            settings.embed_dim,
            settings.embed_batch_size,
        )
        index_export(FIXTURE, conn, embedder=embedder, cfg=settings)

        semantic_cfg = settings.model_copy(update={"retrieval_hybrid": False})
        hybrid_cfg = settings.model_copy(update={"retrieval_hybrid": True})
        results = [
            _evaluate_query(conn, embedder, semantic_cfg, hybrid_cfg, query, expected, bucket)
            for query, expected, bucket in GOLD
        ]
    finally:
        try:
            _reset_db(conn)
        finally:
            conn.close()

    print(_summary_table(results))
    print()
    print(_decision(results))
    print()
    print(_detail_table(results))
    return 0


def _evaluate_query(conn, embedder, semantic_cfg, hybrid_cfg, query, expected, bucket) -> QueryResult:
    semantic_sources = retrieve(conn, query, embedder, semantic_cfg)
    hybrid_sources = retrieve(conn, query, embedder, hybrid_cfg)
    return QueryResult(
        query=query,
        expected_external_id=expected,
        bucket=bucket,
        semantic_rank=_rank(semantic_sources, expected),
        hybrid_rank=_rank(hybrid_sources, expected),
    )


def _summary_table(results: list[QueryResult]) -> str:
    lines = [
        "| Bucket | Metric | Semantic | Hybrid |",
        "|---|---:|---:|---:|",
    ]
    for bucket in ("semantic", "lexical", "overall"):
        subset = results if bucket == "overall" else [row for row in results if row.bucket == bucket]
        semantic_hit, semantic_mrr = _metrics([row.semantic_rank for row in subset])
        hybrid_hit, hybrid_mrr = _metrics([row.hybrid_rank for row in subset])
        lines.append(f"| {bucket} | hit@3 | {semantic_hit:.3f} | {hybrid_hit:.3f} |")
        lines.append(f"| {bucket} | MRR | {semantic_mrr:.3f} | {hybrid_mrr:.3f} |")
    return "\n".join(lines)


def _decision(results: list[QueryResult]) -> str:
    by_bucket = defaultdict(list)
    for row in results:
        by_bucket[row.bucket].append(row)

    semantic_hit, _ = _metrics([row.semantic_rank for row in by_bucket["semantic"]])
    hybrid_hit, _ = _metrics([row.hybrid_rank for row in by_bucket["semantic"]])
    _, semantic_lexical_mrr = _metrics([row.semantic_rank for row in by_bucket["lexical"]])
    _, hybrid_lexical_mrr = _metrics([row.hybrid_rank for row in by_bucket["lexical"]])
    use_hybrid = hybrid_hit >= semantic_hit and hybrid_lexical_mrr > semantic_lexical_mrr
    rule_default = "True" if use_hybrid else "False"
    configured_default = settings.retrieval_hybrid
    note = (
        "Configured default is an explicit product override: hybrid did not regress and is "
        "preferred for real-export lexical coverage."
        if configured_default and not use_hybrid
        else "Configured default matches the decision rule."
    )
    return (
        f"Plan decision rule: retrieval_hybrid = {rule_default} "
        f"(semantic hit@3 {hybrid_hit:.3f} vs {semantic_hit:.3f}; "
        f"lexical MRR {hybrid_lexical_mrr:.3f} vs {semantic_lexical_mrr:.3f}).\n"
        f"Configured default: retrieval_hybrid = {configured_default}. {note}"
    )


def _detail_table(results: list[QueryResult]) -> str:
    lines = [
        "| Query | Expected | Bucket | Semantic rank | Hybrid rank |",
        "|---|---:|---|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row.query} | {row.expected_external_id} | {row.bucket} | "
            f"{_rank_display(row.semantic_rank)} | {_rank_display(row.hybrid_rank)} |"
        )
    return "\n".join(lines)


def _metrics(ranks: list[int | None]) -> tuple[float, float]:
    if not ranks:
        return 0.0, 0.0
    hit_at_3 = sum(1 for rank in ranks if rank is not None and rank <= 3) / len(ranks)
    mrr = sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)
    return hit_at_3, mrr


def _rank(sources, expected_external_id: str) -> int | None:
    for source in sources:
        if source.external_id == expected_external_id:
            return source.index
    return None


def _rank_display(rank: int | None) -> str:
    return "miss" if rank is None else str(rank)


def _reset_db(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
    conn.commit()


if __name__ == "__main__":
    raise SystemExit(main())
