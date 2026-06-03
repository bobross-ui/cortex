from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cortex.store import repository


@dataclass
class Source:
    index: int
    document_id: int
    external_id: str
    platform: str
    content_type: str
    url: str | None
    author_handle: str | None
    created_at: datetime | None
    snippet: str
    context_text: str
    score: float
    cited: bool = False


def retrieve(conn, question: str, embedder, cfg, *, filters=None) -> list[Source]:
    """Embed a question, retrieve matching chunks, and return deduped citation sources."""
    query = f"{cfg.bge_query_instruction} {question}"
    query_embedding = embedder.embed([query])[0]
    filter_kwargs = _filter_kwargs(filters)
    if getattr(cfg, "retrieval_hybrid", False):
        hits = _hybrid_hits(conn, question, query_embedding, cfg, filter_kwargs)
    else:
        hits = repository.search(
            conn,
            query_embedding,
            k=cfg.retrieval_candidates,
            **filter_kwargs,
        )
    best_hits = group_by_document(hits)[: cfg.retrieval_k]
    docs = repository.fetch_documents(conn, [hit["document_id"] for hit in best_hits])

    sources = []
    for hit in best_hits:
        doc_id = hit["document_id"]
        doc = docs.get(doc_id)
        if doc is None:
            continue

        snippet = hit["text"]
        doc_text = doc["text"]
        context_text = (
            doc_text
            if len(doc_text) <= cfg.chat_context_char_cap
            else snippet
        )
        sources.append(
            Source(
                index=len(sources) + 1,
                document_id=doc_id,
                external_id=doc["external_id"],
                platform=doc["source_platform"],
                content_type=doc["content_type"],
                url=doc["url"],
                author_handle=doc["author_handle"],
                created_at=doc["created_at"],
                snippet=snippet,
                context_text=context_text,
                score=_hit_score(hit),
            )
        )

    return sources


def rrf_fuse(semantic_ids: list, lexical_ids: list, k: int = 60) -> list:
    """Fuse best-first rankings with Reciprocal Rank Fusion."""
    scores = _rrf_scores(semantic_ids, lexical_ids, k)
    return sorted(scores, key=lambda _id: scores[_id], reverse=True)


def group_by_document(hits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-scoring chunk per document and return documents in score order."""
    best_by_document: dict[int, tuple[float, int, dict[str, Any]]] = {}
    for position, hit in enumerate(hits):
        row = dict(hit)
        document_id = row["document_id"]
        score = _hit_score(row)
        row["score"] = score

        current = best_by_document.get(document_id)
        if current is None or score > current[0]:
            best_by_document[document_id] = (score, position, row)

    return [
        row
        for _, _, row in sorted(
            best_by_document.values(),
            key=lambda item: (-item[0], item[1]),
        )
    ]


def _filter_kwargs(filters) -> dict[str, str]:
    if filters is None:
        return {}
    if hasattr(filters, "model_dump"):
        filters = filters.model_dump(exclude_none=True)
    elif not isinstance(filters, Mapping):
        filters = vars(filters)

    return {
        key: filters[key]
        for key in ("source_platform", "content_type")
        if filters.get(key) is not None
    }


def _hybrid_hits(conn, question: str, query_embedding, cfg, filters: dict[str, str]) -> list[dict]:
    semantic_hits = repository.search(
        conn,
        query_embedding,
        k=cfg.rrf_candidates,
        **filters,
    )
    lexical_hits = repository.search_fts(
        conn,
        question,
        k=cfg.rrf_candidates,
        **filters,
    )
    semantic_ids = [hit["id"] for hit in semantic_hits]
    lexical_ids = [hit["id"] for hit in lexical_hits]
    fused_ids = rrf_fuse(semantic_ids, lexical_ids, cfg.rrf_k)
    fused_scores = _rrf_scores(semantic_ids, lexical_ids, cfg.rrf_k)
    hits_by_id = {hit["id"]: hit for hit in [*semantic_hits, *lexical_hits]}

    fused_hits = []
    for chunk_id in fused_ids:
        hit = dict(hits_by_id[chunk_id])
        hit["score"] = fused_scores[chunk_id]
        fused_hits.append(hit)
    return fused_hits


def _rrf_scores(semantic_ids: list, lexical_ids: list, k: int) -> dict[Any, float]:
    scores: dict[Any, float] = {}
    for ranking in (semantic_ids, lexical_ids):
        for rank, _id in enumerate(ranking, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
    return scores


def _hit_score(hit: Mapping[str, Any]) -> float:
    if hit.get("score") is not None:
        return float(hit["score"])
    if hit.get("rank") is not None:
        return float(hit["rank"])
    return 0.0
