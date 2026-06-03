from contextlib import contextmanager
from pathlib import Path

import cortex.pipeline.index as index_module
from cortex.config import settings


FIX = Path(__file__).parent / "fixtures" / "twitter"


class _FakeEmbedder:
    model_id = "fake-embedder"
    dim = 384

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]


class _TrackingConnection:
    def __init__(self):
        self.transaction_count = 0
        self.active_transaction = None

    @contextmanager
    def transaction(self):
        assert self.active_transaction is None
        self.transaction_count += 1
        self.active_transaction = self.transaction_count
        try:
            yield
        finally:
            self.active_transaction = None


def test_index_export_uses_one_transaction_for_all_writes(monkeypatch):
    conn = _TrackingConnection()
    write_transactions = []
    next_document_id = 0

    def get_existing_hashes(connection, platform, external_ids):
        assert connection.active_transaction == 1
        return {}

    def get_reusable_vectors(connection, content_hashes, embed_model):
        assert connection.active_transaction == 2
        return {}

    def upsert_document(connection, item, content_hash):
        nonlocal next_document_id
        write_transactions.append(connection.active_transaction)
        next_document_id += 1
        return next_document_id

    def replace_chunks(connection, document_id, chunk_rows):
        write_transactions.append(connection.active_transaction)
        return len(chunk_rows)

    monkeypatch.setattr(index_module, "get_existing_hashes", get_existing_hashes)
    monkeypatch.setattr(index_module, "get_reusable_vectors", get_reusable_vectors)
    monkeypatch.setattr(index_module, "upsert_document", upsert_document)
    monkeypatch.setattr(index_module, "replace_chunks", replace_chunks)

    report = index_module.index_export(FIX, conn, embedder=_FakeEmbedder(), cfg=settings)

    assert report.documents_new == 58
    assert report.chunks_inserted == 60
    assert conn.transaction_count == 3
    assert set(write_transactions) == {3}
