from abc import ABC, abstractmethod


class Embedder(ABC):
    model_id: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed PASSAGES (no query instruction). Normalized unit vectors, batched internally."""


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_id: str, dim: int, batch_size: int):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.dim = dim
        self._batch = batch_size
        self._model = SentenceTransformer(model_id)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=self._batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()
