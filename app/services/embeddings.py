from collections.abc import Sequence
import logging
from time import perf_counter

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 8) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            started_at = perf_counter()
            logger.info("Loading embedding model %s for the first time", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Loaded embedding model %s in %.1f seconds", self.model_name, perf_counter() - started_at)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
