import re
from threading import Event
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.database import Database
from app.services.embeddings import EmbeddingService
from app.services.operations import raise_if_cancelled
from app.services.vector_store import VectorStore


_HEADING_FOCUS_PHRASES = (
    "phương pháp",
    "khái niệm",
    "định nghĩa",
    "ví dụ",
    "ưu điểm",
    "nhược điểm",
    "đánh giá",
    "ứng dụng",
    "quy định",
    "điều khoản",
    "mục tiêu",
    "kết quả",
    "nguyên nhân",
    "giải pháp",
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


class RetrievalService:
    def __init__(
        self,
        database: Database,
        vectors: VectorStore,
        embeddings: EmbeddingService,
        candidate_count: int = 40,
        rrf_k: int = 60,
        reranker_enabled: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        self.database = database
        self.vectors = vectors
        self.embeddings = embeddings
        self.candidate_count = candidate_count
        self.rrf_k = rrf_k
        self.reranker_enabled = reranker_enabled
        self.reranker_model = reranker_model
        self._reranker: CrossEncoder | None = None

    def retrieve(
        self,
        question: str,
        top_k: int,
        document_ids: list[str] | None = None,
        cancel_event: Event | None = None,
    ) -> list[dict[str, Any]]:
        raise_if_cancelled(cancel_event)
        chunks = self.database.list_chunks(document_ids)
        if not chunks:
            return []
        chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
        active_document_ids = sorted({chunk["document_id"] for chunk in chunks})

        query_terms = tokenize(question)
        lexical_ranked: list[dict[str, Any]] = []
        if query_terms:
            bm25 = BM25Okapi([tokenize(chunk["content"]) for chunk in chunks])
            scores = bm25.get_scores(query_terms)
            ranked_indices = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
            lexical_ranked = [
                {**chunks[index], "lexical_score": float(scores[index])}
                for index in ranked_indices[: self.candidate_count]
                if scores[index] > 0
            ]

        raise_if_cancelled(cancel_event)
        query_vector = self.embeddings.encode([question])[0]
        raise_if_cancelled(cancel_event)
        vector_results = self.vectors.search(query_vector, self.candidate_count, active_document_ids)
        raise_if_cancelled(cancel_event)
        # SQLite is the source of truth: ignore stale Qdrant points left by old or partial deletions.
        dense_ranked = [
            {**chunks_by_id[result["chunk_id"]], "dense_score": result["dense_score"]}
            for result in vector_results
            if result.get("chunk_id") in chunks_by_id
        ]
        fused: dict[str, dict[str, Any]] = {}
        for ranking in (dense_ranked, lexical_ranked):
            for rank, result in enumerate(ranking, start=1):
                chunk_id = result.get("chunk_id") or result.get("id")
                if not chunk_id:
                    continue
                if chunk_id not in fused:
                    fused[chunk_id] = {
                        "id": chunk_id,
                        "document_id": result["document_id"],
                        "filename": result["filename"],
                        "page_start": result.get("page_start"),
                        "page_end": result.get("page_end"),
                        "content": result["content"],
                        "rrf_score": 0.0,
                    }
                fused[chunk_id]["rrf_score"] += 1.0 / (self.rrf_k + rank)

        question_lower = question.casefold()
        focus_phrases = [phrase for phrase in _HEADING_FOCUS_PHRASES if phrase in question_lower]
        for candidate in fused.values():
            heading = re.sub(r"\s+", " ", candidate["content"][:120]).strip().casefold()
            if heading.startswith("trong phần này"):
                continue
            candidate["rrf_score"] += 0.008 * sum(phrase in heading for phrase in focus_phrases)

        candidates = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        if self.reranker_enabled and candidates:
            raise_if_cancelled(cancel_event)
            if self._reranker is None:
                self._reranker = CrossEncoder(self.reranker_model)
            pairs = [(question, item["content"]) for item in candidates]
            scores = self._reranker.predict(pairs)
            raise_if_cancelled(cancel_event)
            for item, score in zip(candidates, scores, strict=True):
                item["score"] = float(score)
            candidates.sort(key=lambda item: item["score"], reverse=True)
        else:
            for item in candidates:
                item["score"] = item["rrf_score"]
        return candidates[:top_k]
