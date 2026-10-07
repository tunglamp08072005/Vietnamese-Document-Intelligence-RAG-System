import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


class VectorStore:
    def __init__(self, path: str, collection: str, url: str | None = None, api_key: str | None = None) -> None:
        self.collection = collection
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)

    def ensure_collection(self, vector_size: int = 1024) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def upsert(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=chunk["vector_id"],
                vector=vector,
                payload={
                    "document_id": chunk["document_id"],
                    "chunk_id": chunk["id"],
                    "chunk_index": chunk["chunk_index"],
                    "filename": chunk["filename"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "content": chunk["content"],
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(
        self,
        vector: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if document_ids:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))]
            )
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            {**(point.payload or {}), "dense_score": float(point.score)}
            for point in result.points
        ]

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                )
            ),
            wait=True,
        )

    def close(self) -> None:
        self.client.close()


def make_vector_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.UUID(document_id), str(chunk_index)))
