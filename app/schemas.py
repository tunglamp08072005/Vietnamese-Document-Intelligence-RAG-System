from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    id: str
    filename: str
    created_at: datetime
    chunk_count: int


class UploadOut(DocumentOut):
    already_indexed: bool = False


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = None


class IndexRequest(BaseModel):
    document_ids: list[str] | None = None


class IndexOut(BaseModel):
    indexed_documents: int
    indexed_chunks: int


class SourceOut(BaseModel):
    citation: int
    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    chunk_id: str
    score: float
    content: str


class QueryOut(BaseModel):
    question: str
    answer: str
    answer_mode: str
    sources: list[SourceOut]
