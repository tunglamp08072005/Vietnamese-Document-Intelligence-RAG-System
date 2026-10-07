import hashlib
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.database import Database
from app.schemas import DocumentOut, IndexOut, IndexRequest, QueryOut, QueryRequest, SourceOut, UploadOut
from app.services.documents import chunk_pages, extract_docx, extract_pdf
from app.services.embeddings import EmbeddingService
from app.services.generation import AnswerService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore, make_vector_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(application: FastAPI):
    database = Database(settings.database_path)
    database.initialize()
    vectors = VectorStore(
        path=settings.qdrant_path,
        collection=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    vectors.ensure_collection()
    embeddings = EmbeddingService(
        settings.embedding_model,
        settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    application.state.database = database
    application.state.vectors = vectors
    application.state.embeddings = embeddings
    application.state.retrieval = RetrievalService(
        database=database,
        vectors=vectors,
        embeddings=embeddings,
        candidate_count=settings.retrieval_candidate_count,
        rrf_k=settings.rrf_k,
        reranker_enabled=settings.reranker_enabled,
        reranker_model=settings.reranker_model,
    )
    application.state.answerer = AnswerService(
        settings.ollama_base_url,
        settings.ollama_model,
        settings.ollama_timeout_seconds,
    )
    yield
    vectors.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


def _upsert_in_batches(
    embeddings: EmbeddingService,
    vectors: VectorStore,
    chunks: list[dict],
    batch_size: int,
) -> None:
    batch_size = max(1, batch_size)
    started_at = perf_counter()
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embedding_started_at = perf_counter()
        encoded = embeddings.encode([chunk["content"] for chunk in batch])
        embedding_seconds = perf_counter() - embedding_started_at
        qdrant_started_at = perf_counter()
        vectors.upsert(batch, encoded)
        qdrant_seconds = perf_counter() - qdrant_started_at
        batch_number = start // batch_size + 1
        logger.info(
            "Indexed chunk batch %d/%d (%d chunks): embedding %.1fs, Qdrant %.1fs",
            batch_number,
            total_batches,
            len(batch),
            embedding_seconds,
            qdrant_seconds,
        )
    logger.info("Indexed %d chunks in %.1f seconds", len(chunks), perf_counter() - started_at)


def _document_out(row: dict, already_indexed: bool = False) -> UploadOut:
    return UploadOut(
        id=row["id"],
        filename=row["filename"],
        created_at=row["created_at"],
        chunk_count=row["chunk_count"],
        already_indexed=already_indexed,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadOut, status_code=201, tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> UploadOut:
    upload_started_at = perf_counter()
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ tài liệu PDF hoặc DOCX.")

    read_started_at = perf_counter()
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    read_seconds = perf_counter() - read_started_at
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Kích thước tối đa là {settings.max_upload_mb} MB.")
    if not content:
        raise HTTPException(status_code=400, detail="Tệp tải lên đang trống.")

    digest = hashlib.sha256(content).hexdigest()
    existing = app.state.database.find_document_by_hash(digest)
    if existing:
        return _document_out(existing, already_indexed=True)

    try:
        extraction_started_at = perf_counter()
        pages = await run_in_threadpool(extract_pdf if suffix == ".pdf" else extract_docx, content)
        extraction_seconds = perf_counter() - extraction_started_at
        chunking_started_at = perf_counter()
        text_chunks = await run_in_threadpool(
            chunk_pages, pages, settings.chunk_size_chars, settings.chunk_overlap_chars
        )
        chunking_seconds = perf_counter() - chunking_started_at
    except Exception as error:
        logger.info("Could not extract %s: %s", filename, error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not text_chunks:
        raise HTTPException(status_code=422, detail="Không tìm thấy nội dung để lập chỉ mục.")

    document_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    chunks = [
        {
            "id": f"{document_id}:{index}",
            "document_id": document_id,
            "chunk_index": index,
            "content": chunk.content,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "vector_id": make_vector_id(document_id, index),
            "filename": filename,
        }
        for index, chunk in enumerate(text_chunks)
    ]

    try:
        await run_in_threadpool(
            app.state.database.add_document,
            {"id": document_id, "filename": filename, "sha256": digest, "created_at": now},
            chunks,
        )
        await run_in_threadpool(
            _upsert_in_batches,
            app.state.embeddings,
            app.state.vectors,
            chunks,
            settings.index_batch_size,
        )
    except sqlite3.IntegrityError:
        existing = app.state.database.find_document_by_hash(digest)
        if existing:
            return _document_out(existing, already_indexed=True)
        raise HTTPException(status_code=409, detail="Tài liệu đã được lập chỉ mục.")
    except Exception as error:
        try:
            await run_in_threadpool(app.state.vectors.delete_document, document_id)
        except Exception:
            logger.exception("Failed to roll back vectors for %s", document_id)
        try:
            await run_in_threadpool(app.state.database.remove_document, document_id)
        except Exception:
            logger.exception("Failed to roll back document metadata for %s", document_id)
        logger.exception("Indexing failed for %s", filename)
        raise HTTPException(status_code=503, detail=f"Không thể lập chỉ mục tài liệu: {error}") from error

    logger.info(
        "Upload complete: file=%s bytes=%d pages=%d chunks=%d read=%.1fs extract=%.1fs "
        "chunk=%.1fs total=%.1fs",
        filename,
        len(content),
        len(pages),
        len(chunks),
        read_seconds,
        extraction_seconds,
        chunking_seconds,
        perf_counter() - upload_started_at,
    )
    return UploadOut(
        id=document_id,
        filename=filename,
        created_at=now,
        chunk_count=len(chunks),
    )


@app.post("/documents/index", response_model=IndexOut, tags=["documents"])
def reindex_documents(request: IndexRequest) -> IndexOut:
    chunks = app.state.database.list_chunks(request.document_ids)
    if request.document_ids:
        found = {chunk["document_id"] for chunk in chunks}
        missing = set(request.document_ids) - found
        if missing:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tài liệu: {', '.join(sorted(missing))}")
    if not chunks:
        return IndexOut(indexed_documents=0, indexed_chunks=0)
    try:
        _upsert_in_batches(app.state.embeddings, app.state.vectors, chunks, settings.index_batch_size)
    except Exception as error:
        logger.exception("Reindex failed")
        raise HTTPException(status_code=503, detail=f"Không thể lập chỉ mục lại: {error}") from error
    return IndexOut(indexed_documents=len({chunk["document_id"] for chunk in chunks}), indexed_chunks=len(chunks))


@app.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents() -> list[DocumentOut]:
    return [DocumentOut(**row) for row in app.state.database.list_documents()]


@app.delete("/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(document_id: str) -> None:
    document = app.state.database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    try:
        app.state.vectors.delete_document(document_id)
        app.state.database.remove_document(document_id)
    except Exception as error:
        logger.exception("Could not delete document %s", document_id)
        raise HTTPException(status_code=503, detail=f"Không thể xóa tài liệu: {error}") from error


@app.post("/query", response_model=QueryOut, tags=["query"])
async def query_documents(request: QueryRequest) -> QueryOut:
    if request.document_ids:
        existing_ids = {document["id"] for document in app.state.database.list_documents()}
        missing = set(request.document_ids) - existing_ids
        if missing:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tài liệu: {', '.join(sorted(missing))}")

    try:
        retrieved = await run_in_threadpool(
            app.state.retrieval.retrieve, request.question, request.top_k, request.document_ids
        )
        answer, mode = await app.state.answerer.answer(request.question, retrieved)
    except (httpx.HTTPError, RuntimeError) as error:
        logger.exception("Query pipeline failed")
        raise HTTPException(status_code=502, detail=f"Không thể hoàn tất câu trả lời: {error}") from error
    except Exception as error:
        logger.exception("Query pipeline failed")
        raise HTTPException(status_code=503, detail=f"Không thể tìm kiếm tài liệu: {error}") from error

    sources = [
        SourceOut(
            citation=index,
            document_id=result["document_id"],
            filename=result["filename"],
            page_start=result.get("page_start"),
            page_end=result.get("page_end"),
            chunk_id=result["id"],
            score=result["score"],
            content=result["content"],
        )
        for index, result in enumerate(retrieved, start=1)
    ]
    return QueryOut(question=request.question, answer=answer, answer_mode=mode, sources=sources)
