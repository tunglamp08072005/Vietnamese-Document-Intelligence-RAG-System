import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class Database:
    def __init__(self, path: str) -> None:
        self.path = path

    def initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    vector_id TEXT NOT NULL UNIQUE,
                    UNIQUE(document_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def find_document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return dict(row) if row else None

    def add_document(self, document: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO documents(id, filename, sha256, created_at, chunk_count) VALUES (?, ?, ?, ?, ?)",
                (document["id"], document["filename"], document["sha256"], document["created_at"], len(chunks)),
            )
            connection.executemany(
                """INSERT INTO chunks(id, document_id, chunk_index, content, page_start, page_end, vector_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk["id"],
                        document["id"],
                        chunk["chunk_index"],
                        chunk["content"],
                        chunk["page_start"],
                        chunk["page_end"],
                        chunk["vector_id"],
                    )
                    for chunk in chunks
                ],
            )

    def remove_document(self, document_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        query = """SELECT c.id, c.document_id, c.chunk_index, c.content, c.page_start, c.page_end, c.vector_id,
                          d.filename
                   FROM chunks c JOIN documents d ON d.id = c.document_id"""
        parameters: list[str] = []
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            query += f" WHERE c.document_id IN ({placeholders})"
            parameters.extend(document_ids)
        query += " ORDER BY d.created_at, c.chunk_index"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]
