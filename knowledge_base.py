import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console

from config import (
    CHROMA_DB_PATH, SQLITE_DB_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, TOP_K_RESULTS, DATA_DIR
)

console = Console()


def _file_hash(path: Path) -> str:
    """SHA256 hash of file contents for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class KnowledgeBase:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()
        self._init_chroma()

    # ── SQLite - stores raw OCR text and file metadata ─────────────────────
    def _init_sqlite(self):
        self.sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
        self.sqlite_conn.row_factory = sqlite3.Row
        self.sqlite_conn.executescript("""
            CREATE TABLE IF NOT EXISTS photos (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path    TEXT    NOT NULL UNIQUE,
                file_hash    TEXT    NOT NULL,
                file_name    TEXT    NOT NULL,
                ocr_text     TEXT    NOT NULL DEFAULT '',
                tokens_used  INTEGER NOT NULL DEFAULT 0,
                has_text     INTEGER NOT NULL DEFAULT 0,
                error        TEXT,
                processed_at TEXT    NOT NULL,
                indexed_at   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_photos_hash     ON photos(file_hash);
            CREATE INDEX IF NOT EXISTS idx_photos_has_text ON photos(has_text);
        """)
        self.sqlite_conn.commit()

    # ── ChromaDB - vector index for semantic search ────────────────────────
    def _init_chroma(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write operations ───────────────────────────────────────────────────
    def save_ocr_result(
        self,
        file_path: Path,
        ocr_text: str,
        tokens_used: int,
        error: str | None,
    ) -> int:
        """
        Store OCR result in SQLite. Returns the row ID.
        Uses upsert so re-running is always safe.
        """
        file_hash = _file_hash(file_path)
        has_text = 1 if ocr_text.strip() else 0
        now = datetime.utcnow().isoformat()

        cursor = self.sqlite_conn.execute(
            """
            INSERT INTO photos
                (file_path, file_hash, file_name, ocr_text, tokens_used,
                 has_text, error, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                ocr_text     = excluded.ocr_text,
                tokens_used  = excluded.tokens_used,
                has_text     = excluded.has_text,
                error        = excluded.error,
                processed_at = excluded.processed_at
            """,
            (
                str(file_path), file_hash, file_path.name,
                ocr_text, tokens_used, has_text, error, now,
            ),
        )
        self.sqlite_conn.commit()
        return cursor.lastrowid

    def index_in_chroma(self, row_id: int, file_path: Path, ocr_text: str):
        """Add or update a document in ChromaDB. Only call for photos with text."""
        doc_id = f"photo_{row_id}"
        self.collection.upsert(
            ids=[doc_id],
            documents=[ocr_text],
            metadatas=[{
                "file_path": str(file_path),
                "file_name": file_path.name,
                "row_id": row_id,
            }],
        )
        self.sqlite_conn.execute(
            "UPDATE photos SET indexed_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), row_id),
        )
        self.sqlite_conn.commit()

    # ── Query operations ───────────────────────────────────────────────────
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Semantic search over the knowledge base.
        Returns list of result dicts sorted by relevance.
        """
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "file_name": meta["file_name"],
                "file_path": meta["file_path"],
                "similarity": round(1 - dist, 4),  # cosine distance → similarity
            })
        return hits

    def keyword_search(self, query: str, top_k: int = 20) -> list[dict]:
        """SQLite LIKE search - useful for exact character/number matches."""
        cursor = self.sqlite_conn.execute(
            """
            SELECT id, file_name, file_path, ocr_text
            FROM photos
            WHERE has_text = 1 AND ocr_text LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", top_k),
        )
        return [
            {
                "text": row["ocr_text"],
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "similarity": None,
            }
            for row in cursor.fetchall()
        ]

    def get_all_texts(self) -> list[dict]:
        """Retrieve all OCR texts (for summarization)."""
        cursor = self.sqlite_conn.execute(
            "SELECT file_name, ocr_text FROM photos WHERE has_text = 1 ORDER BY file_name"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Return processing statistics."""
        cursor = self.sqlite_conn.execute("""
            SELECT
                COUNT(*)                                          AS total,
                SUM(has_text)                                     AS with_text,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                SUM(tokens_used)                                  AS total_tokens,
                SUM(CASE WHEN indexed_at IS NOT NULL THEN 1 ELSE 0 END) AS indexed
            FROM photos
        """)
        return dict(cursor.fetchone())

    def is_already_processed(self, file_path: Path) -> bool:
        """Check if a file has been successfully processed (for resume support)."""
        cursor = self.sqlite_conn.execute(
            "SELECT id FROM photos WHERE file_path = ? AND error IS NULL",
            (str(file_path),),
        )
        return cursor.fetchone() is not None

    def get_photo_by_name(self, name: str) -> dict | None:
        """Fetch a photo record by file name or path."""
        cursor = self.sqlite_conn.execute(
            "SELECT * FROM photos WHERE file_path = ? OR file_name = ?",
            (name, Path(name).name),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def close(self):
        self.sqlite_conn.close()
