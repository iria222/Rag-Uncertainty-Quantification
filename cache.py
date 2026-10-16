"""
This module provides functions for managing the RAG cache, 
including calculating a cache key, loading cached data, and saving new cache data.
"""

from config import CACHE_VERSION, CHUNK_SIZE, CHUNK_OVERLAP, CACHE_DIR, EMB_MODEL_NAME, MANIFEST_DIR, CHUNK_DIR, INDEX_DIR
from pathlib import Path
import faiss
import json

def calculate_cache_key(pdf_paths: list[Path]) -> str:
    """Hash the corpus and indexing settings to identify a valid cache."""

    import hashlib
    import json

    pdf_fingerprints = []
    for pdf_path in pdf_paths:
        file_stat = pdf_path.stat()
        pdf_fingerprints.append(
            {
                "name": pdf_path.name,
                "size": file_stat.st_size,
                "modified": file_stat.st_mtime_ns,
            }
        )

    cache_definition = {
        "version": CACHE_VERSION,
        "model": EMB_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "pdfs": pdf_fingerprints,
    }
    serialized_definition = json.dumps(
        cache_definition,
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(
        serialized_definition.encode("utf-8")
    ).hexdigest()

def load_rag_cache(cache_key: str):
    """Load cached chunks and their FAISS index when the key is valid."""


    required_paths = [MANIFEST_DIR, CHUNK_DIR, INDEX_DIR]
    if not all(cache_path.is_file() for cache_path in required_paths):
        return None

    try:
        manifest = json.loads(MANIFEST_DIR.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != cache_key:
            return None

        cached_chunks = json.loads(CHUNK_DIR.read_text(encoding="utf-8"))
        cached_index = faiss.read_index(str(INDEX_DIR))
        if (
            not isinstance(cached_chunks, list)
            or cached_index.ntotal != len(cached_chunks)
        ):
            return None
        return cached_chunks, cached_index
    except Exception:
        # A missing, incomplete, or corrupt cache is rebuilt automatically.
        return None

def save_rag_cache(cache_key: str, chunks: list[dict], vector_index) -> None:
    """Persist chunks, the FAISS index, and cache metadata atomically."""


    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    temporary_chunks_path = CHUNK_DIR.with_suffix(".json.tmp")
    temporary_index_path = INDEX_DIR.with_suffix(".faiss.tmp")
    temporary_manifest_path = MANIFEST_DIR.with_suffix(".json.tmp")

    temporary_chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_chunks_path.replace(CHUNK_DIR)

    faiss.write_index(vector_index, str(temporary_index_path))
    temporary_index_path.replace(INDEX_DIR)

    temporary_manifest_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "documents": len({chunk["source"] for chunk in chunks}),
                "chunks": len(chunks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Writing the manifest last marks the cache as complete.
    temporary_manifest_path.replace(MANIFEST_DIR)