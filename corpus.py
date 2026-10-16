"""PDF text extraction and chunking pipeline for RAG corpus generation.

This module processes PDF documents by extracting their raw text and applying
size-based chunking strategies.
"""
import faiss

from cache import calculate_cache_key, load_rag_cache
from config import CHUNK_SIZE, CHUNK_OVERLAP, WORKERS
from config import DATA_DIR
from pathlib import Path
from pypdf import PdfReader
from joblib import Parallel, delayed
from utils import debug_print

def discover_pdf_paths() -> list[Path]:
    """Find all PDF files available in the configured data directory.
    
    Returns:
        list[Path]: List of PDF file paths."""
    
    data_directory = Path(DATA_DIR)
    if not data_directory.is_dir():
        return []
    return sorted(
        file_path
        for file_path in data_directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".pdf"
    )

# =========================
# PDF TEXT EXTRACTION
# =========================

def get_document_names(pdf_paths:list[Path]) -> list[dict]:
    """Return a list of document names from the given PDF paths.
    
    Args:
        pdf_paths (list[Path]): List of PDF file paths.
    Returns:
        list[dict]: List of dictionaries containing the document names."""
    
    return [{"source": pdf_path.name} for pdf_path in pdf_paths]

def extract_page_text(page) -> str:
    """Extract text from a single PDF page object.
    
    Args:
        page: A PDF page object.
    Returns:
        str: The extracted text from the page."""
    
    plain_text = page.extract_text() or ""

    if not plain_text.strip():
        # If the page has no extractable text, return an empty string.
        return ""

    character_count = len(plain_text.strip())
    space_ratio = plain_text.count(" ") / max(character_count, 1)

    # Very little whitespace may indicate incorrect concatenation of text blocks
    if character_count > 200 and space_ratio < 0.02:
        layout_text = (
            page.extract_text(
                extraction_mode = "layout",
                layout_mode_space_vertically = False
            )
            or ""
        )
        # If the layout extraction yields more text than the plain extraction, return it.
        if len(layout_text.strip()) > len(plain_text.strip()):
            return layout_text.strip()

    return plain_text.strip()

def extract_pdf_document(pdf_path: Path) -> dict:
    """Extract the non-empty text pages and source name from one PDF document.
    
    Args:
        pdf_path (Path): Path to the PDF file to process.

    Returns:
        dict: Dictionary containing the pages of the PDF with their text and its source name.
    """

    try:
        pdf_reader = PdfReader(pdf_path)
        pages = []
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = extract_page_text(page)
            if page_text and page_text.strip():
                pages.append({"page": page_number, "text": page_text.strip()})
    
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF '{pdf_path}': {e}")

    if not pages:
        raise ValueError(f"No extractable text found in PDF '{pdf_path}'. "
                         "Run OCR on the document or check if it contains text.")
    
    return {"source": pdf_path.name, "pages": pages}


def load_documents(pdf_paths: list[Path]) -> list[dict]:
    """Load PDF documents in parallel and extract their text content.
    
    Args:
        pdf_paths (list[Path]): List of PDF file paths to process.
        
    Returns:
        list[dict]: List of dictionaries containing the extracted text and source name for each PDF document.
    """

    if not pdf_paths:
        return []

    worker_count = min(WORKERS, len(pdf_paths))
    if worker_count == 1:
        return [extract_pdf_document(pdf_paths[0])]

    debug_print(f"Loading {len(pdf_paths)} PDF documents using {worker_count} workers...")

    return Parallel(
        n_jobs=worker_count,
        backend="loky"
    )(
        delayed(extract_pdf_document)(pdf_path) 
        for pdf_path in pdf_paths
    )


# =========================
# TEXT CHUNKING
# =========================

def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """ Split text into overlapping chunks of specified size and overlap. 
    
    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int): The maximum number of words in each chunk.
        chunk_overlap (int): The number of words to overlap between consecutive chunks.
        
    Returns:
        list[str]: A list of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")

    words = text.split()
    step = chunk_size - chunk_overlap
    chunks = []
    
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks

def build_chunks(documents: list[dict]) -> list[dict]:
    """Build text chunks from a list of documents, preserving source and page information.
    
    Args:
        documents (list[dict]): List of documents, each containing 'source' and 'pages
        information.
        
    Returns:
        list[dict]: A list of dictionaries, each representing a text chunk with its source and
        page information."""
    
    chunks = []
    for document_id, document in enumerate(documents):
        document_pages = document.get("pages") or [
            {
                "page": document.get("page"),
                "text": document.get("text")
            }
        ]
        chunk_id = 0
        for document_page in document_pages:
            for chunk_text in split_text(document_page["text"]):
                chunks.append(
                    {
                        "id": f"doc-{document_id}-chunk-{chunk_id}",
                        "source": document["source"],
                        "page": document_page["page"],
                        "text": chunk_text,
                    }
                )
                chunk_id += 1
    return chunks



def prepare_corpus() -> tuple[str, list[dict], "faiss.Index"]:
    """Prepare the RAG corpus by discovering PDF files, extracting text, and creating chunks.
    
    Returns:
        tuple: A tuple containing the cache key, list of text chunks, and vector index.
    """
    pdf_paths = discover_pdf_paths()
    cache_key = calculate_cache_key(pdf_paths)
    cached_data = load_rag_cache(cache_key)

    if cached_data is not None:
        debug_print("Found cached data for the current set of PDF documents. " \
        "Using cached data.")

        chunks,cached_vector_index = cached_data
        documents = get_document_names(pdf_paths)

        debug_print(f"Loaded {len(documents)} documents and {len(chunks)} chunks from cache.")
    
    else:
        debug_print("No cached data found for the current set of PDF documents." \
        "Processing PDF documents and creating new cache.")
    
        documents = load_documents(pdf_paths)
        chunks = build_chunks(documents)
        cached_vector_index = None

        debug_print(f"Processed {len(documents)} documents and created {len(chunks)} chunks.")
        
    return cache_key,chunks,cached_vector_index