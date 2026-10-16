"""
This module provides functions for parsing, normalizing, and extracting numeric inline citations from text
and matching them against a reference list section at the end of the generated text.
"""

import re
import unicodedata


def format_source_location(source: str, page: int | None = None) -> str:
    """Format a document name and optional page as a readable location."""
    if page is None:
        return source
    return f"{source}, p. {page}"

def extract_numeric_inline_labels(answer_text: str) -> list[int]:
    """Extract unique numeric citation IDs (e.g., [1], [2]) from the text body"""
    
    if not answer_text:
        return[]
    
    body_text = re.split(r"(?:referencias|references)\s*:", answer_text, flags=re.IGNORECASE)[0]
    matches = re.findall(r"\[(\d+)\]", body_text)
    
    return list(dict.fromkeys(map(int,matches)))

def parse_reference_section(answer_text:str) -> dict[int, dict]:
    """Parse the reference section at the end of the text mapping IDs to source and page"""
    
    references = {}
    if not answer_text:
        return references
    
    ref_split = re.split(r"(?:referencias|references)\s*:", answer_text, flags=re.IGNORECASE)
    if len(ref_split) < 2:
        return references
    
    ref_text = ref_split[1]
    
    # Pattern to read lines like: [1] Archivo: doc.pdf, Pág: 3
    line_pattern = re.compile(
        r"\[(?P<id>\d+)\]\s*(?:[Aa]rchivo|[Ff]ile)\s*:\s*(?P<source>[^;,\n]+?)(?:[;,]\s*(?:[Pp]ág|[Pp]age|p\.?)\s*:?\s*(?P<page>\d+))?",
        flags=re.IGNORECASE
    )
    
    for match in line_pattern.finditer(ref_text):
        citation_id = int(match.group("id"))
        source = match.group("source").strip().replace("`","")
        page_str = match.group("page")
        page = int(page_str) if page_str else None
        
        references[citation_id] = {
            "source": source,
            "page": page
        }

    return references
    

def extract_inline_citation_references(answer_text: str) -> list[dict]:
    """Expand parsed numeric citations using the references section mapping."""
    
    refs_map = parse_reference_section(answer_text)
    numeric_labels = extract_numeric_inline_labels(answer_text)

    references = []
    seen_references = set()
    for label_id in numeric_labels:
        if label_id in refs_map:
            ref = refs_map[label_id]
            ref_key = (ref["source"], ref["page"])
            if ref_key not in seen_references:
                references.append({"source": ref["source"], "page": ref["page"]})
                seen_references.add(ref_key)
                
    return references

def strip_inline_citations_for_evaluation(answer_text: str) -> str:
    """Remove numeric citations and the references section before answer evaluation"""

    if not answer_text:
        return ""
        
    # Delete references section
    text_body = re.split(r"(?:referencias|references)\s*:", answer_text, flags=re.IGNORECASE)[0]
    
    # Delete reference numbers in text
    clean_text = re.sub(r"\[\d+\]", "", text_body)
    
    # Clean spaces and punctuation
    clean_text = re.sub(r"[ \t]+([,.;:!?])", r"\1", clean_text)
    clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)
    clean_text = re.sub(r"[ \t]*\n[ \t]*", "\n", clean_text)
    return re.sub(r"\n{3,}", "\n\n", clean_text).strip()

def _normalize_accent_insensitive_text(text: str) -> str:
    """Apply Unicode normalization and remove combining diacritics."""

    decomposed_text = unicodedata.normalize(
        "NFD",
        unicodedata.normalize("NFKC", text),
    )
    return "".join(
        character
        for character in decomposed_text
        if not unicodedata.combining(character)
    )

def normalize_citation_label(label: str) -> str:
    """Normalize citation variants so they can be compared reliably."""

    normalized_label = unicodedata.normalize("NFKC", label).strip()
    if normalized_label.startswith("[") and normalized_label.endswith("]"):
        normalized_label = normalized_label[1:-1].strip()
    normalized_label = normalized_label.replace("`", "")
    normalized_label = re.sub(
        r"^(?:file|archivo|ficheiro):\s*",
        "",
        normalized_label,
        flags=re.IGNORECASE,
    )
    normalized_label = _normalize_accent_insensitive_text(normalized_label)
    normalized_label = " ".join(normalized_label.casefold().split())
    return re.sub(
        r"(?:\s*[;,]\s*|\s+)"
        r"(?:paginas?|pags?\.?|pages?|pp?\.?)\s*:?\s*(\d+)\s*$",
        r", p. \1",
        normalized_label,
    )

def extract_recognized_citation_references(
    answer_text: str,
    available_references: list[str],
) -> list[str]:
    """Return cited references that exactly match retrieved evidence sources."""

    canonical_references = {
        normalize_citation_label(reference): reference
        for reference in available_references
    }
    recognized_references = []
    seen_references = set()
    for citation in extract_inline_citation_references(answer_text):
        normalized_label = normalize_citation_label(
            format_source_location(citation["source"], citation["page"])
        )
        reference = canonical_references.get(normalized_label)
        if reference is None or normalized_label in seen_references:
            continue
        recognized_references.append(reference)
        seen_references.add(normalized_label)
    return recognized_references

def is_citation_free_abstention(answer_text: str) -> bool:
    """Recognize only the exact citation-free abstentions allowed by the prompt."""

    if not answer_text:
        return False
    
    normalized_answer = _normalize_accent_insensitive_text(answer_text).casefold()
    
    return (
        "[abstention]" in normalized_answer or
        "no dispongo de suficiente información" in normalized_answer or
        "no tengo información suficiente" in normalized_answer
    )

def format_context_label(chunk: dict, index:int) -> str:
    """Build the source label added before a retrieved context chunk."""

    page = chunk.get("page")
    if page is None:
        return f"[{index}] Archivo: {chunk['source']}"
    return f"[{index}] Archivo: {chunk['source']}; Pág: {page}"

