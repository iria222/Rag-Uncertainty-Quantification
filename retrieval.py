
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from cache import save_rag_cache
from config import DEVICE, EMB_MODEL_NAME, RERANK_MODEL_NAME
from utils import free_model, debug_print



def get_faiss_index(cache_key : str, chunks : list[dict], cached_vector_index : faiss.Index) -> faiss.Index:
    """Creates a FAISS index for the given chunks using the specified embedding model. 
    If a cached index is provided, it will be used instead of creating a new one.

    Args:
        cache_key (str): The key for caching the FAISS index.
        chunks (list[dict]): The list of chunks for which to create the FAISS index.
        cached_vector_index (faiss.Index): The cached FAISS index, if available.

    Returns:
        faiss.Index: The created FAISS index.
    """
    if cached_vector_index is not None:
        debug_print("Using cached FAISS index for the chunks")
        return cached_vector_index
    debug_print("Loading embedding model for FAISS index creation in {DEVICE.upper()}.")
    
    embedding_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)

    
    debug_print("Creating new FAISS index for the chunks.")

    chunk_vectors = embedding_model.encode(
        [chunk['text'] for chunk in chunks],
        batch_size = 64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    vector_index = faiss.IndexFlatIP(chunk_vectors.shape[1])
    vector_index.add(chunk_vectors)
    save_rag_cache(cache_key, chunks, vector_index)

    debug_print("FAISS index created.")
    
    debug_print("Freeing up memory used by the embedding model.")
    free_model(embedding_model)
    
    return vector_index

def retrieve_relevant_chunks(query: str, vector_index: faiss.Index, chunks: list[dict], top_k: int = 100) -> list[dict]:
    """Retrieve the most relevant chunks for a given query using the FAISS index.

    Args:
        query (str): The input query for which to retrieve relevant chunks.
        vector_index (faiss.Index): The FAISS index containing chunk embeddings.
        chunks (list[dict]): The list of chunks corresponding to the FAISS index.
        top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 100.

    Returns:
        list[dict]: A list of the most relevant chunks for the given query with their similarity scores.
    """
    if not query.strip():
        debug_print("Empty query provided. Returning an empty list of relevant chunks.")
        return []
    
    k = min(max(top_k,1), vector_index.ntotal)
    
    debug_print(f"Loading embedding model for query encoding in {DEVICE.upper()}.")
    embedding_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)
    query_vector = embedding_model.encode(
        [query], 
        normalize_embeddings=True, 
        show_progress_bar=True
    ).astype(np.float32)
    
    debug_print("Freeing up memory used by the embedding model after query encoding.")
    free_model(embedding_model)
    
    scores, positions = vector_index.search(query_vector, k)
    relevant_chunks = [ 
                       {**chunks[int(pos)], "score":float(score)}
                       for score, pos in zip(scores[0], positions[0], strict = True) 
                       if pos >= 0 
                ]

    return relevant_chunks


def rerank_chunks(query: str, relevant_chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank the retrieved chunks based on their relevance to the query using a cross-encoder model.
    
    Args:
        query (str): The input query for which to rerank the relevant chunks.
        relevant_chunks (list[dict]): The list of relevant chunks to be reranked.
        top_k (int, optional): The number of top reranked chunks to return. Defaults to 5.
    
    Returns:
        list[dict]: A list of the top reranked chunks with their rerank scores.
    """
    if not query.strip():
        debug_print("Empty query provided. Returning an empty list of reranked chunks.")
        return []
    
    pairs = [(query, chunk['text']) for chunk in relevant_chunks]
    
    debug_print(f"Loading reranking model in {DEVICE.upper()}.")
    rerank_model = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
    
    rerank_scores = rerank_model.predict(pairs, show_progress_bar=True)
    scored = [
        {**chunk, "rerank_score": float(score)} 
        for chunk, score in zip(relevant_chunks, rerank_scores, strict = True)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_reranked = scored[:top_k]
    
    debug_print("Freeing up memory used by the reranking model.")
    free_model(rerank_model)
    
    return top_reranked