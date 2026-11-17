
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag_uncertainty.utils.cache import save_rag_cache
from rag_uncertainty.config import DEVICE, EMB_MODEL_NAME, RERANK_MODEL_NAME
from rag_uncertainty.utils.utils import free_model, debug_print

_global_embedding_model = None
_global_reranking_model = None

def load_embedding_model():
    """Loads the embedding model in memory if its not loaded
    """
    global _global_embedding_model
    if _global_embedding_model is None:
        debug_print(f"Loading embedding model in {DEVICE.upper()}")
        _global_embedding_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)
    else:
        debug_print("Using resident embedding model from memory")
    return _global_embedding_model

def load_reranking_model():
    """Loads the reranking model in memory if its not loaded
    """
    global _global_reranking_model
    if _global_reranking_model is None:
        debug_print(f"Loading reranking model in {DEVICE.upper()}")
        _global_reranking_model = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
    else:
        debug_print("Using resident reranking model from memory")
    return _global_reranking_model

def free_retrieval_models():
    """Free embedding and reranking models from memory
    """
    global _global_reranking_model, _global_embedding_model
    
    if _global_embedding_model is not None:
        debug_print("Freeing embedding model from memory...")
        free_model(_global_embedding_model)
        _global_embedding_model = None
    
    if _global_reranking_model is not None:
        debug_print("Freeing reranking model from memory...")
        free_model(_global_reranking_model)
        _global_reranking_model = None

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
    
    embedding_model = load_embedding_model()
    
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
    
    return vector_index

def retrieve_and_rerank(query:str, 
                        vector_index: faiss.Index, 
                        chunks:list[dict], 
                        top_k_retrieved: int = 100, 
                        top_k_reranked: int = 5) -> tuple[list[dict], dict[str, list[float]]]:
    
    """Retrieve and rerank the most relevant chunks for a given query using the FAISS index and a cross-encoder model.
    
    Args:
        query (str): The input query for which to retrieve and rerank relevant chunks.
        vector_index (faiss.Index): The FAISS index containing chunk embeddings.
        chunks (list[dict]): The list of chunks corresponding to the FAISS index.
        top_k_retrieved (int, optional): The number of top relevant chunks to retrieve. Defaults to 100.
        top_k_reranked (int, optional): The number of top reranked chunks to return. Defaults to 5.
        
    Returns:
        tuple[list[dict], dict[str, list[float]]]: A tuple containing the list of top reranked chunks and a dictionary with the scores of all reranked chunks.
    """
    relevant_chunks = retrieve_relevant_chunks(query, vector_index, chunks, top_k=top_k_retrieved)
    all_reranked = rerank_chunks(query, relevant_chunks)
    
    score_lists ={
        "retrieval": [c["score"] for c in relevant_chunks],
        "rerank": [c["rerank_score"] for c in all_reranked],
    }
    
    return all_reranked[:top_k_reranked], score_lists
    

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
    
    embedding_model = load_embedding_model()
            
    query_vector = embedding_model.encode(
        [query], 
        normalize_embeddings=True, 
        show_progress_bar=True
    ).astype(np.float32)
    
    scores, positions = vector_index.search(query_vector, k)
    relevant_chunks = [ 
                       {**chunks[int(pos)], "score":float(score)}
                       for score, pos in zip(scores[0], positions[0], strict = True) 
                       if pos >= 0 
                ]

    return relevant_chunks


def rerank_chunks(query: str, relevant_chunks: list[dict]) -> list[dict]:
    """Rerank the retrieved chunks based on their relevance to the query using a cross-encoder model.
    
    Args:
        query (str): The input query for which to rerank the relevant chunks.
        relevant_chunks (list[dict]): The list of relevant chunks to be reranked.
    
    Returns:
        list[dict]: A list of the reranked chunks with their rerank scores.
    """
    if not query.strip():
        debug_print("Empty query provided. Returning an empty list of reranked chunks.")
        return []
    
    pairs = [(query, chunk['text']) for chunk in relevant_chunks]
    
    rerank_model = load_reranking_model()
            
    
    rerank_scores = rerank_model.predict(pairs, show_progress_bar=True)
    scored = [
        {**chunk, "rerank_score": float(score)} 
        for chunk, score in zip(relevant_chunks, rerank_scores, strict = True)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    
    return scored