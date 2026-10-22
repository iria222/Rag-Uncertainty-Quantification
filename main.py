""" Main script for the RAG pipeline."""

import json
import argparse
import os

from corpus import prepare_corpus

from retrieval import get_faiss_index, rerank_chunks, retrieve_relevant_chunks, free_retrieval_models
from generation import generate_answer, free_generation_model
from utils import debug_print

def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Path to the input JSON file containing the queries.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/results.json",
        help="Path to the output JSON file to save the results.",
    )
    return parser.parse_args()

def run_rag_pipeline(query: str, vector_index, chunks: list[dict], top_k_retrieved: int = 100, top_k_reranked: int = 5):
    """Runs the RAG pipeline for a given query, retrieving and reranking relevant chunks.
    
    Args:
        query (str): The input query for which to retrieve and rerank relevant chunks.
        vector_index (faiss.Index): The FAISS index containing chunk embeddings.
        chunks (list[dict]): The list of chunks corresponding to the FAISS index.
        top_k_retrieved (int, optional): The number of top relevant chunks to retrieve. Defaults to 100.
        top_k_reranked (int, optional): The number of top reranked chunks to return. Defaults to 5."""
        
    debug_print(f"Seaching relevant fragments for the query: '{query}'")
    relevant_chunks = retrieve_relevant_chunks(query, vector_index, chunks, top_k=top_k_retrieved)
    
    debug_print(f"Applying reranking to the {len(relevant_chunks)} retrieved chunks.")
    reranked_chunks = rerank_chunks(query, relevant_chunks, top_k=top_k_reranked)
    
    free_retrieval_models()
    
    generated_response, load_time, generation_time, reliability = generate_answer(query, reranked_chunks)
    
    reliability_signals = build_reliability_signals(generated_response, load_time, generation_time, reliability)
    
    return reranked_chunks, generated_response, reliability_signals
    

def main():
    args = parse_arguments()
    
    print("\n========================================")
    print("INICIALIZANDO SISTEMA RAG")
    print("========================================")

    # Chunking the corpus
    cache_key, chunks, cached_vector_index = prepare_corpus()

    # Preparing the FAISS index
    vector_index = get_faiss_index(cache_key, chunks, cached_vector_index)
        
    if args.input_json:
        if not os.path.exists(args.input_json):
            print(f"Input file doesn't exist: {args.input_json}")
            return 1
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Check writing permissions
                if not os.access(output_dir, os.W_OK):
                    print(f"Error: No write permissions for output directory: {output_dir}")
                    return 1
            except Exception as e:
                print(f"Error setting up output directory: {e}")
                return 1
                
        return batch_pipeline(args, chunks, vector_index)
    
    else:
        # Interactive mode: prompt the user for a single query
        interactive_pipeline(chunks, vector_index)

def batch_pipeline(args, chunks, vector_index):

    debug_print(f"Loading queries from: {args.input_json}")
    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            queries_to_process = json.load(f)
        debug_print(f"Loaded {len(queries_to_process)} queries from the JSON file")
    except Exception as e:
        debug_print(f"Error loading the JSON file: {e}")
        return
    
    debug_print(f"--- Executing retrieval and reranking for {len(queries_to_process)} queries ---")
    prepared_batch = []
        
    for query in queries_to_process:
        query_id = query.get("id", "unknown_id")
        question = query.get("question")
        if not question:
            debug_print(f"Empty query with ID: {query_id}. Omitted.")
            continue
            
        debug_print(f"--- Processing retrieval for query with ID: {query_id} ---")
        relevant_chunks = retrieve_relevant_chunks(question, vector_index, chunks)
        reranked_chunks = rerank_chunks(question, relevant_chunks)
        prepared_batch.append({
            "query": query,
            "reranked_chunks": reranked_chunks
        })
        
    free_retrieval_models()
    debug_print(f"--- Starting answer generation for {len(prepared_batch)} queries ---")
    results_export = []
    
    for item in prepared_batch:
        query = item["query"]
        reranked_chunks = item["reranked_chunks"]
        
        query_id = query.get("id", "unknown_id")
        question = query.get("question")
        
        debug_print(f"Generating answer for query with ID: {query_id}")
        generated_response, load_time, generation_time = generate_answer(question, reranked_chunks)
        
        reliability_signals = build_reliability_signals(generated_response, load_time, generation_time)
        
        results_export.append({
            "id": query_id,
            "category": query.get("category"),
            "question": question,
            "ground_truth": query.get("ground_truth"),
            "answerable": query.get("answerable"),
            "retrieved_chunks": reranked_chunks,
            "generated_response": generated_response,
            "reliability_signals": reliability_signals
        })
            
    if results_export:
        debug_print(f"Saving results in {args.output}...")
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results_export, f, ensure_ascii=False, indent=4)
            debug_print("Saved results successfully.")
        except Exception as e:
            debug_print(f"Error when saving the results: {e}")
    free_generation_model()

def build_reliability_signals(generated_response, load_time, generation_time):
    # TODO: Implement the calculation of reliability signals based on the reranked chunks and the generated response.
    reliability_signals = {
            "entropy": None,
            "confidence_score": None,
            "decision": "abstain" if "[ABSTENCION]" in generated_response else "answer" ,      # 'answer', 'abstain', or 'escalate'
            "model_load_time_sec": round(load_time, 2),
            "generation_time_sec": round(generation_time, 2)
        }
    return reliability_signals

def interactive_pipeline(chunks, vector_index):
    print(f"\n{'='*60}")
    print("\n--- MODO INTERACTIVO ---")
    print("Para salir de la aplicación escribe '/salir' o '/exit'")
        
    while True:
        try:
            user_input = input("\nEscribe tu pregunta > ").strip()
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo del sistema...")
            break
            
        if user_input.lower() in ["/salir","/exit"]:
            print("Cerrando el asistente RAG. ¡Hasta pronto!")
            break
            
        if not user_input:
            print("Por favor, introduce una pregunta válida.")
            continue
            
        retrieved_chunks, generated_response, _ = run_rag_pipeline(user_input, vector_index, chunks)
        free_generation_model()
            
        print(f"\n{'='*60}")
        print("RESULTADO DE LA CONSULTA:")
        print(f"Respuesta generada: {generated_response}")
        print("\nFragmentos más relevantes recuperados:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            score = chunk.get('rerank_score', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"{i}. [Score: {score_str}] - Fichero: {chunk.get('source', 'Desconocido')} (Pág. {chunk.get('page','N/A')})")
            print(f"     Extracto: {chunk['text'][:180]}...")
        print(f"\n{'='*60}")
            


if __name__ == "__main__":
    main()