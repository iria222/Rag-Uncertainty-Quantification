""" Main script for the RAG pipeline."""

import json
import argparse
import os

from rag_uncertainty.core.corpus import prepare_corpus
from rag_uncertainty.evaluation.retrieval_metrics import compute_retrieval_metrics
from rag_uncertainty.core.retrieval import get_faiss_index, free_retrieval_models, retrieve_and_rerank
from rag_uncertainty.core.generation import generate_answer, free_generation_model
from rag_uncertainty.utils.utils import debug_print
from rag_uncertainty.utils.checkpoint import load_checkpoint, get_processed_ids, save_checkpoint
from rag_uncertainty.config import CHECKPOINT_SAVE_FREQUENCY, OUTPUT_DIR, TOP_K_RERANKED, TOP_K_RETRIEVED, SHOW_METRICS

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
        default=str(OUTPUT_DIR / "results.json"),
        help="Path to the output JSON file to save the results.",
    )
    return parser.parse_args()

def run_rag_pipeline(query: str, 
                     vector_index, chunks: list[dict], 
                     top_k_retrieved: int = TOP_K_RETRIEVED, 
                     top_k_reranked: int = TOP_K_RERANKED) -> dict:
    """Runs the RAG pipeline for a given query, retrieving and reranking relevant chunks.
    
    Args:
        query (str): The input query for which to retrieve and rerank relevant chunks.
        vector_index (faiss.Index): The FAISS index containing chunk embeddings.
        chunks (list[dict]): The list of chunks corresponding to the FAISS index.
        top_k_retrieved (int, optional): The number of top relevant chunks to retrieve. Defaults to 100.
        top_k_reranked (int, optional): The number of top reranked chunks to return. Defaults to 5."""
        
    debug_print(f"Seaching relevant fragments for the query: '{query}'")
    reranked_chunks, score_lists = retrieve_and_rerank(query, vector_index, chunks, top_k_retrieved, top_k_reranked)
    
    free_retrieval_models()
    
    retrieval_assessment = assess_retrieval(score_lists)
    #TODO: if retrieval_assessment["decision"]["abstain"]: skip generation here
    
    generated_results = generate_answer(query, reranked_chunks)
    reliability_signals = build_reliability_signals(generated_results, retrieval_assessment)
        
    return {"reranked_chunks": reranked_chunks, 
            "answer": generated_results["answer"], 
            "references": generated_results["references"],
            "reliability_signals": reliability_signals,
            }
    

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
    """ Run the RAG pipeline in batch mode, processing queries from an input JSON file and saving results to an output JSON file.
    
    Args:
        args: Command-line arguments containing input and output paths.
        chunks (list[dict]): The list of chunks corresponding to the FAISS index.
        vector_index (faiss.Index): The FAISS index containing chunk embeddings.
    """

    debug_print(f"Loading queries from: {args.input_json}")
    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            queries_to_process = json.load(f)
            total_queries = len(queries_to_process)
        debug_print(f"Loaded {total_queries} queries from the JSON file")
    except Exception as e:
        debug_print(f"Error loading the JSON file: {e}")
        return
    
    input_filename = os.path.basename(args.input_json)
    input_stem = os.path.splitext(input_filename)[0]
    
    user_output_filename = os.path.basename(args.output)
    if not user_output_filename:
        user_output_filename = "results.json"
        
    output_path = os.path.join(OUTPUT_DIR, input_stem, user_output_filename)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
            
    
    retrieval_cache_path = os.path.join(output_dir, "retrieval_cache.json")
        
    debug_print(f"Output results will be saved to: {output_path}")
    
    args.output = output_path
    # Load previous state through checkpoint
    results_export = load_checkpoint(output_path)
    processed_ids = get_processed_ids(results_export)
    
    if processed_ids:
        debug_print(f"Checkpoint found: {len(processed_ids)} queries already processed. Resuming...")
        
    queries_pending = [
        q for q in queries_to_process
        if q.get("id", "unknown_id") not in processed_ids
    ]
    
    total_pending = len(queries_pending)
    
    if total_pending == 0:
        debug_print("All queries were already processed")
        return
    
    retrieval_cache = run_retrieval_phase(chunks, vector_index, retrieval_cache_path, queries_pending, total_pending)
    
    run_generation_phase(args, results_export, queries_pending, retrieval_cache)
    
    debug_print(f"Process ended successfully. All results saved in {output_path}.")

def run_generation_phase(args, results_export, queries_pending, retrieval_cache):
    """ Run the answer generation phase for the pending queries, using the retrieval cache to avoid redundant retrievals.
    
    Args:
        args: Command-line arguments containing output paths.
        results_export (list): The list of already processed results to which new results will be appended
        queries_pending (list): The list of queries that still need to be processed
        retrieval_cache (list): The cache of retrieved and reranked chunks for previously processed queries
    """
    retrieved_map = {item["query"]["id"]: item for item in retrieval_cache if "query" in item and "id" in item["query"]}
    prepared_batch = [
        retrieved_map[q.get("id", "unknown_id")]
        for q in queries_pending
        if q.get("id", "unknown_id") in retrieved_map
    ]
    
    total_prepared = len(prepared_batch)
    debug_print(f"--- Starting answer generation for {total_prepared} queries ---")
        
    changes_saved = True
    
    for idx, item in enumerate(prepared_batch, 1):
        query = item["query"]
        reranked_chunks = item["reranked_chunks"]
        retrieval_assessment = item["retrieval_assessment"]
        
        query_id = query.get("id", "unknown_id")
        question = query.get("question")
        
        debug_print(f"[{idx}/{total_prepared}] Generating answer for query with ID: {query_id}")
        generated_results = generate_answer(question, reranked_chunks)
        
        reliability_signals = build_reliability_signals(generated_results, retrieval_assessment)
        
        results_export.append({
            "id": query_id,
            "category": query.get("category"),
            "question": question,
            "ground_truth": query.get("ground_truth"),
            "answerable": query.get("answerable"),
            "retrieved_chunks": reranked_chunks,
            "empty_answer": generated_results["empty_answer"],
            "raw_response": generated_results["raw_answer"],
            "generated_response": generated_results["answer"],
            "references": generated_results["references"],
            "reliability_signals": reliability_signals
        })
        
        changes_saved = False
        # Save checkpoint
        if idx % CHECKPOINT_SAVE_FREQUENCY == 0:
            save_checkpoint(args.output, results_export)
            changes_saved = True
            debug_print(f"[CHECKPOINT] Checkpoint updated in disk (Progress: {idx}/{total_prepared})")
        
            
    if not changes_saved:
        save_checkpoint(args.output, results_export)
        debug_print("[CHECKPOINT] Final checkpoint saved in disk\n")
    free_generation_model()

def run_retrieval_phase(chunks, vector_index, retrieval_cache_path, queries_pending) -> list[dict]:
    """ Run the retrieval and reranking phase for the pending queries, using a checkpoint to avoid redundant retrievals.
    
    Args:
        chunks (list[dict]): The list of chunks corresponding to the FAISS index.
        vector_index (faiss.Index): The FAISS index containing chunk embeddings.
        retrieval_cache_path (str): The path to the retrieval cache file.
        queries_pending (list): The list of queries that still need to be processed.
    
    Returns:
        list[dict]: The updated retrieval cache containing retrieved and reranked chunks for all processed queries
    """
    retrieval_cache = load_checkpoint(retrieval_cache_path)
    retrieved_ids = {item["query"]["id"] for item in retrieval_cache if "query" in item and "id" in item["query"]}
    if retrieved_ids:
        debug_print(f"[CHEKPOINT] Checkpoint found (Retrieval): {len(retrieved_ids)} queries already retrieved")
    
    queries_needing_retrieval = [
        q for q in queries_pending
        if q.get("id", "unknown_id") not in retrieved_ids
    ]
    
    total_to_retrieved = len(queries_needing_retrieval)
    
    changes_saved = False
    if total_to_retrieved > 0:
        debug_print(f"--- Executing retrieval and reranking for {total_to_retrieved} queries ---")
            
        for idx, query in enumerate(queries_needing_retrieval,1):
            query_id = query.get("id", "unknown_id")
            question = query.get("question")
            if not question:
                debug_print(f"[{idx}/{total_to_retrieved}]Empty query with ID: {query_id}. Omitted.")
                continue
                
            debug_print(f"--- [{idx}/{total_to_retrieved}] Processing retrieval for query with ID: {query_id} ---")
            reranked_chunks, score_lists = retrieve_and_rerank(question, vector_index, chunks, TOP_K_RETRIEVED, TOP_K_RERANKED)
            
            retrieval_cache.append({
                "query": query,
                "reranked_chunks": reranked_chunks,
                "score_lists": score_lists,
            })
            
            changes_saved = False
            
            if idx % CHECKPOINT_SAVE_FREQUENCY == 0:
                save_checkpoint(retrieval_cache_path, retrieval_cache)
                changes_saved = True
                debug_print(f"[CHECKPOINT] Retrieval checkpoint updated in disk (Progress: {idx}/{total_to_retrieved})")
                
        
        if not changes_saved:
            save_checkpoint(retrieval_cache_path, retrieval_cache)
            debug_print(f"[CHECKPOINT] Final retrieval checkpoint updated in disk (Progress: {idx}/{total_to_retrieved})")
            
    else:
        debug_print("All pending queries already have their retrieval cached. Skipping retrieval phase...")
        
    for item in retrieval_cache:
        item["retrieval_assessment"] = assess_retrieval(item.get("score_lists"))
    free_retrieval_models()
    return retrieval_cache

def assess_retrieval(score_lists: dict | None) -> dict:
    """ Assess the retrieval quality based on the score lists, computing retrieval metrics.
    
    Args:
        score_lists (dict | None): The score lists obtained from retrieval and reranking.
    
    Returns:
        dict: A dictionary containing the computed retrieval metrics and any decisions based on them.
    """
    
    metrics = compute_retrieval_metrics(score_lists) if score_lists else None
    return{
        "metrics": metrics,
        #TODO: Decide to abstain or continue with generation based on retrieval metrics
        
    }

def build_reliability_signals(generated_results: dict, retrieval_assessment: dict) -> dict:
    p = generated_results["p_true"]
    reliability_signals = {
            "retrieval_metrics": retrieval_assessment["metrics"],
            "uncertainty_metrics": generated_results["uncertainty_metrics"],
            "p_true_score": round(p, 4) if p is not None else None,
            "mass_ab": round(generated_results["mass_ab"], 4) if generated_results["mass_ab"] is not None else None,
            "model_abstained": generated_results["model_abstained"],
            "self_eval_abstained": generated_results["self_eval_abstained"],
            "decision": "abstain" if "[ABSTENCION]" in generated_results["answer"] else "answer" ,      # 'answer', 'abstain', or 'escalate'
            "model_load_time_sec": round(generated_results["load_time"], 2),
            "generation_time_sec": round(generated_results["generation_time"], 2),
            "self_eval_time_sec": round(generated_results["eval_time"], 2)
        }
    return reliability_signals

def interactive_pipeline(chunks, vector_index):
    """ Run the RAG pipeline in interactive mode, prompting the user for queries and displaying results.
    
    Args:
        chunks (list[dict]): The list of chunks corresponding to the FAISS index.
        vector_index (faiss.Index): The FAISS index containing chunk embeddings.
    """
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
            
        result = run_rag_pipeline(user_input, vector_index, chunks)
        free_generation_model()
        
        retrieved_chunks = result["reranked_chunks"]
        signals = result["reliability_signals"]
        generated_response = result["answer"]
        references = result["references"]
                    
        print(f"\n{'='*60}")
        print("RESULTADO DE LA CONSULTA:")
        print(f"Respuesta generada: {generated_response}")
        if references:
            print(references)
        print("\nFragmentos más relevantes recuperados:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            score = chunk.get('rerank_score', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"{i}. [Score: {score_str}] - Fichero: {chunk.get('source', 'Desconocido')} (Pág. {chunk.get('page','N/A')})")
            print(f"     Extracto: {chunk['text'][:180]}...")
        print(f"\n{'='*60}")
        
        if SHOW_METRICS:
            print("\nMétricas de retrieval:")
            for source, values in (signals["retrieval_metrics"] or {}).items():
                formatted = ", ".join(f"{k}={v:.4f}" if v is not None else f"{k}=N/A"
                                      for k,v in values.items())
                print(f" [{source}] {formatted}")

if __name__ == "__main__":
    main()