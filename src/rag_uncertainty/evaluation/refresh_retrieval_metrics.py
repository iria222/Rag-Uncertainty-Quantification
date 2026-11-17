"""Module to calculate retrieval metrics offline in the results file based on cached score lists.

Usage:
python -m rag_uncertainty.evaluation.refresh_retrieval_metrics --results results.json --cache cache.json
"""

import argparse

from rag_uncertainty.evaluation.retrieval_metrics import compute_retrieval_metrics
from rag_uncertainty.utils.checkpoint import load_checkpoint, save_checkpoint

def refresh_retrieval_metrics(results_path: str, cache_path: str) -> int:
    """Refresh retrieval metrics in the results file based on the cached score lists.
    
    Args:
        results_path (str): Path to the results JSON file.
        cache_path (str): Path to the cache JSON file containing score lists.
        
    Returns:
        int: The number of entries updated with new retrieval metrics."""
    
    results = load_checkpoint(results_path)
    cache = load_checkpoint(cache_path)
    
    score_map = {
        item["query"]["id"]: item.get("score_lists")
        for item in cache if "query" in item and "id" in item["query"]
    }
    
    updated = 0
    
    for entry in results:
        score_lists = score_map.get(entry["id"])
        if not score_lists:
            continue
        entry["reliability_signals"]["retrieval_metrics"] = compute_retrieval_metrics(score_lists)
        updated += 1
        
    save_checkpoint(results_path, results)
    return updated

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--cache", required=True)
    
    args = parser.parse_args()
    
    print(f"Updated {refresh_retrieval_metrics(args.results, args.cache)} entries")