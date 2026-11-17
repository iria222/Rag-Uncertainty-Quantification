import numpy as np


def get_ordered_scores(scores: np.ndarray, top_k: int | None = None) -> np.ndarray:
    
    scores = np.asarray(scores, dtype=np.float64)
    
    if scores.size == 0:
        return scores
    
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must contain only finite values.")
    
    # Ensure descending order
    scores = np.sort(scores)[::-1]
    
    return scores[:top_k] if top_k is not None else scores

def compute_max_score(scores: np.ndarray) -> float:
    
    return float(scores[0])

def compute_score_gap(scores: np.ndarray) -> float:
    
    score_gap = (
        float(scores[0] - scores[-1])
        if len(scores) >= 2 else 0.0
    )
    
    return score_gap

def compute_std(scores: np.ndarray) -> float:
    
    return float(np.std(scores, ddof=0))

def compute_lsmv(scores: np.ndarray) -> float:
    return float(compute_max_score(scores) * compute_std(scores))

# Add/remove metrics here
RETRIEVAL_METRICS = {
    "max_score": compute_max_score,
    "score_gap": compute_score_gap,
    "std": compute_std,
    "lsmv": compute_lsmv
}

SCORE_SOURCES = {
    "retrieval": {"k": 25},
    "rerank": {"k": 25}
}

def compute_retrieval_metrics(score_lists: dict[str, list[float]],
                              metrics: dict = RETRIEVAL_METRICS,
                              score_sources: dict = SCORE_SOURCES) -> dict:
    """Compute retrieval metrics for the given score lists based on the specified metrics and score sources.
    
    Args:
        score_lists (dict[str, list[float]]): A dictionary containing lists of scores.
        metrics (dict, optional): A dictionary of metric functions to compute. Defaults to RETRIEVAL_METRICS.
        score_sources (dict, optional): A dictionary specifying the score sources and their configurations. Defaults to SCORE_SOURCES.
    
    Returns:
        dict: A dictionary containing the computed metrics for each score source.
    """
    result = {}
    
    for source_name, cfg in score_sources.items():
        scores = get_ordered_scores(score_lists.get(source_name, []), top_k=cfg["k"])
        
        if scores.size == 0:
            result[source_name] = {name: None for name in metrics}
            continue
        result[source_name] = {name: fn(scores) for name, fn in metrics.items()}
    
    return result