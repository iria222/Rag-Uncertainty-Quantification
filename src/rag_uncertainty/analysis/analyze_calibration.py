"""Offline analysis of calibration and abstention

Use:
python -m rag_uncertainty.analysis.analyze_calibration results.json
python -m rag_uncertainty.analysis.analyze_calibration results.json --labels labels.json
python -m rag_uncertainty.analysis.analyze_calibration --fewshot selfeval_fewshot.json

Priority order:
1. --labels: JSON {"<id>": true/false}
2. Field "correct" in results.json
3. Heuristic: recall of ground truth in answer
"""

import argparse
import json
import re
import unicodedata

import numpy as np
from sklearn.metrics import roc_auc_score

THRESHOLDS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def _normalize(text:str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub("[^a-z0-9 ]+", " ", text)

def heuristic_correct(answer:str, ground_truth: str, min_recall:float = 0.6) -> bool:
    """Heuristic to determine if the answer is correct based on the ground truth
    Proxy: if the answer contains at least min_recall fraction of the words in the ground truth, it is considered correct
    
    Args:
        answer (str): generated answer
        ground_truth (str): the ground truth answer
        min_recall (float): minimum fraction of words in ground truth that must be present in answer for it to be considered correct
        
    Returns:
        bool: whether the answer is considered correct or not
    """
    body = re.split(r"\n\s*referencias\s*:", answer, maxsplit=1, flags = re.IGNORECASE)[0]
    gt_tokens = {t for t in _normalize(ground_truth or "").split() if len(t) >3}
    if not gt_tokens:
        return False
    ans_tokens = set(_normalize(body).split())
    return len(gt_tokens & ans_tokens) /len(gt_tokens) >= min_recall

def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Computes the expected calibration error (ECE) of a set of predictions
    
    Args:
        p (np.ndarray): predicted probabilities
        y (np.ndarray): true labels
        n_bins (int): number of bins to use for calibration error computation
    
    Returns:
        float: the expected calibration error
    """
    order = np.argsort(p)
    p,y = p[order], y[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    return float(np.mean([abs(p[b].mean() - y[b].mean()) for b in bins if len(b) > 0]))

def bootstrap_auroc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Computes a bootstrap confidence interval for the AUROC of a set of predictions
    
    Args:
        y (np.ndarray): true labels
        p (np.ndarray): predicted probabilities
        n_boot (int): number of bootstrap samples
        seed (int): random seed for reproducibility
    
    Returns:
        tuple: lower and upper bounds of the confidence interval
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0,n,n)
        if 0 < y [idx].sum() < n:
            vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5,97.5])
    return float(lo), float(hi)

def drop_fewshot_overlap(items: list, fewshot_path:str) -> list:
    """Drop items which id or question appear in the few-shot examples

    Args:
        items (list): List of items to filter
        fewshot_path (str, optional): Path of the few-shot examples file. Defaults to SELF_EVAL_FEWSHOT_PATH.

    Returns:
        list: The filtered list of items
    """
    with open(fewshot_path, "r", encoding = "utf-8") as f:
        examples = json.load(f)["examples"]
    ids = {str(e["origin_id"]) for e in examples}
    question = {_normalize(e["question"]).strip() for e in examples}
    kept = [i for i in items if str(i["id"]) not in ids and _normalize(i["question"]).strip() not in question]
    print(f"Dropped {len(items) - len(kept)} items that coincide with the few-shot examples")
    return kept

def load_items(path: str, labels_path: str |None):
    """Loads items from a JSON file and optionally applies labels from another JSON file
    
    Args:
        path (str): Path to the JSON file containing the items
        labels_path (str | None): Path to the JSON file containing the labels.
    
    Returns:
        list: The list of loaded items
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = {}
    if labels_path:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    
    items = []
    for d in data:
        rs = d.get("reliability_signals", {})
        raw = d.get("raw_response") or d.get("generated_response") or ""
        answerable = bool(d.get("answerable"))
        model_abstained = bool(rs.get("model_abstained", "[ABSTENCION]" in raw))
        
        if str(d["id"]) in labels:
            correct = bool(labels[str(d["id"])])
        elif "correct" in d:
            correct = bool(d["correct"])
        else:
            correct = heuristic_correct(raw, d.get("ground_truth", ""))
            
        items.append({
            "id": d["id"],
            "question": d.get("question", ""),
            "answerable": answerable,
            "model_abstained": model_abstained,
            "p_true": rs.get("confidence_score"),
            "good": answerable and correct and not model_abstained,
        })
        
    return items


def discriminatrion_report(items):
    """Prints a discrimination report for the given items
    
    Args:
        items (list): List of items to analyze
    """
    answered = [i for i in items if not i["model_abstained"] and i["p_true"] is not None]
    print(f"\nNo abstained answers: {len(answered)}")
    if len(answered) < 10:
        print("Answered number too low to calculate metrics")
        return
    p = np.array([i["p_true"] for i in answered])
    y = np.array([i["good"] for i in answered], dtype = float)
    print(f"Good: {int(y.sum())} | Bad (incorrect or answered not answerable questions): {int((1-y).sum())}")
    print(f"Average P(True): good={p[y == 1].mean():.3f} bad={p[y == 0].mean():.3f}")
    
    if 0 < y.sum() < len(y):
        lo, hi = bootstrap_auroc_ci(y,p)
        print(f"AUROC (P(True vs good answer)): {roc_auc_score(y,p):.3f}  IC95% [{lo:.3f}, {hi:.3f}]  (0.5 = random)")
    print(f"ECE (10 bins, equal population): {expected_calibration_error(p,y):.3f}")
    print(f"Brier: {float(np.mean((p-y) ** 2)):.3f}")
    
def sweep_report(items):
    """Prints a sweep report for the given items
    
    Args:
        items (list): List of items to analyze
    """
    n = len(items)
    n_ans = sum(i["answerable"] for i in items)
    n_unans = n - n_ans
    print(f"\nTotal: {n}  |  answerable: {n_ans}  |  unanswerable: {n_unans}")
    
    label_width = 30
    header = (f"{'Threshold':>{label_width}} {'Coverage':>10} {'Acc.sel.':>9} {'Hallucin.':>8} {'False abst.':>12} {'Correct abst.':>13}")
    print("\n" + header)
    print("-" * len(header))
    
    def row(label, tau):
        answered = [
            i for i in items
            if not i["model_abstained"] and (tau is None or (i["p_true"] is not None and i["p_true"] >=tau))
        ]
        
        abstained = [i for i in items if i not in answered]
        n_answered = len(answered)
        n_good = sum(i["good"] for i in answered)
        coverage = n_answered / n if n else 0.0
        sel_acc = n_good / n_answered if n_answered else float("nan")
        
        halluc = (n_answered - n_good) / n if n else 0.0
        false_abst = sum(i["answerable"] for i in abstained) / n_ans if n_ans else float("nan")
        correct_abst = sum(not i["answerable"] for i in abstained) / n_unans if n_unans else float("nan")
        print(f"{label:>{label_width}} {coverage:>10.3f} {sel_acc:>9.3f} {halluc:>8.3f} {false_abst:>12.3f} {correct_abst:>12.3f}")
        
    row("without Self Evaluation (SE)", None)
    for tau in THRESHOLDS:
        row(f"{tau:.1f}", tau)
        
    print(
        "\nCoverage = % of answered questions  |  Acc. sel. = % of good generated answers\n"
        "Hallucin. = bad generated answers / total  | False abst. = abstentions for answerable questions\n"
        "Correct. abst. = abstetions over not answerable questions (without Self Evaluation (SE) = just abstention by prompt)"
    )
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="results.json generated by the RAG pipeline")
    parser.add_argument("--labels", default=None, help="JSON {id: bool} with the correction of each answer")
    parser.add_argument("--fewshot", default=None, help="selfeval_fewshot.json: drop those items from evaluation")
    args = parser.parse_args()
    
    items = load_items(args.results, args.labels)
    if args.fewshot:
        items = drop_fewshot_overlap(items, args.fewshot)
    discriminatrion_report(items)
    sweep_report(items)
    
if __name__ == "__main__":
    main()
    