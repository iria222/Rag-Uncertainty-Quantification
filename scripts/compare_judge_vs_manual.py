"""Compare the LLM judge labels against manual annotations

Usage:
python scripts/compare_judge_vs_manual.py --judge results_judged.json --gold manual_annotations.json"""

import argparse
import json
from collections import Counter
from pathlib import Path

CORRECTNESS_ORDER = ["incorrect", "partially_correct", "correct"]
FAITHFULNESS_ORDER = ["unsupported", "partially_supported", "supported"]

def load_json(path:Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def norm(text:str) -> str:
    return " ".join((text or "").split())

def weight(i: int, j: int, k: int, weights: str | None) -> float:
        if weights == "quadratic":
            return ((i-j) / (k - 1)) **2
        return 0.0 if i == j else 1.0
    
def cohen_kappa(gold:list, pred: list, labels: list, weights: str | None = None) -> float | None:
    n = len(gold)
    if n == 0:
        return None
    
    k = len(labels)
    idx = {label: i for i, label in enumerate(labels)}
    observed = [[0] * k for _ in range(k)]
    for g, p in zip(gold, pred):
        observed[idx[g]][idx[p]] += 1
    
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    
    num = sum(weight(i,j, k, weights) * observed[i][j] for i in range(k) for j in range(k))
    den = sum(weight(i,j, k,  weights) * row[i] * col[j] / n for i in range(k) for j in range(k))
    
    if den == 0:
        return None
    return 1- num / den

def fmt(value: float | None) -> str:
    return "n/a (undefined)" if value is None else f"{value:.3f}"

def print_confusion(gold: list, pred: list, labels: list) -> None:
    counts = Counter(zip(gold, pred))
    width = max(len(label) for label in labels) + 2
    print("gold \\ judge".ljust(width + 2) + "".join(label.ljust(width) for label in labels))
    for g in labels:
        print(g.ljust(width + 2) + "".join(str(counts.get((g, p), 0)).ljust(width) for p in labels))
        
def report_dimension(name:str, gold:list, pred: list, order: list, ids: list, reasonings: list) -> None:
    valid = [(g,p,i,r) for g, p, i, r in zip(gold, pred, ids, reasonings) if g in order and p in order]
    if len(valid) < len(gold):
        excluded = [i for g, p, i, _ in zip(gold, pred, ids, reasonings) if g not in order or p not in order]
        print(f"\nNOTE: {len(excluded)} record(s) excluded from {name} (label outside {order}): {excluded}")
    gold, pred, ids, reasonings = (list(x) for x in zip(*valid)) if valid else ([],[],[],[])
    n = len (gold)
    if n == 0:
        print(f"\n=== {name}: no comparable records ===")
        return
    agree = sum(g == p for g, p in zip(gold, pred))
    print(f"\n=== {name} (n={n}) ===")
    print(f"Raw agreement: {agree}/{n} = {agree / n:.1%}" if n else "No records")
    print(f"Cohen's kappa:            {fmt(cohen_kappa(gold, pred, order))}")
    print(f"Weighted kappa (quadr.):  {fmt(cohen_kappa(gold, pred, order, 'quadratic'))}")
    print("Gold distribution: ", dict(Counter(gold)))
    print("Judge distribution:", dict(Counter(pred)))
    print()
    print_confusion(gold, pred, order)
    
    disagreements = [(i,g,p,r) for i, g, p, r in zip(ids, gold, pred, reasonings) if g!=p]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)})")
        for i,g,p,r in disagreements:
            print(f" - {i}: gold = {g} | judge= {p}")
            print(f" judge reasoning: {norm(r)[:200]}")
            
def main():
    parser = argparse.ArgumentParser(description = "Compare LLM judge vs manual annotations")
    parser.add_argument("--judge", type=Path, required = True, help= "Judge output JSON (records with 'llm_judge')")
    parser.add_argument("--gold", type=Path, default=None, help="Manual annotations JSON(default: same file as --judge)")
    args = parser.parse_args()
    
    judged = load_json(args.judge)
    gold_records = load_json(args.gold) if args.gold else judged
    gold_by_id = {r["id"]: r for r in gold_records}
    
    missing = [r["id"] for r in judged if r["id"] not in gold_by_id]
    if missing:
        print(f"WARNING: {len(missing)} judged record(s) have no manual annotation and are ignored: {missing[:5]}")
    
    pairs = []
    n_errors = 0
    for rec in judged:
        gold = gold_by_id.get(rec["id"])
        if gold is None:
            continue
        judge = rec.get("llm_judge") or {}
        if "error" in judge or "correctness" not in judge:
            n_errors += 1
            continue
        pairs.append((rec, gold, judge))
        
    print(f"Records judged: {len(judged)} | matched with annotations: {len(pairs) + n_errors} | judge errors skipped: {n_errors}")
    
    mismatched = [r["id"] for r, g, _ in pairs if norm(r.get("generated_response", "")) != norm(g.get("generated_response", ""))]
    if mismatched:
        print(f"\nWARNING: {len(mismatched)} record(s) have a different generated_response in the judge output and in the "
              f"annotations. The agreement for those is not meaningful: {mismatched[:5]}{' ...' if len(mismatched) > 5 else ''}")
    
    # Abstentions are labelled in code by the judge, so they are excluded from agreement (including them would inflate kappa)
    # They are checked separately as a sanity test
    gold_abstained = [g["correctness"] == "not_applicable" for _, g, _ in pairs]
    judge_abstained = [j["correctness"] == "not_applicable" for _,_,j in pairs]
    abst_disagree = [r["id"] for (r, _, _), a, b in zip(pairs, gold_abstained, judge_abstained) if a !=b]
    print(f"\nAbstentions: gold={sum(gold_abstained)} | judge={sum(judge_abstained)} | disagreements={len(abst_disagree)}")
    
    if abst_disagree:
        print(f"Abstention mismatch (check detect_abstention / the annotations): {abst_disagree}")
    
    answered = [(r,g,j) for (r,g,j), a, b in zip(pairs, gold_abstained, judge_abstained) if not a and not b]
    print(f"Answered records compared: {len(answered)}")
    if not answered:
        return
    
    ids = [r["id"] for r,_,_ in answered]
    reasonings = [j.get("reasoning", "") for _,_, j in answered]
    
    report_dimension("CORRECTNESS", [g["correctness"] for _, g, _ in answered],
                       [j["correctness"] for _, _, j in answered], CORRECTNESS_ORDER, ids, reasonings)
    
    report_dimension("FAITHFULNESS", [g["faithfulness"] for _,g,_ in answered],
                        [j["faithfulness"] for _,_,j in answered], FAITHFULNESS_ORDER, ids, reasonings)
    
    # Binary view of faithfulness
    comparable = [(g, j) for _, g, j in answered
                  if g["faithfulness"] in FAITHFULNESS_ORDER and j["faithfulness"] in FAITHFULNESS_ORDER]
    gold_bin = ["supported" if g["faithfulness"] == "supported" else "not_fully_supported" for g,_ in comparable]
    pred_bin = ["supported" if j["faithfulness"] == "supported" else "not_fully_supported" for _,j in comparable]
    agree = sum(a == b for a, b in zip(gold_bin, pred_bin))
    print(f"\n=== FAITHFULNESS, binary (supported vs rest) ===")
    if not gold_bin:
        return
    print(f"Raw agreement: {agree}/{len(gold_bin)} = {agree / len(gold_bin):.1%}")
    print(f"Gold 'not fully supported': {gold_bin.count('not_fully_supported')} | "
          f"Judge: {pred_bin.count('not_fully_supported')}")
    
if __name__ == "__main__":
    main()
