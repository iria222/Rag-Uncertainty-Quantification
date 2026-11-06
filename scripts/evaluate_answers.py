import argparse
import json
from pathlib import Path

from rag_uncertainty.evaluation.llm_judge import LLMJudge, load_fewshot_examples

def load_json(path: Path):
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)
        
def main():
    parser = argparse.ArgumentParser(description="Evaluate saved RAG answers using an LLM judge")
    
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    
    parser.add_argument("--save-every", type=int, default=5, help="Save the checkpoint every N records")
    parser.add_argument("--max-chunks", type=int, default=None, help="Max retrieved chunks per case")
    parser.add_argument("--max-chunks-chars", type=int, default=None, help="Max characters per chunk")
    parser.add_argument("--max-input-tokens", type=int, default=None, help="Error out if the prompt exceeds this")
    
    args = parser.parse_args()
    
    
    
    records = load_json(args.input)
    examples = load_fewshot_examples(args.examples)
    
    judge = LLMJudge(
        max_chunks = args.max_chunks,
        max_chunks_chars = args.max_chunks_chars,
        max_input_tokens = args.max_input_tokens,
    )
    
    try:
        evaluated_records = judge.evaluate_dataset(
            records = records,
            fewshot_examples = examples,
            output_path = args.output,
            save_every=args.save_every,
        )
    finally:
        judge.unload_model()
        
    n_errors = sum("error" in r.get("llm_judge", {}) for r in evaluated_records)
    print(f"\nResults saved to: {args.output}")
    print(f"Records evaluated: {len(evaluated_records)} | with errors: {n_errors}")
    
    if n_errors:
        print("Re-run the same command to retry only the failed records")
    
if __name__ == "__main__":
    main()