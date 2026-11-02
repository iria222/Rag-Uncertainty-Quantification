import json
import sys

def analyze_json_results(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1
    
    total_questions = len(data)
    
    answerable = 0
    unanswerable = 0
    
    answerable_answered = 0
    answerable_abstained = 0
    
    unanswerable_answered = 0
    unanswerable_abstained = 0
    
    for item in data:
        is_answerable = item.get("answerable", False)
        reliability = item.get("reliability_signals", {})
        decision = reliability.get("decision")
        response_text = item.get("generated_response", "")
        
        abstained = decision == "abstain" or "[ABSTENCION]" in response_text
        
        if is_answerable:
            answerable+=1
            if abstained:
                answerable_abstained += 1
            else:
                answerable_answered += 1
        else:
            unanswerable += 1
            if abstained:
                unanswerable_abstained += 1
            else:
                unanswerable_answered += 1
                
    print_results(total_questions, answerable, unanswerable, answerable_answered, answerable_abstained, unanswerable_answered, unanswerable_abstained)

def print_results(total_questions, answerable, unanswerable, answerable_answered, answerable_abstained, unanswerable_answered, unanswerable_abstained):
    print("=" * 55)
    print("        MODEL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Total analyzed questions: {total_questions}\n")

    print(f"1. ANSWERABLE Questions: {answerable}")
    print(
        f"   ├─ Answered by model: {answerable_answered} "
        f"({(answerable_answered/answerable*100) if answerable else 0:.1f}%)"
    )
    print(
        f"   └─ Unanswered (Abstained/Failed): {answerable_abstained} "
        f"({(answerable_abstained/answerable*100) if answerable else 0:.1f}%)\n"
    )

    print(f"2. UNANSWERABLE Questions: {unanswerable}")
    print(
        f"   ├─ Correctly abstained (Unanswered): {unanswerable_abstained} "
        f"({(unanswerable_abstained/unanswerable*100) if unanswerable else 0:.1f}%)"
    )
    print(
        f"   └─ Hallucinated / Incorrectly answered: {unanswerable_answered} "
        f"({(unanswerable_answered/unanswerable*100) if unanswerable else 0:.1f}%)"
    )
    print("=" * 55)
    
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        raise ValueError(
            "Usage Error: You must provide the JSON file path as an argument.\n"
            "Example: python -m rag_uncertainty.analysis.metrics_summary [results.json]"
        )
    analyze_json_results(file_name)