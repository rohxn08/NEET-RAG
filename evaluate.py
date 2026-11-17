"""
Evaluation script: python evaluate.py
Runs all questions in eval/eval_set.json and prints a score table.
"""
import json
from src.rag import ask

EVAL_PATH = "eval/eval_set.json"


def run_eval():
    with open(EVAL_PATH) as f:
        questions = json.load(f)

    results = []
    for q in questions:
        response = ask(q["question"])
        expected_refused = q["expected"] == "refuse"
        actual_refused   = response.refused

        if expected_refused and actual_refused:
            outcome = "PASS (correct refusal)"
        elif not expected_refused and not actual_refused:
            outcome = "PASS (answered)"
        elif expected_refused and not actual_refused:
            outcome = "FAIL (should have refused)"
        else:
            outcome = "FAIL (refused but shouldn't have)"

        results.append({
            "id":       q["id"],
            "question": q["question"][:70],
            "expected": q["expected"],
            "refused":  actual_refused,
            "outcome":  outcome,
        })

    passed = sum(1 for r in results if r["outcome"].startswith("PASS"))
    total  = len(results)

    print(f"\n{'='*72}")
    print(f"  NEET RAG EVALUATION  |  {passed}/{total} passed")
    print(f"{'='*72}")
    print(f"{'ID':<4}  {'Expected':<10}  {'Refused':<8}  {'Outcome':<30}  Question")
    print("-" * 72)
    for r in results:
        print(f"{r['id']:<4}  {r['expected']:<10}  {str(r['refused']):<8}  {r['outcome']:<30}  {r['question']}")
    print(f"{'='*72}\n")
    print(f"  Score: {passed}/{total} = {passed/total:.0%}\n")


if __name__ == "__main__":
    run_eval()
