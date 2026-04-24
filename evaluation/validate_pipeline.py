"""Pipeline self-test — run this before any real evaluation.

Verifies:
1. Known-correct solutions score Pass@1 = 1.0
2. Known-wrong solutions score Pass@1 = 0.0
3. Formatting failures are caught
4. Syntax errors are caught
5. Timeout is enforced

Run locally before Colab:
    python evaluation/validate_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.runner import evaluate_problem
from evaluation.metrics import pass_at_1, compute_all_metrics

TEST_CASES_DIR = ROOT / "data" / "test_cases"


def get_first_test_problem() -> tuple[str, str] | None:
    """Get the first available problem_id and its correct solution from test split."""
    test_path = ROOT / "data" / "processed" / "test.json"
    if not test_path.exists():
        return None

    with test_path.open() as f:
        records = json.load(f)

    if not records:
        return None

    rec = records[0]
    return rec["problem_id"], rec["output"]


def run_validation() -> bool:
    print("=" * 60)
    print("Evaluation Pipeline Validation")
    print("=" * 60)

    first = get_first_test_problem()
    if first is None:
        print("FAIL — could not load test split")
        return False

    problem_id, correct_output = first
    print(f"\nUsing problem: {problem_id}")

    all_passed = True

    # ---- Test 1: correct solution should pass ----------------------------
    print("\n[1/5] Correct solution → expect PASS")
    result = evaluate_problem(problem_id, correct_output, TEST_CASES_DIR)
    if result["passed"]:
        print("  ✓ PASS")
    else:
        print(f"  ✗ FAIL — {result['error_category']}")
        all_passed = False

    # ---- Test 2: wrong solution should fail with logic_error -------------
    print("\n[2/5] Wrong solution → expect logic_error")
    wrong_output = (
        correct_output.rsplit("Code:", 1)[0]
        + "Code:\n"
        + "class Solution:\n"
        + "    def checkSubarraySum(self, nums, k):\n"
        + "        return False\n"
    )
    result = evaluate_problem(problem_id, wrong_output, TEST_CASES_DIR)
    if not result["passed"]:
        print(f"  ✓ FAIL as expected — {result['error_category']}")
    else:
        print("  ✗ Wrong solution passed — pipeline bug")
        all_passed = False

    # ---- Test 3: missing Code: section → formatting_failure --------------
    print("\n[3/5] Missing Code: section → expect formatting_failure")
    no_code_output = "Approach: Use a hashmap.\n\nReasoning:\n1. Iterate.\n2. Return."
    result = evaluate_problem(problem_id, no_code_output, TEST_CASES_DIR)
    if result["error_category"] == "formatting_failure":
        print("  ✓ formatting_failure caught correctly")
    else:
        print(f"  ✗ Expected formatting_failure, got {result['error_category']}")
        all_passed = False

    # ---- Test 4: syntax error → syntax_error -----------------------------
    print("\n[4/5] Syntax error → expect syntax_error")
    syntax_error_output = "Approach: x\n\nReasoning:\n1. x\n\nCode:\ndef broken(\n    return 1\n"
    result = evaluate_problem(problem_id, syntax_error_output, TEST_CASES_DIR)
    if result["error_category"] == "syntax_error":
        print("  ✓ syntax_error caught correctly")
    else:
        print(f"  ✗ Expected syntax_error, got {result['error_category']}")
        all_passed = False

    # ---- Test 5: infinite loop → timeout ---------------------------------
    print("\n[5/5] Infinite loop → expect timeout")
    timeout_output = (
        "Approach: x\n\nReasoning:\n1. x\n\nCode:\n"
        "class Solution:\n"
        "    def checkSubarraySum(self, nums, k):\n"
        "        while True:\n"
        "            pass\n"
    )
    result = evaluate_problem(problem_id, timeout_output, TEST_CASES_DIR, timeout=3)
    if result["error_category"] == "timeout":
        print("  ✓ timeout caught correctly")
    else:
        print(f"  ✗ Expected timeout, got {result['error_category']}")
        all_passed = False

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    if all_passed:
        print("All 5 validation checks passed. Pipeline is ready.")
    else:
        print("Some validation checks failed. Fix before running evaluation.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    ok = run_validation()
    raise SystemExit(0 if ok else 1)