"""Evaluation runner — executes model output against test cases.

Flow:
    raw model output
        → extract Code: section
        → write to temp file
        → execute via subprocess with timeout
        → compare stdout to expected output
        → return pass/fail + error category

Error categories:
    formatting_failure  — no Code: section in output
    syntax_error        — ast.parse fails
    runtime_error       — process crashes (IndexError, TypeError etc.)
    logic_error         — runs but output doesn't match expected
    timeout             — exceeds 5 second limit
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TEST_CASES_DIR = ROOT / "data" / "test_cases"


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(raw_text: str) -> str | None:
    """Extract the Code: section from structured model output.

    Returns None if Code: section is missing — logged as formatting_failure.
    """
    if "Code:" not in raw_text:
        return None
    code = raw_text.split("Code:", 1)[1].strip()
    # strip markdown fences if present
    if code.startswith("```"):
        lines = code.splitlines()
        # drop first line (```python or ```) and last line (```)
        inner = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            if line.strip() == "```" and in_block:
                break
            if in_block:
                inner.append(line)
        code = "\n".join(inner).strip()
    return code if code else None


def parse_reasoning(raw_text: str) -> str:
    """Extract Approach + Reasoning text (everything before Code:).

    Used for BLEU/ROUGE computation.
    Returns empty string if not found.
    """
    if "Code:" not in raw_text:
        return raw_text.strip()
    return raw_text.split("Code:", 1)[0].strip()


# ---------------------------------------------------------------------------
# Execution harness
# ---------------------------------------------------------------------------

HARNESS_TEMPLATE = """\
{prompt}

{code}

import json, sys

raw = json.loads(sys.stdin.read())
entry_point = raw["entry_point"]
args        = raw.get("args", [])
kwargs      = raw.get("kwargs", {{}})
expected    = raw["expected_output"]

if "." in entry_point:
    result = eval(entry_point)(*args, **kwargs)
else:
    fn = globals().get(entry_point)
    if fn is None:
        raise NameError(f"entrypoint '{{entry_point}}' not found")
    result = fn(*args, **kwargs)

match = result == expected
print(json.dumps({{"result": result, "expected": expected, "passed": match}}))
"""


def _classify_error(returncode: int, stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if "SyntaxError" in stderr:
        return "syntax_error"
    if any(e in stderr for e in (
        "IndexError", "TypeError", "ValueError",
        "ZeroDivisionError", "AttributeError", "KeyError", "NameError"
    )):
        return "runtime_error"
    if returncode != 0:
        return "runtime_error"
    return "logic_error"


def run_solution(
    code: str,
    entry_point: str,
    test_case: dict[str, Any],
    prompt: str = "",
    timeout: int = 5,
) -> dict[str, Any]:
    """Run one test case in an isolated subprocess.

    Returns:
        {
            "passed": bool,
            "error_type": str | None,
            "actual": Any,
            "expected": Any,
        }
    """
    harness = HARNESS_TEMPLATE.format(prompt=prompt, code=code)
    payload = json.dumps({
        "entry_point": entry_point,
        "args": test_case.get("args", []),
        "kwargs": test_case.get("kwargs", {}),
        "expected_output": test_case["expected_output"],
    })

    timed_out = False
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(harness)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        proc = type("P", (), {"returncode": 1, "stdout": "", "stderr": "timeout"})()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if timed_out or proc.returncode != 0:
        return {
            "passed": False,
            "error_type": _classify_error(proc.returncode, proc.stderr, timed_out),
            "actual": None,
            "expected": test_case["expected_output"],
        }

    try:
        output = json.loads(proc.stdout.strip())
        passed = output.get("passed", False)
        return {
            "passed": passed,
            "error_type": None if passed else "logic_error",
            "actual": output.get("result"),
            "expected": test_case["expected_output"],
        }
    except json.JSONDecodeError:
        return {
            "passed": False,
            "error_type": "runtime_error",
            "actual": None,
            "expected": test_case["expected_output"],
        }


def evaluate_problem(
    problem_id: str,
    raw_model_output: str,
    test_cases_dir: Path = TEST_CASES_DIR,
    timeout: int = 5,
) -> dict[str, Any]:
    """Evaluate one model output against its test cases.

    Returns:
        {
            "problem_id": str,
            "passed": bool,
            "error_category": str | None,
            "reasoning_text": str,      # for BLEU/ROUGE
            "per_case_results": list,
        }
    """
    reasoning_text = parse_reasoning(raw_model_output)

    # step 1 — extract code
    code = parse_output(raw_model_output)
    if code is None:
        return {
            "problem_id": problem_id,
            "passed": False,
            "error_category": "formatting_failure",
            "reasoning_text": reasoning_text,
            "per_case_results": [],
        }

    # step 2 — syntax check before subprocess
    try:
        ast.parse(code)
    except SyntaxError:
        return {
            "problem_id": problem_id,
            "passed": False,
            "error_category": "syntax_error",
            "reasoning_text": reasoning_text,
            "per_case_results": [],
        }

    # step 3 — load test cases
    tc_path = test_cases_dir / f"{problem_id}.json"
    if not tc_path.exists():
        return {
            "problem_id": problem_id,
            "passed": False,
            "error_category": "missing_test_case_file",
            "reasoning_text": reasoning_text,
            "per_case_results": [],
        }

    with tc_path.open() as f:
        tc_data = json.load(f)

    entry_point = tc_data.get("entry_point", "")
    prompt = tc_data.get("prompt", "")
    test_cases = tc_data.get("test_cases", [])

    # step 4 — run each test case
    per_case_results = []
    for tc in test_cases:
        result = run_solution(code, entry_point, tc, prompt, timeout)
        per_case_results.append(result)
        if not result["passed"]:
            return {
                "problem_id": problem_id,
                "passed": False,
                "error_category": result["error_type"],
                "reasoning_text": reasoning_text,
                "per_case_results": per_case_results,
            }

    return {
        "problem_id": problem_id,
        "passed": True,
        "error_category": None,
        "reasoning_text": reasoning_text,
        "per_case_results": per_case_results,
    }


def evaluate_batch(
    results: list[dict[str, str]],
    test_cases_dir: Path = TEST_CASES_DIR,
    timeout: int = 5,
) -> list[dict[str, Any]]:
    """Evaluate a batch of model outputs.

    Args:
        results: list of {"problem_id": str, "model_output": str}

    Returns:
        list of evaluate_problem() results
    """
    evaluated = []
    for i, item in enumerate(results):
        result = evaluate_problem(
            problem_id=item["problem_id"],
            raw_model_output=item["model_output"],
            test_cases_dir=test_cases_dir,
            timeout=timeout,
        )
        evaluated.append(result)
        if i % 10 == 0:
            passed = sum(1 for r in evaluated if r["passed"])
            print(f"Progress: {i}/{len(results)} — passed so far: {passed}")

    return evaluated