"""Step 6 — Generate executable test cases for every candidate record.

Runs BEFORE build_dataset.py split stage.

Strict schema per test case:
    {
        "args": [],
        "kwargs": {...},
        "expected_output": <any>
    }

LeetCode records: test cases parsed directly from input_output field (no Groq).

Inputs:
    data/raw/leetcode_structured.jsonl

Outputs:
    data/test_cases/{problem_id}.json
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

from utils import (
    extract_code,
    load_jsonl,
    save_json,
)


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TEST_CASES_DIR = ROOT / "data" / "test_cases"


# ---------------------------------------------------------------------------
# Entrypoint inference fallback
# ---------------------------------------------------------------------------

def infer_entrypoint(code: str) -> str | None:
    """Return the first callable name found in code.

    Checks class methods first (returns 'ClassName().method'),
    then falls back to top-level functions.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and not member.name.startswith("_"):
                    return f"{node.name}().{member.name}"
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name

    return None


# ---------------------------------------------------------------------------
# LeetCode: parse input_output field directly (no Groq needed)
# ---------------------------------------------------------------------------

def parse_input_string(input_str: str) -> dict | None:
    """Parse 'nums = [3,3], target = 6' into {'nums': [3,3], 'target': 6}.
    
    Handles JSON null/true/false by replacing with Python equivalents first.
    """
    # replace JSON literals with Python equivalents for eval
    sanitized = (
        input_str
        .replace("null", "None")
        .replace("true", "True")
        .replace("false", "False")
    )
    try:
        result = eval(f"dict({sanitized})")  # noqa: S307
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


def parse_output_string(output_str: str) -> Any:
    """Parse '[0, 1]' or 'None'/'null' into a Python value."""
    sanitized = (
        output_str.strip()
        .replace("null", "None")
        .replace("true", "True")
        .replace("false", "False")
    )
    try:
        return eval(sanitized)  # noqa: S307
    except Exception:
        return None


def extract_leetcode_testcases(
    input_output: list[dict],
    max_cases: int = 5,
) -> list[dict] | None:
    converted = []
    for item in input_output:
        if len(converted) >= max_cases:
            break
        kwargs = parse_input_string(item.get("input", ""))
        if kwargs is None:
            continue
        expected = parse_output_string(item.get("output", ""))
        converted.append({
            "args": [],
            "kwargs": kwargs,
            "expected_output": expected,
        })

    if len(converted) < 3:
        return None

    return converted


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_records(records: list[dict]) -> tuple[int, int, int]:
    saved = skipped = failed = 0

    for i, rec in enumerate(records):
        problem_id = rec["problem_id"]
        out_path = TEST_CASES_DIR / f"{problem_id}.json"

        # resume support — skip already processed
        if out_path.exists():
            skipped += 1
            continue

        code = extract_code(rec.get("output", ""))
        if not code:
            print(f"  [{i}] FAIL — no code section: {problem_id}")
            failed += 1
            continue

        # entry_point is stored for LeetCode; infer only as a fallback
        entry_point = rec.get("entry_point", "").strip()
        if not entry_point:
            entry_point = infer_entrypoint(code)
        if not entry_point:
            print(f"  [{i}] FAIL — no entrypoint found: {problem_id}")
            failed += 1
            continue

        # prompt contains ListNode/TreeNode helpers needed by execution harness
        prompt = rec.get("prompt", "")

        input_output = rec.get("input_output") or []
        if not input_output:
            print(f"  [{i}] FAIL — missing input_output for LeetCode: {problem_id}")
            failed += 1
            continue

        test_cases = extract_leetcode_testcases(input_output)
        if not test_cases:
            print(f"  [{i}] FAIL — could not parse LeetCode input_output: {problem_id}")
            failed += 1
            continue

        # save — include entry_point and prompt so execution gate can use them
        save_json(out_path, {
            "problem_id": problem_id,
            "entry_point": entry_point,    # used by execution harness
            "prompt": prompt,              # used by execution harness (helpers)
            "test_cases": test_cases[:3],  # exactly 3
            "test_cases_source": "leetcode_dataset",
        })
        saved += 1

        if i % 20 == 0:
            print(
                f"Progress: {i}/{len(records)} — "
                f"saved:{saved} skipped:{skipped} failed:{failed}"
            )

        time.sleep(0.05)

    return saved, skipped, failed


if __name__ == "__main__":
    TEST_CASES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating test cases from LeetCode input_output only")

    source_files = [
        RAW_DIR / "leetcode_structured.jsonl",
    ]

    all_records = []
    for src in source_files:
        if not src.exists():
            print(f"Skipping (not found): {src.name}")
            continue
        loaded = load_jsonl(src)
        print(f"Loaded {len(loaded)} from {src.name}")
        all_records.extend(loaded)

    print(f"\nTotal records: {len(all_records)}")
    saved, skipped, failed = process_records(all_records)
    print(f"\nDone — saved:{saved} skipped:{skipped} failed:{failed}")
    print(f"Test cases written to: {TEST_CASES_DIR}")
