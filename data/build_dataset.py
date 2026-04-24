"""Steps 3, 4, 5 — Validate, deduplicate, execute-gate, and split.

Pipeline order:
    1. Load LeetCode structured source
    2. Validate format + syntax
    3. Deduplicate
    4. Execution gate — runs each solution against its test cases
    5. Split into train/val/test with seed 42

Only execution-passed records enter the split files.

Inputs:
    data/raw/leetcode_structured.jsonl
    data/test_cases/{problem_id}.json         (must exist per record)

Outputs:
    data/processed/train.json
    data/processed/val.json
    data/processed/test.json
    data/processed/filtering_log.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (
    build_text_field,
    extract_code,
    has_required_headers,
    is_valid_python,
    load_json,
    load_jsonl,
    normalize_whitespace,
    run_all_testcases,
    save_json,
    validate_testcase_file,
)


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TEST_CASES_DIR = ROOT / "data" / "test_cases"

DEDUP_THRESHOLD = 0.85
SEED = 42
MIN_FINAL_COUNT = 50  # fail-fast threshold


# ---------------------------------------------------------------------------
# Stage 1 — format + syntax validation
# ---------------------------------------------------------------------------

def validate(records: list[dict]) -> tuple[list[dict], dict]:
    valid = []
    log = {"missing_headers": 0, "bad_syntax": 0, "empty_input": 0}

    for rec in records:
        if not rec.get("input", "").strip():
            log["empty_input"] += 1
            continue
        if not has_required_headers(rec.get("output", "")):
            log["missing_headers"] += 1
            continue
        code = extract_code(rec["output"])
        if not code or not is_valid_python(code):
            log["bad_syntax"] += 1
            continue
        valid.append(rec)

    return valid, log


# ---------------------------------------------------------------------------
# Stage 2 — deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    if not records:
        return [], 0

    texts = [normalize_whitespace(r["input"]).lower() for r in records]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)

    keep, dropped = [], set()
    for i in range(len(records)):
        if i in dropped:
            continue
        keep.append(records[i])
        sims = (matrix[i] @ matrix.T).toarray()[0]
        for j in range(i + 1, len(records)):
            if sims[j] > DEDUP_THRESHOLD:
                dropped.add(j)

    return keep, len(dropped)


# ---------------------------------------------------------------------------
# Stage 3 — execution gate
# ---------------------------------------------------------------------------

def execution_gate(
    records: list[dict],
    test_cases_dir: Path,
    dry_run: bool = False,
) -> tuple[list[dict], dict]:
    passed = []
    log = {
        "missing_testcase_file": 0,
        "invalid_testcase_schema": 0,
        "execution_passed": 0,
        "execution_failed": 0,
        "execution_timeout": 0,
        "execution_runtime_error": 0,
        "execution_logic_error": 0,
    }

    limit = 20 if dry_run else len(records)

    for rec in records[:limit]:
        problem_id = rec["problem_id"]
        tc_path = test_cases_dir / f"{problem_id}.json"

        if not tc_path.exists():
            log["missing_testcase_file"] += 1
            continue

        try:
            tc_data = load_json(tc_path)
        except Exception:
            log["missing_testcase_file"] += 1
            continue

        if not validate_testcase_file(tc_data):
            log["invalid_testcase_schema"] += 1
            continue

        code = extract_code(rec["output"])
        if not code:
            log["execution_failed"] += 1
            continue

        entrypoint = tc_data.get("entry_point") or tc_data.get("entrypoint_name")
        if not entrypoint:
            log["execution_failed"] += 1
            continue

        result = run_all_testcases(
            code=code,
            entry_point=entrypoint,
            test_cases=tc_data["test_cases"],
            prompt=tc_data.get("prompt", ""),   # passes ListNode/TreeNode helpers
        )

        if result["passed"]:
            log["execution_passed"] += 1
            passed.append(rec)
        else:
            log["execution_failed"] += 1
            error_type = result.get("error_type", "unknown")
            if error_type == "timeout":
                log["execution_timeout"] += 1
            elif error_type == "runtime_error":
                log["execution_runtime_error"] += 1
            elif error_type == "logic_error":
                log["execution_logic_error"] += 1

    return passed, log


# ---------------------------------------------------------------------------
# Stage 4 — split
# ---------------------------------------------------------------------------

def split(records: list[dict]) -> dict[str, list[dict]]:
    data = list(records)
    random.Random(SEED).shuffle(data)
    n = len(data)
    t = int(n * 0.70)
    v = int(n * 0.15)
    return {
        "train": data[:t],
        "val": data[t:t + v],
        "test": data[t + v:],
    }


def add_text_field(records: list[dict]) -> list[dict]:
    for rec in records:
        rec["text"] = build_text_field(
            rec["problem_id"], rec["input"], rec["output"]
        )
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run execution gate on first 20 records only for fast verification.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ---- load LeetCode source --------------------------------------------
    leetcode_path = RAW_DIR / "leetcode_structured.jsonl"
    if not leetcode_path.exists():
        raise FileNotFoundError(f"Missing required source file: {leetcode_path}")

    all_records = load_jsonl(leetcode_path)
    print(f"Loaded {len(all_records)} from {leetcode_path.name}")

    print(f"\nTotal ingested: {len(all_records)}")

    # ---- stage 1: validate -----------------------------------------------
    valid, val_log = validate(all_records)
    print(f"After validation: {len(valid)} — dropped: {val_log}")

    # ---- stage 2: deduplicate --------------------------------------------
    deduped, n_dupes = deduplicate(valid)
    print(f"After dedup: {len(deduped)} — removed {n_dupes} duplicates")

    # ---- stage 3: execution gate -----------------------------------------
    if args.dry_run:
        print("\nDRY RUN — execution gate on first 20 records only")

    exec_passed, exec_log = execution_gate(
        deduped, TEST_CASES_DIR, dry_run=args.dry_run
    )
    print(f"After execution gate: {len(exec_passed)} passed")
    print(f"Execution log: {exec_log}")

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run for full pipeline.")
        raise SystemExit(0)

    # ---- fail-fast check -------------------------------------------------
    if len(exec_passed) < MIN_FINAL_COUNT:
        raise RuntimeError(
            f"Only {len(exec_passed)} records passed execution gate. "
            f"Minimum required: {MIN_FINAL_COUNT}. "
            f"Check generate_testcases.py output and execution logs."
        )

    # ---- add text field --------------------------------------------------
    exec_passed = add_text_field(exec_passed)

    # ---- stage 4: split --------------------------------------------------
    splits = split(exec_passed)
    for name, data in splits.items():
        save_json(PROCESSED_DIR / f"{name}.json", data)
        print(f"{name}: {len(data)} records → data/processed/{name}.json")

    # ---- filtering log ---------------------------------------------------
    filtering_log = {
        "total_ingested": len(all_records),
        "manual_seed_source_used": False,
        **val_log,
        "duplicates_removed": n_dupes,
        **exec_log,
        "valid_final_count": len(exec_passed),
        "split_train": len(splits["train"]),
        "split_val": len(splits["val"]),
        "split_test": len(splits["test"]),
    }
    save_json(PROCESSED_DIR / "filtering_log.json", filtering_log)
    print(f"\nFiltering log → data/processed/filtering_log.json")
    print(filtering_log)
