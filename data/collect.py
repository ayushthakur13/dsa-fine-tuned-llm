"""Step 1 — Collect raw problems and code from HuggingFace.

Outputs:
    data/raw/leetcode_raw.jsonl
"""

from __future__ import annotations
from pathlib import Path
from datasets import load_dataset
from utils import build_problem_id, save_jsonl


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def collect_leetcode() -> None:
    print("Loading LeetCode dataset...")
    ds = load_dataset("newfacade/LeetCodeDataset", split="train")

    if len(ds) > 0:
        print("LeetCode row keys:", list(ds[0].keys()))

    records = []
    for i, row in enumerate(ds):
        difficulty = str(row.get("difficulty") or "").strip().lower()
        if difficulty not in {"easy", "medium"}:
            continue

        description = str(row.get("problem_description") or "").strip()
        code = str(row.get("completion") or "").strip()
        title = str(row.get("task_id") or f"problem_{i}").strip()
        entry_point = str(row.get("entry_point") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        input_output = row.get("input_output") or []

        if not description or not code:
            continue

        records.append({
            "problem_id": build_problem_id("leetcode", title, i),
            "source": "leetcode",
            "difficulty": difficulty,
            "title": title,
            "input": description,
            "code": code,
            "entry_point": entry_point,       # e.g. "Solution().twoSum"
            "prompt": prompt,                  # helpers: ListNode, TreeNode etc.
            "input_output": input_output,      # pre-existing test cases
        })

    save_jsonl(RAW_DIR / "leetcode_raw.jsonl", records)
    print(f"LeetCode: saved {len(records)} records → data/raw/leetcode_raw.jsonl")


if __name__ == "__main__":
    collect_leetcode()
