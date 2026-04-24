"""Step 2 — Generate Approach + Reasoning for each raw record using Groq API.

Inputs:
    data/raw/leetcode_raw.jsonl

Outputs:
    data/raw/leetcode_structured.jsonl
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from utils import load_jsonl, save_jsonl, has_required_headers


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Set it in your shell or add GROQ_API_KEY=... to .env in repo root."
    )

client = Groq(api_key=GROQ_API_KEY)

PROMPT_TEMPLATE = """\
You are an expert DSA teacher. Given a Python solution for a DSA problem, write:
1. A single sentence Approach describing the core strategy
2. A numbered step-by-step Reasoning that matches the code logic exactly

Problem:
{problem}

Code:
{code}

Respond in EXACTLY this format and nothing else:
Approach: <one sentence>

Reasoning:
1. <step>
2. <step>
3. <step>"""


def call_groq(problem: str, code: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(problem=problem, code=code)
                }],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Groq error (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
    return None


def build_output(approach_reasoning: str, code: str) -> str:
    return f"{approach_reasoning}\n\nCode:\n{code.strip()}"


def process_source(input_path: Path, output_path: Path, limit: int) -> None:
    raw_records = load_jsonl(input_path)
    print(f"\nProcessing {input_path.name} — {len(raw_records)} raw records (limit={limit})")

    # resume support — skip already processed
    existing = load_jsonl(output_path)
    existing_ids = {r["problem_id"] for r in existing}
    structured = list(existing)

    for i, rec in enumerate(raw_records[:limit]):
        if rec["problem_id"] in existing_ids:
            continue

        result = call_groq(rec["input"], rec["code"])
        if result is None:
            print(f"  [{i}] SKIP — Groq returned None")
            continue

        # validate Groq actually returned both headers
        if "Approach:" not in result or "Reasoning:" not in result:
            print(f"  [{i}] SKIP — missing headers in Groq response")
            continue

        output = build_output(result, rec["code"])
        structured.append({**rec, "output": output})

        # save after every record for resume support
        save_jsonl(output_path, structured)

        if i % 20 == 0:
            print(f"  Progress: {i}/{min(limit, len(raw_records))} — total structured: {len(structured)}")

        time.sleep(0.5)  # respect Groq rate limits

    print(f"Done — {len(structured)} structured records → {output_path.name}")


LEETCODE_LIMIT = int(os.getenv("LEETCODE_LIMIT", "800"))


if __name__ == "__main__":
    print(f"Using Groq model: {MODEL_NAME}")
    print(f"LeetCode limit: {LEETCODE_LIMIT}")
    process_source(
        RAW_DIR / "leetcode_raw.jsonl",
        RAW_DIR / "leetcode_structured.jsonl",
        limit=LEETCODE_LIMIT,
    )
