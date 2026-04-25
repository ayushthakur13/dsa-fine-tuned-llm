"""Phase 4.1 — Base model zero-shot evaluation.

Loads Mistral 7B Instruct v0.2 in 4-bit, runs zero-shot inference
on the full test set (89 problems), evaluates with the Phase 3 pipeline,
and saves results + metrics.

No format instructions. Just the raw problem description.

Outputs:
    evaluation/results/base_model_outputs.json
    evaluation/results/base_model_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.runner import evaluate_batch  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"
PROCESSED_DIR = ROOT / "data" / "processed"
TEST_CASES_DIR = ROOT / "data" / "test_cases"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1


def load_test_split() -> list[dict]:
    path = PROCESSED_DIR / "test.json"
    with path.open() as f:
        return json.load(f)


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    return model, tokenizer


def build_zero_shot_prompt(problem: str) -> str:
    """Minimal prompt — just the problem, no format instructions."""
    return f"[INST] {problem.strip()} [/INST]"


def generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # decode only the new tokens, not the prompt
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_reference_reasoning(records: list[dict]) -> dict[str, str]:
    """Build reference reasoning dict for BLEU/ROUGE computation."""
    refs = {}
    for rec in records:
        output = rec.get("output", "")
        reasoning = output.split("Code:")[0].strip() if "Code:" in output else ""
        refs[rec["problem_id"]] = reasoning
    return refs


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test split...")
    test_records = load_test_split()
    print(f"Test problems: {len(test_records)}")

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # generate outputs
    print("\nGenerating zero-shot outputs...")
    raw_outputs = []
    for i, rec in enumerate(test_records):
        prompt = build_zero_shot_prompt(rec["input"])
        output = generate(model, tokenizer, prompt)
        raw_outputs.append({
            "problem_id": rec["problem_id"],
            "model_output": output,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(test_records)}] generated")

    # save raw outputs
    with (RESULTS_DIR / "base_model_outputs.json").open("w") as f:
        json.dump(raw_outputs, f, indent=2)
    print("Raw outputs saved.")

    # evaluate
    print("\nEvaluating...")
    evaluated = evaluate_batch(raw_outputs, TEST_CASES_DIR)

    # compute metrics
    ref_reasoning = build_reference_reasoning(test_records)
    metrics = compute_all_metrics(evaluated, ref_reasoning)

    # save metrics
    with (RESULTS_DIR / "base_model_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Base Model (Zero-Shot) Results ===")
    print(f"Pass@1:  {metrics['pass_at_1']}")
    print(f"BLEU:    {metrics['bleu']}")
    print(f"ROUGE-L: {metrics['rouge_l']}")
    print(f"Passed:  {metrics['passed']} / {metrics['total_problems']}")
    print(f"Error breakdown: {metrics['error_breakdown']}")
    print("\nResults saved to evaluation/results/base_model_metrics.json")