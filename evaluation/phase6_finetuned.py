"""Phase 6 — Fine-tuned adapter evaluation.

Loads Mistral 7B Instruct v0.2 in 4-bit, attaches the fine-tuned LoRA adapter,
runs inference on the full test split, evaluates with the Phase 3 pipeline,
and saves metrics plus a final comparison report.

Outputs:
    evaluation/results/finetuned_model_outputs.json
    evaluation/results/finetuned_model_metrics.json
    evaluation/results/final_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.runner import evaluate_batch  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"
PROCESSED_DIR = ROOT / "data" / "processed"
TEST_CASES_DIR = ROOT / "data" / "test_cases"
TRAINING_CFG_PATH = ROOT / "training" / "config.yaml"

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1

SYSTEM_PROMPT = """\
You are an expert DSA problem solver. For every problem you MUST respond in exactly this format and no other:

Approach: [One sentence describing the core strategy]

Reasoning:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Code:
[Python solution only, no markdown fences, no explanation]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned adapter (Phase 6)")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model id. Defaults to training/config.yaml model.name",
    )
    parser.add_argument(
        "--adapter-id",
        type=str,
        default=None,
        help="LoRA adapter id or local adapter path. Defaults to training/config.yaml hub.hub_model_id",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Generation max_new_tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Generation temperature",
    )
    return parser.parse_args()


def load_training_config() -> dict:
    with TRAINING_CFG_PATH.open() as f:
        return yaml.safe_load(f)


def load_test_split() -> list[dict]:
    path = PROCESSED_DIR / "test.json"
    with path.open() as f:
        return json.load(f)


def load_model_and_tokenizer(base_model: str, adapter_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, adapter_id)
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def build_prompt(problem: str) -> str:
    return (
        f"[INST] {SYSTEM_PROMPT}\n\n"
        f"Solve the following DSA problem:\n\n{problem.strip()} [/INST]"
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_reference_reasoning(records: list[dict]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for rec in records:
        output = rec.get("output", "")
        reasoning = output.split("Code:")[0].strip() if "Code:" in output else ""
        refs[rec["problem_id"]] = reasoning
    return refs


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def build_comparison_report(finetuned_metrics: dict) -> dict:
    report = {
        "base": load_json_if_exists(RESULTS_DIR / "base_model_metrics.json"),
        "prompt_engineered": load_json_if_exists(RESULTS_DIR / "prompt_model_metrics.json"),
        "finetuned": finetuned_metrics,
    }

    table = []
    for name, key in [
        ("Base", "base"),
        ("Prompt-Engineered", "prompt_engineered"),
        ("Fine-Tuned", "finetuned"),
    ]:
        m = report[key]
        if m is None:
            table.append({
                "model": name,
                "pass_at_1": None,
                "bleu": None,
                "rouge_l": None,
                "note": "metrics file not found",
            })
            continue

        table.append({
            "model": name,
            "pass_at_1": m.get("pass_at_1"),
            "bleu": m.get("bleu"),
            "rouge_l": m.get("rouge_l"),
            "passed": m.get("passed"),
            "total_problems": m.get("total_problems"),
            "error_breakdown": m.get("error_breakdown"),
        })

    return {
        "comparison_table": table,
        "raw": report,
    }


if __name__ == "__main__":
    args = parse_args()
    cfg = load_training_config()

    base_model = args.base_model or cfg["model"]["name"]
    adapter_id = args.adapter_id or cfg["hub"]["hub_model_id"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test split...")
    test_records = load_test_split()
    print(f"Test problems: {len(test_records)}")

    print("Loading base model + adapter...")
    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_id}")
    model, tokenizer = load_model_and_tokenizer(base_model, adapter_id)
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nGenerating fine-tuned outputs...")
    raw_outputs = []
    for i, rec in enumerate(test_records):
        prompt = build_prompt(rec["input"])
        output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        raw_outputs.append({
            "problem_id": rec["problem_id"],
            "model_output": output,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(test_records)}] generated")

    with (RESULTS_DIR / "finetuned_model_outputs.json").open("w") as f:
        json.dump(raw_outputs, f, indent=2)
    print("Raw outputs saved.")

    print("\nEvaluating...")
    evaluated = evaluate_batch(raw_outputs, TEST_CASES_DIR)

    ref_reasoning = build_reference_reasoning(test_records)
    metrics = compute_all_metrics(evaluated, ref_reasoning)

    with (RESULTS_DIR / "finetuned_model_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    comparison = build_comparison_report(metrics)
    with (RESULTS_DIR / "final_comparison.json").open("w") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== Fine-Tuned Model Results ===")
    print(f"Pass@1:  {metrics['pass_at_1']}")
    print(f"BLEU:    {metrics['bleu']}")
    print(f"ROUGE-L: {metrics['rouge_l']}")
    print(f"Passed:  {metrics['passed']} / {metrics['total_problems']}")
    print(f"Error breakdown: {metrics['error_breakdown']}")

    print("\nResults saved to:")
    print("  evaluation/results/finetuned_model_outputs.json")
    print("  evaluation/results/finetuned_model_metrics.json")
    print("  evaluation/results/final_comparison.json")
