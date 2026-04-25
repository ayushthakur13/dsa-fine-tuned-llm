"""Phase 4.2 — Prompt-engineered model evaluation.

Same base model as phase4_baseline.py but with:
- Structured system prompt enforcing Approach / Reasoning / Code format
- 2 few-shot examples from the training set

Outputs:
    evaluation/results/prompt_model_outputs.json
    evaluation/results/prompt_model_metrics.json
"""

from __future__ import annotations

import json
import random
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
FEW_SHOT_COUNT = 2
SEED = 42


SYSTEM_PROMPT = """\
You are an expert DSA problem solver. For every problem you MUST respond in exactly this format and no other:

Approach: [One sentence describing the core strategy]

Reasoning:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Code:
[Python solution only, no markdown fences, no explanation]"""


def load_test_split() -> list[dict]:
    path = PROCESSED_DIR / "test.json"
    with path.open() as f:
        return json.load(f)


def load_train_split() -> list[dict]:
    path = PROCESSED_DIR / "train.json"
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


def get_few_shot_examples(train_records: list[dict], n: int = 2) -> list[dict]:
    """Sample n examples from training set as few-shot demonstrations."""
    rng = random.Random(SEED)
    return rng.sample(train_records, min(n, len(train_records)))


def build_few_shot_prompt(
    problem: str,
    few_shot_examples: list[dict],
) -> str:
    """Build structured prompt with system instructions and few-shot examples."""
    messages = []

    # system instruction
    messages.append(f"[INST] {SYSTEM_PROMPT} [/INST]")
    messages.append("Understood. I will always respond with Approach, Reasoning, and Code sections.")

    # few-shot examples
    for ex in few_shot_examples:
        messages.append(f"[INST] Solve the following DSA problem:\n\n{ex['input'].strip()} [/INST]")
        messages.append(ex["output"].strip())

    # actual problem
    messages.append(f"[INST] Solve the following DSA problem:\n\n{problem.strip()} [/INST]")

    return "\n".join(messages)


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
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_reference_reasoning(records: list[dict]) -> dict[str, str]:
    refs = {}
    for rec in records:
        output = rec.get("output", "")
        reasoning = output.split("Code:")[0].strip() if "Code:" in output else ""
        refs[rec["problem_id"]] = reasoning
    return refs


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits...")
    test_records = load_test_split()
    train_records = load_train_split()
    print(f"Test: {len(test_records)} | Train: {len(train_records)}")

    few_shot = get_few_shot_examples(train_records, FEW_SHOT_COUNT)
    print(f"Few-shot examples selected: {[e['problem_id'] for e in few_shot]}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # generate outputs
    print("\nGenerating prompt-engineered outputs...")
    raw_outputs = []
    for i, rec in enumerate(test_records):
        prompt = build_few_shot_prompt(rec["input"], few_shot)
        output = generate(model, tokenizer, prompt)
        raw_outputs.append({
            "problem_id": rec["problem_id"],
            "model_output": output,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(test_records)}] generated")

    # save raw outputs
    with (RESULTS_DIR / "prompt_model_outputs.json").open("w") as f:
        json.dump(raw_outputs, f, indent=2)
    print("Raw outputs saved.")

    # evaluate
    print("\nEvaluating...")
    evaluated = evaluate_batch(raw_outputs, TEST_CASES_DIR)

    # compute metrics
    ref_reasoning = build_reference_reasoning(test_records)
    metrics = compute_all_metrics(evaluated, ref_reasoning)

    # save metrics
    with (RESULTS_DIR / "prompt_model_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Prompt-Engineered Model Results ===")
    print(f"Pass@1:  {metrics['pass_at_1']}")
    print(f"BLEU:    {metrics['bleu']}")
    print(f"ROUGE-L: {metrics['rouge_l']}")
    print(f"Passed:  {metrics['passed']} / {metrics['total_problems']}")
    print(f"Error breakdown: {metrics['error_breakdown']}")
    print("\nResults saved to evaluation/results/prompt_model_metrics.json")