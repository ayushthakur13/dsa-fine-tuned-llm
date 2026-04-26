"""Phase 7 — FastAPI backend.

Endpoints:
    POST /generate   — generate structured output from selected model
    POST /evaluate   — execute generated code against test cases
    POST /compare    — run all model variants on one problem
    GET  /logs       — retrieve recent inference logs
    GET  /health     — health check
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.runner import evaluate_problem  # noqa: E402


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DSA Solver API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory log store
# ---------------------------------------------------------------------------

LOGS: list[dict[str, Any]] = []
MAX_LOGS = 200


def add_log(entry: dict) -> None:
    LOGS.append(entry)
    if len(LOGS) > MAX_LOGS:
        LOGS.pop(0)


# ---------------------------------------------------------------------------
# Model registry — lazy loaded
# ---------------------------------------------------------------------------

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_ID = "ayushpratapsingh/dsa-fine-tuned-llm"
TEST_CASES_DIR = ROOT / "data" / "test_cases"

_models: dict[str, Any] = {}
_tokenizer = None


def get_runtime_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def preload_models() -> None:
    """Warm up tokenizer + models so first user request is fast."""
    get_tokenizer()
    get_base_model()
    get_finetuned_model()


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def get_base_model():
    if "base" not in _models:
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=get_bnb_config(),
                device_map="auto",
            )
        else:
            # CPU fallback for local runs where CUDA is unavailable.
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float32,
            )
            model.to("cpu")
        model.config.use_cache = False
        model.eval()
        _models["base"] = model
    return _models["base"]


def get_finetuned_model():
    if "finetuned" not in _models:
        base = get_base_model()
        model = PeftModel.from_pretrained(base, ADAPTER_ID)
        model.eval()
        _models["finetuned"] = model
    return _models["finetuned"]


SYSTEM_PROMPT = """\
You are an expert DSA problem solver. For every problem you MUST respond in exactly this format:

Approach: [One sentence describing the core strategy]

Reasoning:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Code:
[Python solution only, no markdown fences]"""

FEW_SHOT_EXAMPLES = []


def load_few_shot():
    global FEW_SHOT_EXAMPLES
    if FEW_SHOT_EXAMPLES:
        return
    train_path = ROOT / "data" / "processed" / "train.json"
    if train_path.exists():
        records = json.load(open(train_path))
        import random
        rng = random.Random(42)
        FEW_SHOT_EXAMPLES = rng.sample(records, min(2, len(records)))


def build_prompt(problem: str, variant: str) -> str:
    if variant == "base":
        return f"[INST] {problem.strip()} [/INST]"

    load_few_shot()
    messages = []
    messages.append(f"[INST] {SYSTEM_PROMPT} [/INST]")
    messages.append("Understood. I will always respond with Approach, Reasoning, and Code.")
    for ex in FEW_SHOT_EXAMPLES:
        messages.append(f"[INST] Solve the following DSA problem:\n\n{ex['input'].strip()} [/INST]")
        messages.append(ex["output"].strip())
    messages.append(f"[INST] Solve the following DSA problem:\n\n{problem.strip()} [/INST]")
    return "\n".join(messages)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    device = get_runtime_device()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    problem: str
    model_variant: str = "finetuned"  # base | prompt | finetuned


class GenerateResponse(BaseModel):
    request_id: str
    model_variant: str
    raw_output: str
    approach: str
    reasoning: str
    code: str
    latency_seconds: float


class EvaluateRequest(BaseModel):
    problem_id: str
    model_output: str


class CompareRequest(BaseModel):
    problem: str
    problem_id: str | None = None


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------

def parse_sections(raw: str) -> dict[str, str]:
    approach, reasoning, code = "", "", ""
    if "Approach:" in raw:
        after_approach = raw.split("Approach:", 1)[1]
        approach = after_approach.split("Reasoning:")[0].strip() if "Reasoning:" in after_approach else after_approach.strip()
    if "Reasoning:" in raw:
        after_reasoning = raw.split("Reasoning:", 1)[1]
        reasoning = after_reasoning.split("Code:")[0].strip() if "Code:" in after_reasoning else after_reasoning.strip()
    if "Code:" in raw:
        code = raw.split("Code:", 1)[1].strip()
    return {"approach": approach, "reasoning": reasoning, "code": code}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "runtime_device": get_runtime_device(),
        "tokenizer_loaded": _tokenizer is not None,
        "base_model_loaded": "base" in _models,
        "finetuned_model_loaded": "finetuned" in _models,
    }


@app.post("/warmup")
def warmup() -> dict:
    start = time.time()
    preload_models()
    latency = round(time.time() - start, 2)
    return {
        "status": "ready",
        "warmup_seconds": latency,
        "base_model_loaded": "base" in _models,
        "finetuned_model_loaded": "finetuned" in _models,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest) -> GenerateResponse:
    if req.model_variant not in ("base", "prompt", "finetuned"):
        raise HTTPException(status_code=400, detail="model_variant must be base, prompt, or finetuned")

    tokenizer = get_tokenizer()

    if req.model_variant == "finetuned":
        model = get_finetuned_model()
    else:
        model = get_base_model()

    prompt = build_prompt(req.problem, req.model_variant)

    start = time.time()
    raw_output = generate(model, tokenizer, prompt)
    latency = round(time.time() - start, 2)

    sections = parse_sections(raw_output)
    request_id = str(uuid.uuid4())[:8]

    add_log({
        "request_id": request_id,
        "model_variant": req.model_variant,
        "latency_seconds": latency,
        "timestamp": time.time(),
        "has_code": bool(sections["code"]),
    })

    return GenerateResponse(
        request_id=request_id,
        model_variant=req.model_variant,
        raw_output=raw_output,
        approach=sections["approach"],
        reasoning=sections["reasoning"],
        code=sections["code"],
        latency_seconds=latency,
    )


@app.post("/evaluate")
def evaluate_endpoint(req: EvaluateRequest) -> dict:
    if not req.problem_id:
        raise HTTPException(status_code=400, detail="problem_id is required")

    result = evaluate_problem(
        problem_id=req.problem_id,
        raw_model_output=req.model_output,
        test_cases_dir=TEST_CASES_DIR,
    )

    add_log({
        "request_id": str(uuid.uuid4())[:8],
        "type": "evaluate",
        "problem_id": req.problem_id,
        "passed": result["passed"],
        "error_category": result["error_category"],
        "timestamp": time.time(),
    })

    return result


@app.post("/compare")
def compare_endpoint(req: CompareRequest) -> dict:
    tokenizer = get_tokenizer()
    results = {}

    for variant in ("base", "prompt", "finetuned"):
        model = get_finetuned_model() if variant == "finetuned" else get_base_model()
        prompt = build_prompt(req.problem, variant)

        start = time.time()
        raw_output = generate(model, tokenizer, prompt)
        latency = round(time.time() - start, 2)

        sections = parse_sections(raw_output)
        eval_result = None

        if req.problem_id:
            eval_result = evaluate_problem(
                problem_id=req.problem_id,
                raw_model_output=raw_output,
                test_cases_dir=TEST_CASES_DIR,
            )

        results[variant] = {
            "raw_output": raw_output,
            "approach": sections["approach"],
            "reasoning": sections["reasoning"],
            "code": sections["code"],
            "latency_seconds": latency,
            "evaluation": eval_result,
        }

    add_log({
        "request_id": str(uuid.uuid4())[:8],
        "type": "compare",
        "problem_id": req.problem_id,
        "timestamp": time.time(),
    })

    return results


@app.get("/logs")
def logs_endpoint(limit: int = 20) -> dict:
    return {
        "logs": LOGS[-limit:],
        "total": len(LOGS),
    }