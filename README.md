# DSA-Solver: Fine-Tuned LLM for Structured DSA Problem Solving and Code Generation

---

# 1. 📌 Project Overview

## Objective

Build a system that improves the reliability and correctness of LLM-generated solutions for Data Structures and Algorithms (DSA) problems using:

* Parameter-Efficient Fine-Tuning (PEFT)
* Structured output generation
* Execution-based evaluation
* Optional Retrieval-Augmented Generation (RAG)

---

## Core Idea

This is NOT a chatbot.

This is:

> An LLM evaluation and improvement system that measures and enhances code generation performance using controlled experiments.

---

## Key Focus Areas

* Fine-tuning (core)
* Evaluation pipeline (most important)
* Model comparison (critical)
* RAG (enhancement only — optional)
* Gradio frontend for demonstration

---

# 2. 🧩 Problem Statement

Large Language Models often generate incorrect, incomplete, or logically flawed solutions for structured DSA problems. Prompt engineering alone does not guarantee correctness or consistency.

This project aims to improve code generation reliability by:

* Fine-tuning an LLM using PEFT techniques (QLoRA)
* Enforcing structured outputs (approach + reasoning + Python code)
* Evaluating outputs using execution-based validation (Pass@1)
* Comparing performance with baseline and prompt-engineered models

---

# 3. 🎯 Expected Outcomes

* Fine-tuned model produces more accurate and executable Python code with structured reasoning.
* Measurable improvement over base and prompt-engineered models using Pass@1 as primary metric, and BLEU/ROUGE as secondary metrics.
* Reduction in logical errors, syntax issues, and incomplete reasoning.
* Optional lightweight RAG improves performance in pattern-based problems.
* Clear identification of hallucinations and failure cases through error analysis.
* End-to-end system usable for DSA practice and evaluation, with a Gradio demo frontend.

---

# 4. ⚙️ Features

* Fine-tuning using QLoRA/LoRA for efficient adaptation of base LLM.
* Curated dataset (~600–800 problems) with structured format and proper 70/15/15 train/validation/test split.
* Baseline comparison: Base vs Prompt-Engineered vs Fine-tuned vs Fine-tuned + RAG (optional).
* Automated evaluation pipeline using Python subprocess execution and test cases.
* Pass@1 as primary correctness metric; BLEU and ROUGE as secondary reasoning-quality metrics.
* Structured output enforcement: every model response follows Approach → Reasoning → Code format.
* Error analysis system for syntax, logic, edge-case failures, and hallucinated reasoning.
* Lightweight RAG using FAISS for retrieving similar problems (enhancement only — not core).
* Logging, monitoring, and token usage tracking.
* Gradio frontend for input, output visualization, and model comparison.

---

# 5. 🏗️ System Architecture

## Layers

### 1. Data Layer

* Dataset (DSA problems + structured solutions in Python)
* Train/validation/test split (70/15/15)
* Test cases per problem (input → expected output pairs)
* Vector DB (FAISS) — for RAG only

---

### 2. Model Layer

* Base LLM: Mistral 7B or LLaMA 3 8B
* Prompt-engineered variant (same base, structured CoT prompt)
* Fine-tuned model (QLoRA via Hugging Face PEFT)
* Fine-tuned + RAG (optional enhancement)

---

### 3. Evaluation Layer (CORE)

* Code extraction from structured model output
* Python execution via subprocess with timeout
* Test case runner (input → stdout comparison)
* Pass@1 computation across test set
* BLEU/ROUGE computation on reasoning text
* Error categorization (syntax / logic / edge-case / hallucination)

---

### 4. API Layer

* POST /generate
* POST /evaluate
* POST /compare
* GET /logs

---

### 5. Frontend Layer

* Gradio interface (lightweight, quick to deploy)
* Input: problem description + model selector
* Output: Approach, Reasoning, Code, Execution result
* Model comparison view

---

# 6. 📦 Dataset Design

## Target Size

* 600–800 high-quality, structured problems (realistic and sufficient for fine-tuning a 7B model with QLoRA)

---

## Sources and Retrieval Strategy

### Primary Sources

**1. LeetCode (via neetcode.io or leetcode-dataset on HuggingFace)**

* HuggingFace dataset: `mhhmm/leetcode-solutions-python` — contains problem descriptions + Python solutions
* Covers Easy/Medium problems across arrays, strings, hashmaps, trees, graphs, DP
* Filter to Easy and Medium only; skip Hard problems (too noisy for structured reasoning)

**2. CodeSearchNet (Python subset)**

* HuggingFace dataset: `code_search_net`, language=Python
* Use the `func_documentation_string` as the problem description and `func_code_string` as the solution
* Filter to algorithmic functions only (sort, search, traverse, etc.) using keyword matching

**3. GeeksForGeeks / InterviewBit scraped summaries (manual curation)**

* For 50–100 classic DSA problems (Two Sum, Fibonacci, Binary Search, BFS/DFS, etc.), manually write structured examples or use GPT-4 API to generate structured outputs
* These serve as seed/anchor examples for quality calibration

### Supplementary

* `codealpaca` dataset on HuggingFace (subset filtered to algorithmic problems)
* GitHub repos: `keon/algorithms` (Python), `TheAlgorithms/Python` — well-documented, clean code

---

## Structured Output Format

Every record must follow this exact format:

```json
{
  "input": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
  "output": "Approach: Use a hashmap to store seen values and their indices for O(n) lookup.\n\nReasoning:\n1. Initialize an empty hashmap.\n2. Iterate through the array with index i.\n3. Compute complement = target - nums[i].\n4. If complement is in hashmap, return [hashmap[complement], i].\n5. Otherwise store nums[i] -> i in hashmap.\n\nCode:\ndef two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []"
}
```

Every record must also have an associated test case file:

```json
{
  "problem_id": "two_sum_001",
  "test_cases": [
    {"input": "nums=[2,7,11,15], target=9", "expected_output": "[0, 1]"},
    {"input": "nums=[3,2,4], target=6", "expected_output": "[1, 2]"},
    {"input": "nums=[3,3], target=6", "expected_output": "[0, 1]"}
  ]
}
```

---

## Preprocessing Steps

1. Remove duplicate problems (deduplicate on problem description similarity using cosine distance).
2. Normalize formatting: consistent section headers (`Approach:`, `Reasoning:`, `Code:`).
3. Validate Python code: run all solutions through `py_compile` to catch syntax errors before training.
4. Filter out problems with no valid test cases.
5. Ensure consistent structure across all records.

---

## Split

* Train: 70% (~420–560 problems)
* Validation: 15% (~90–120 problems) — used for hyperparameter tuning
* Test: 15% (~90–120 problems) — **strictly unseen, used only for final evaluation**

---

# 7. 🧠 Model Strategy

## Base Model

* Mistral 7B Instruct v0.2 (preferred) or LLaMA 3 8B Instruct
* Reason: Instruction-following capability, open weights, compatible with QLoRA

---

## Fine-Tuning

* Method: QLoRA (4-bit quantized base + LoRA adapters)
* Framework: Hugging Face Transformers + PEFT + bitsandbytes
* Training environment: Google Colab Pro (A100) or Kaggle (2x T4)

---

## QLoRA Configuration

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Quantization | 4-bit NF4 |
| Batch size | 4 (gradient accumulation = 4) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Max seq length | 1024 |

---

## Why QLoRA

* Memory efficient — 7B model fits in ~6–8GB VRAM with 4-bit quantization
* Faster training cycles — suitable for limited GPU hours
* Proven approach for task-specific fine-tuning on code generation tasks

---

# 8. 🔍 Prompt Engineering

All models (including the baseline prompt-engineered variant) use a structured system prompt:

```
You are an expert DSA problem solver. For every problem, you MUST respond in exactly this format:

Approach: [One sentence describing the core strategy]

Reasoning:
1. [Step 1]
2. [Step 2]
...

Code:
[Python solution only, no markdown fences]
```

The prompt-engineered baseline uses Chain-of-Thought (CoT) prompting with this template but NO fine-tuning, serving as a direct comparison point against the fine-tuned model.

---

# 9. 🧪 Evaluation Pipeline (MOST IMPORTANT)

## Structured Output Parsing

Before execution, extract the `Code:` section from the model output using a simple regex or string split on the `Code:` header. If the section is missing, classify as a formatting failure and mark as fail.

---

## Python Execution Flow

```
Model Output
    ↓
Extract Code Section
    ↓
Write to temp file (solution.py)
    ↓
Execute: subprocess.run(["python", "solution.py"], input=test_input, timeout=5)
    ↓
Capture stdout
    ↓
Compare with expected_output (strip whitespace)
    ↓
Pass / Fail
```

---

## Python Execution Example

```python
import subprocess, tempfile, os

def run_solution(code: str, test_input: str, expected: str, timeout: int = 5) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python", tmp_path],
            input=test_input, capture_output=True,
            text=True, timeout=timeout
        )
        actual = result.stdout.strip()
        passed = actual == expected.strip()
        return {"passed": passed, "actual": actual, "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"passed": False, "actual": None, "error": "Timeout"}
    except Exception as e:
        return {"passed": False, "actual": None, "error": str(e)}
    finally:
        os.unlink(tmp_path)
```

---

## Edge Case Handling

* Syntax error in extracted code → fail, log as "syntax error"
* Timeout (> 5 seconds) → fail, log as "timeout / TLE"
* Output mismatch → fail, log as "logic error"
* Missing `Code:` section → fail, log as "formatting failure"
* Runtime exception (IndexError, TypeError, etc.) → fail, log as "runtime error"

---

## Pass@1 Computation

Pass@1 = (Number of problems where first attempt passed all test cases) / (Total problems in test set)

Computed separately for each model variant to produce the comparison table.

---

# 10. 📊 Metrics

## Primary Metric

* **Pass@1** — true correctness. A problem is "passed" only if the generated code passes ALL associated test cases on the first try.

## Secondary Metrics

* **BLEU** — measures n-gram overlap between generated reasoning text and reference reasoning. Computed on the Approach + Reasoning sections only (not the code). Useful for comparing how closely the model follows structured reasoning patterns.
* **ROUGE-L** — measures longest common subsequence recall between generated and reference reasoning. Supplementary to BLEU.

> Note: BLEU and ROUGE are weak proxies for reasoning correctness. They are included to satisfy course evaluation requirements and for completeness. Pass@1 is the only metric that truly measures whether the system works.

---

# 11. 🧠 Error Analysis

Categorize all failures from the test set into:

| Category | Description | Example |
|---|---|---|
| Syntax Error | Generated code has Python syntax errors | Missing colon, wrong indentation |
| Logic Error | Code runs but produces wrong output | Off-by-one in binary search |
| Edge Case Failure | Passes main cases, fails edge cases | Empty array, single element |
| Runtime Error | Code crashes during execution | IndexError, ZeroDivisionError |
| Formatting Failure | Model did not produce a `Code:` section | Hallucinated a different format |
| Hallucinated Reasoning | Approach described does not match code | Says "use BFS" but code uses DFS |
| Timeout | Code exceeds 5-second limit | Naive O(n²) on large inputs |

Produce a failure breakdown table per model variant. Key insight to highlight:

> "Fine-tuned model reduces syntax errors and formatting failures significantly, but edge-case failures require more data or test-case-augmented training."

---

# 12. 🔄 RAG (Enhancement Only — Optional)

## Purpose

Retrieve similar solved problems from the training set and inject them as context into the prompt, helping the model recognize patterns.

## Implementation

* Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
* Vector DB: FAISS (local, no server needed)
* Top-k retrieval: 2 similar problems injected into the prompt
* Only applied at inference time — no changes to fine-tuning

## Role

* May improve Pass@1 on pattern-heavy problems (e.g., sliding window, two pointers)
* Enhancement, not core — skip if time is constrained
* If implemented, compare Fine-tuned vs Fine-tuned + RAG to measure delta

---

# 13. 🗄️ Data Storage

## Vector DB

* FAISS (local) — for RAG retrieval index

## Relational / Document DB

* MongoDB or PostgreSQL
* Store:
  * problems (id, description, test_cases)
  * model_outputs (problem_id, model_variant, raw_output, extracted_code, pass_fail, error_category)
  * metrics (model_variant, pass_at_1, bleu, rouge, timestamp)
  * logs (request_id, model_used, latency, token_count)

---

# 14. 📈 Logging & Monitoring

* Track per inference: model used, latency, input tokens, output tokens
* Aggregate metrics per model variant
* Store all raw outputs for offline error analysis

---

# 15. 💰 Cost Control

* Primary inference: local (Ollama or Hugging Face local) to avoid API costs
* Use OpenAI/Anthropic API only for base model comparison or dataset generation
* Set hard budget thresholds before any API-based generation

---

# 16. 💻 Frontend (Gradio)

## Why Gradio (Not Next.js)

* Zero setup overhead — pip install gradio, 30 lines of code
* Sufficient for demonstration purposes
* Evaluators can interact with the system without any deployment complexity
* Allows focus on the core system (evaluation pipeline and fine-tuning)

## Features

* Text input: problem description
* Dropdown: model selector (Base / Prompt / Fine-tuned / Fine-tuned + RAG)
* Output tabs: Approach | Reasoning | Code | Execution Result
* Side-by-side comparison mode
* Metrics panel: Pass@1, BLEU scores per model

---

# 17. 🧠 User Flow

1. User inputs a DSA problem description
2. Selects model variant (or runs all 4 for comparison)
3. System generates structured output (Approach + Reasoning + Code)
4. System executes the code against test cases
5. System displays Pass/Fail result and execution output
6. User views error analysis and comparative metrics

---

# 18. 🔌 Backend APIs

* POST /generate — generate structured output from selected model
* POST /evaluate — execute generated code against test cases, return pass/fail
* POST /compare — run all model variants on a problem, return side-by-side results
* GET /logs — retrieve inference and evaluation logs

---

# 19. 🧠 Key Comparisons

| Model | Purpose | Expected Pass@1 |
|---|---|---|
| Base | Baseline — no prompting | Lowest |
| Prompt-Engineered | Structured CoT prompt, no fine-tuning | Medium |
| Fine-tuned (QLoRA) | Core contribution | Highest |
| Fine-tuned + RAG | Enhancement | Slight improvement over fine-tuned |

---

# 20. ⚠️ Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Poor dataset quality | Validate all code with py_compile before training; manually review 50+ samples |
| Evaluation pipeline failures | Test pipeline on 20 problems before running full evaluation |
| QLoRA training instability | Use established configs; monitor validation loss |
| RAG adding complexity | Treat as optional; build only after core pipeline works |
| Time overrun on frontend | Gradio instead of Next.js — 30 min setup |

---

# 21. ⏱️ Time Allocation

* Dataset collection and preprocessing: 25%
* Fine-tuning: 20%
* Evaluation pipeline: 35%
* Backend APIs: 10%
* Gradio frontend: 5%
* RAG (if time permits): 5%

---

# 22. 🎯 Final Positioning

This project demonstrates:

* LLM fine-tuning using PEFT (QLoRA)
* Structured prompt engineering (Chain-of-Thought)
* Execution-based evaluation with Pass@1
* BLEU/ROUGE secondary metrics on reasoning quality
* Error analysis including hallucination and failure categorization
* Optional retrieval-augmented generation
* End-to-end system with Gradio demo

---

## Final Statement

> "We built a system to evaluate and improve LLM performance on structured DSA code generation tasks using QLoRA fine-tuning, with automated Python execution-based validation (Pass@1), structured reasoning enforcement, and optional retrieval-based enhancement — demonstrated through a Gradio interface."