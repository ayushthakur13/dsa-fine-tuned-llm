"""Phase 7 — Gradio frontend.

Connects to FastAPI backend at localhost:8000.
Provides:
- Single model generation with approach/reasoning/code tabs
- Side-by-side model comparison
- Metrics panel with final results
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent
GENERATE_TIMEOUT_SECONDS = 3600
COMPARE_TIMEOUT_SECONDS = 3600


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def call_generate(problem: str, model_variant: str) -> dict:
    try:
        resp = requests.post(
            f"{API_URL}/generate",
            json={"problem": problem, "model_variant": model_variant},
            timeout=GENERATE_TIMEOUT_SECONDS,
        )
        if not resp.ok:
            return {"error": f"Backend {resp.status_code}: {resp.text[:500]}"}
        try:
            return resp.json()
        except ValueError:
            return {"error": f"Backend returned non-JSON response: {resp.text[:500]}"}
    except requests.exceptions.ReadTimeout:
        return {
            "error": (
                "Backend timed out while loading model weights. "
                "Run POST /warmup once and retry."
            )
        }
    except Exception as e:
        return {"error": str(e)}


def call_compare(problem: str) -> dict:
    try:
        resp = requests.post(
            f"{API_URL}/compare",
            json={"problem": problem},
            timeout=COMPARE_TIMEOUT_SECONDS,
        )
        if not resp.ok:
            return {"error": f"Backend {resp.status_code}: {resp.text[:500]}"}
        try:
            return resp.json()
        except ValueError:
            return {"error": f"Backend returned non-JSON response: {resp.text[:500]}"}
    except requests.exceptions.ReadTimeout:
        return {
            "error": (
                "Backend timed out while loading model weights. "
                "Run POST /warmup once and retry."
            )
        }
    except Exception as e:
        return {"error": str(e)}


def load_metrics() -> dict:
    path = ROOT / "evaluation" / "results" / "final_comparison.json"
    if path.exists():
        return json.load(open(path))
    return {}


# ---------------------------------------------------------------------------
# Tab 1 — Single model generation
# ---------------------------------------------------------------------------

def generate_single(problem: str, model_variant: str):
    if not problem.strip():
        return "Please enter a problem.", "", "", ""

    result = call_generate(problem, model_variant)

    if "error" in result:
        return f"Error: {result['error']}", "", "", ""

    approach  = result.get("approach", "Not found")
    reasoning = result.get("reasoning", "Not found")
    code      = result.get("code", "Not found")
    latency   = result.get("latency_seconds", "?")
    info      = f"Model: {model_variant} | Latency: {latency}s"

    return info, approach, reasoning, code


# ---------------------------------------------------------------------------
# Tab 2 — Side-by-side comparison
# ---------------------------------------------------------------------------

def generate_compare(problem: str):
    if not problem.strip():
        empty = "Please enter a problem."
        return empty, empty, empty, empty, empty, empty

    result = call_compare(problem)

    if "error" in result:
        err = f"Error: {result['error']}"
        return err, err, err, err, err, err

    def fmt(variant):
        r = result.get(variant, {})
        approach  = r.get("approach", "—")
        reasoning = r.get("reasoning", "—")
        code      = r.get("code", "—")
        latency   = r.get("latency_seconds", "?")
        return (
            f"**Approach:** {approach}\n\n"
            f"**Reasoning:**\n{reasoning}\n\n"
            f"**Latency:** {latency}s"
        ), code

    base_text,    base_code    = fmt("base")
    prompt_text,  prompt_code  = fmt("prompt")
    ft_text,      ft_code      = fmt("finetuned")

    return base_text, base_code, prompt_text, prompt_code, ft_text, ft_code


# ---------------------------------------------------------------------------
# Tab 3 — Metrics panel
# ---------------------------------------------------------------------------

def get_metrics_table():
    metrics = load_metrics()
    if not metrics:
        return "No metrics found. Run Phase 6 evaluation first."

    rows = []
    mapping = {
        "base_model": "Base Model",
        "prompt_engineered": "Prompt-Engineered",
        "fine_tuned": "Fine-Tuned (QLoRA)",
    }
    for key, label in mapping.items():
        m = metrics.get(key, {})
        rows.append([
            label,
            m.get("pass_at_1", "—"),
            m.get("bleu", "—"),
            m.get("rouge_l", "—"),
            m.get("passed", "—"),
            m.get("total_problems", "—"),
        ])
    return rows


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="DSA Solver — Fine-Tuned LLM") as demo:
    gr.Markdown("# DSA Solver — Fine-Tuned LLM\nCSE3720 · Generative AI and LLMs · End-Term Project")
    gr.Markdown(f"Backend API: {API_URL}")

    with gr.Tab("Generate"):
        gr.Markdown("### Single Model Generation")
        with gr.Row():
            problem_input = gr.Textbox(
                label="DSA Problem",
                placeholder="Enter a DSA problem description...",
                lines=5,
            )
            model_selector = gr.Dropdown(
                choices=["base", "prompt", "finetuned"],
                value="finetuned",
                label="Model Variant",
            )
        generate_btn = gr.Button("Generate", variant="primary")
        info_output = gr.Textbox(label="Info", interactive=False)
        with gr.Tabs():
            with gr.Tab("Approach"):
                approach_output = gr.Textbox(label="Approach", lines=3, interactive=False)
            with gr.Tab("Reasoning"):
                reasoning_output = gr.Textbox(label="Reasoning", lines=8, interactive=False)
            with gr.Tab("Code"):
                code_output = gr.Code(label="Code", language="python")

        generate_btn.click(
            fn=generate_single,
            inputs=[problem_input, model_selector],
            outputs=[info_output, approach_output, reasoning_output, code_output],
        )

    with gr.Tab("Compare Models"):
        gr.Markdown("### Side-by-Side Model Comparison")
        compare_input = gr.Textbox(
            label="DSA Problem",
            placeholder="Enter a DSA problem to compare all models...",
            lines=5,
        )
        compare_btn = gr.Button("Compare All Models", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Base Model (Zero-Shot)**")
                base_out   = gr.Markdown()
                base_code  = gr.Code(language="python", label="Code")
            with gr.Column():
                gr.Markdown("**Prompt-Engineered**")
                prompt_out  = gr.Markdown()
                prompt_code = gr.Code(language="python", label="Code")
            with gr.Column():
                gr.Markdown("**Fine-Tuned (QLoRA)**")
                ft_out   = gr.Markdown()
                ft_code  = gr.Code(language="python", label="Code")

        compare_btn.click(
            fn=generate_compare,
            inputs=[compare_input],
            outputs=[base_out, base_code, prompt_out, prompt_code, ft_out, ft_code],
        )

    with gr.Tab("Metrics"):
        gr.Markdown("### Evaluation Results — Test Set (89 problems)")
        metrics_table = gr.Dataframe(
            headers=["Model", "Pass@1", "BLEU", "ROUGE-L", "Passed", "Total"],
            value=get_metrics_table(),
            interactive=False,
        )
        gr.Markdown("""
        **Error breakdown (Fine-Tuned):**
        - Runtime errors: 47 — method name hallucination
        - Logic errors: 28 — wrong algorithm
        - Syntax errors: 6 — reduced from 25 by fine-tuning
        - Formatting failures: 2 — reduced from 89 by fine-tuning
        """)

    with gr.Tab("Logs"):
        gr.Markdown("### Recent Inference Logs")
        refresh_btn = gr.Button("Refresh Logs")
        logs_output = gr.JSON(label="Logs")

        def fetch_logs():
            try:
                resp = requests.get(f"{API_URL}/logs", timeout=10)
                return resp.json()
            except Exception as e:
                return {"error": str(e)}

        refresh_btn.click(fn=fetch_logs, outputs=[logs_output])


if __name__ == "__main__":
    demo.launch(share=True)