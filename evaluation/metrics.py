"""Metric computation — Pass@1, BLEU, ROUGE.

Pass@1 is the primary metric.
BLEU and ROUGE are computed on reasoning text only (Approach + Reasoning sections).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Pass@1
# ---------------------------------------------------------------------------

def pass_at_1(evaluated_results: list[dict[str, Any]]) -> float:
    """Compute Pass@1 across evaluated results.

    A problem passes only if ALL its test cases passed on the first attempt.
    """
    if not evaluated_results:
        return 0.0
    total = len(evaluated_results)
    passed = sum(1 for r in evaluated_results if r["passed"])
    return passed / total


# ---------------------------------------------------------------------------
# Error breakdown
# ---------------------------------------------------------------------------

def error_breakdown(evaluated_results: list[dict[str, Any]]) -> dict[str, int]:
    """Count failures by error category."""
    counts: dict[str, int] = Counter()
    for r in evaluated_results:
        if not r["passed"]:
            category = r.get("error_category") or "unknown"
            counts[category] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def compute_bleu(
    hypotheses: list[str],
    references: list[str],
) -> float:
    """Compute corpus BLEU on reasoning text pairs.

    Args:
        hypotheses: list of model-generated reasoning strings
        references:  list of reference reasoning strings (same order)
    """
    if not hypotheses or not references:
        return 0.0

    tokenized_refs = [[ref.lower().split()] for ref in references]
    tokenized_hyps = [hyp.lower().split() for hyp in hypotheses]

    smoothing = SmoothingFunction().method1
    return corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothing)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(
    hypotheses: list[str],
    references: list[str],
) -> float:
    """Compute average ROUGE-L F1 on reasoning text pairs."""
    if not hypotheses or not references:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for hyp, ref in zip(hypotheses, references):
        score = scorer.score(ref, hyp)
        scores.append(score["rougeL"].fmeasure)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Full metrics report
# ---------------------------------------------------------------------------

def compute_all_metrics(
    evaluated_results: list[dict[str, Any]],
    reference_reasoning: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute all metrics and return a summary dict.

    Args:
        evaluated_results:   output of evaluate_batch()
        reference_reasoning: optional dict of {problem_id: reasoning_text}
                             from the reference dataset. Required for BLEU/ROUGE.
    """
    p1 = pass_at_1(evaluated_results)
    errors = error_breakdown(evaluated_results)

    report: dict[str, Any] = {
        "pass_at_1": round(p1, 4),
        "total_problems": len(evaluated_results),
        "passed": sum(1 for r in evaluated_results if r["passed"]),
        "failed": sum(1 for r in evaluated_results if not r["passed"]),
        "error_breakdown": errors,
        "bleu": None,
        "rouge_l": None,
    }

    if reference_reasoning:
        hyps, refs = [], []
        for r in evaluated_results:
            pid = r["problem_id"]
            if pid in reference_reasoning and r.get("reasoning_text"):
                hyps.append(r["reasoning_text"])
                refs.append(reference_reasoning[pid])

        if hyps:
            report["bleu"] = round(compute_bleu(hyps, refs), 4)
            report["rouge_l"] = round(compute_rouge_l(hyps, refs), 4)

    return report