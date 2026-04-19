"""Metric helpers scaffold for Pass@1, BLEU, and ROUGE."""


def pass_at_1(passed: int, total: int) -> float:
    if total == 0:
        return 0.0
    return passed / total
