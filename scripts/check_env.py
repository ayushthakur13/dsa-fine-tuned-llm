"""Minimal environment sanity check for next development steps."""

import importlib.util
import sys


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "peft",
    "datasets",
    "faiss",
    "gradio",
    "fastapi",
    "pymongo",
    "nltk",
    "rouge_score",
]


def main() -> int:
    if sys.version_info < (3, 10):
        print("Python 3.10+ is required")
        return 1

    missing = [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]

    if missing:
        print("Missing imports:", ", ".join(missing))
        return 1

    print(f"Python version OK: {sys.version.split()[0]}")
    print("All required imports are available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
