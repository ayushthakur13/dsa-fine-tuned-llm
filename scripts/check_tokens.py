"""Token presence check for HF and Groq credentials."""

import os


REQUIRED_KEYS = ["HF_TOKEN", "GROQ_API_KEY"]


def main() -> int:
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        print("Missing env vars:", ", ".join(missing))
        print("Tip: run `set -a; source .env; set +a` before this check.")
        return 1

    print("All required tokens are present in the environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
