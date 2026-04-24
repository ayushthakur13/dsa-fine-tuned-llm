"""Shared helpers for Phase 2 pipeline."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tempfile
import textwrap
import sys
from pathlib import Path
from typing import Any


REQUIRED_HEADERS = ("Approach:", "Reasoning:", "Code:")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def has_required_headers(output: str) -> bool:
    return all(h in output for h in REQUIRED_HEADERS)


def extract_code(output: str) -> str | None:
    if "Code:" not in output:
        return None
    return output.split("Code:", 1)[1].strip()


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_problem_id(source: str, title: str, idx: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    return f"{source}-{slug}-{idx:05d}"


def build_text_field(problem_id: str, input_text: str, output_text: str) -> str:
    """Build the full instruction-tuning string for SFTTrainer."""
    return (
        f"<s>[INST] Solve the following DSA problem:\n\n{input_text} [/INST]\n"
        f"{output_text}</s>"
    )


# ---------------------------------------------------------------------------
# Testcase schema validation
# ---------------------------------------------------------------------------

def validate_testcase_schema(tc: Any) -> bool:
    """Validate a single test case against the strict executable schema.

    Required fields:
        args           — list of positional arguments
        kwargs         — dict of keyword arguments (can be empty)
        expected_output — any JSON-serializable value
    """
    if not isinstance(tc, dict):
        return False
    if not isinstance(tc.get("args"), list):
        return False
    if not isinstance(tc.get("kwargs"), dict):
        return False
    if "expected_output" not in tc:
        return False
    return True


def validate_testcase_file(tc_data: dict[str, Any]) -> bool:
    """Validate the full test case file structure."""
    if not isinstance(tc_data, dict):
        return False
    if "problem_id" not in tc_data:
        return False
    # Backward/forward compatibility: allow either field name.
    entry = tc_data.get("entry_point") or tc_data.get("entrypoint_name")
    if not isinstance(entry, str) or not entry.strip():
        return False
    test_cases = tc_data.get("test_cases")
    if not isinstance(test_cases, list) or len(test_cases) < 3:
        return False
    return all(validate_testcase_schema(tc) for tc in test_cases)


# ---------------------------------------------------------------------------
# Subprocess execution harness
# ---------------------------------------------------------------------------

HARNESS_TEMPLATE = textwrap.dedent("""\
{prompt}

{code}

import json, sys
from collections import deque

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_linked_list(values):
    if not isinstance(values, list):
        return values
    dummy = ListNode(0)
    current = dummy
    for value in values:
        current.next = ListNode(value)
        current = current.next
    return dummy.next


def build_tree(values):
    if not isinstance(values, list) or not values:
        return values
    nodes = [None if value is None else TreeNode(value) for value in values]
    children = nodes[::-1]
    root = children.pop()
    for node in nodes:
        if node is None:
            continue
        if children:
            node.left = children.pop()
        if children:
            node.right = children.pop()
    return root


def normalize_value(name, value):
    if not isinstance(value, list):
        return value
    lower_name = name.lower()
    if any(item is None for item in value):
        return build_tree(value)
    if lower_name in {{"head", "l1", "l2", "list1", "list2", "node", "nodes", "root"}}:
        return build_linked_list(value)
    return value


def normalize_args_kwargs(args, kwargs):
    normalized_args = [normalize_value(f"arg{{index}}", value) for index, value in enumerate(args)]
    normalized_kwargs = {{key: normalize_value(key, value) for key, value in kwargs.items()}}
    return normalized_args, normalized_kwargs


def serialize_linked_list(node):
    values = []
    seen = 0
    while node is not None and seen < 1000:
        values.append(node.val)
        node = node.next
        seen += 1
    return values


def serialize_tree(root):
    if root is None:
        return None
    values = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node is None:
            values.append(None)
            continue
        values.append(node.val)
        queue.append(node.left)
        queue.append(node.right)
    while values and values[-1] is None:
        values.pop()
    return values


def serialize_output(value):
    if isinstance(value, ListNode):
        return serialize_linked_list(value)
    if isinstance(value, TreeNode):
        return serialize_tree(value)
    return value


raw = json.loads(sys.stdin.read())
entry_point = raw["entry_point"]
args        = raw.get("args", [])
kwargs      = raw.get("kwargs", {{}})
expected    = raw["expected_output"]
args, kwargs = normalize_args_kwargs(args, kwargs)

# LeetCode: class method via eval e.g. "Solution().isHappy"
# Plain function entrypoint support is kept for compatibility.
if "." in entry_point:
    result = eval(entry_point)(*args, **kwargs)
else:
    fn = globals().get(entry_point)
    if fn is None:
        raise NameError(f"entrypoint '{{entry_point}}' not found in globals")
    result = fn(*args, **kwargs)

result = serialize_output(result)
expected = serialize_output(expected)
match = result == expected
print(json.dumps({{"result": result, "expected": expected, "passed": match}}))
""")


def _classify_error(returncode: int, stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode != 0:
        if "SyntaxError" in stderr:
            return "syntax_error"
        if any(e in stderr for e in ("IndexError", "TypeError", "ValueError",
                                      "ZeroDivisionError", "AttributeError",
                                      "KeyError", "NameError")):
            return "runtime_error"
        return "runtime_error"
    return "logic_error"


def run_single_testcase(
    code: str,
    entry_point: str,
    test_case: dict[str, Any],
    prompt: str = "",
    timeout: int = 5,
) -> dict[str, Any]:
    harness = HARNESS_TEMPLATE.format(prompt=prompt, code=code)
    payload = json.dumps({
        "entry_point": entry_point,
        "args": test_case.get("args", []),
        "kwargs": test_case.get("kwargs", {}),  # populated for LeetCode
        "expected_output": test_case["expected_output"],
    })

    timed_out = False
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(harness)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        proc = type("P", (), {"returncode": 1, "stdout": "", "stderr": "timeout"})()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if timed_out or proc.returncode != 0:
        return {
            "passed": False,
            "error_type": _classify_error(proc.returncode, proc.stderr, timed_out),
            "actual": None,
            "expected": test_case["expected_output"],
        }

    try:
        output = json.loads(proc.stdout.strip())
        passed = output.get("passed", False)
        return {
            "passed": passed,
            "error_type": None if passed else "logic_error",
            "actual": output.get("result"),
            "expected": test_case["expected_output"],
        }
    except json.JSONDecodeError:
        return {
            "passed": False,
            "error_type": "runtime_error",
            "actual": None,
            "expected": test_case["expected_output"],
        }


def run_all_testcases(
    code: str,
    entry_point: str,
    test_cases: list[dict[str, Any]],
    prompt: str = "",
    timeout: int = 5,
) -> dict[str, Any]:
    results = []
    for tc in test_cases:
        result = run_single_testcase(code, entry_point, tc, prompt, timeout)
        results.append(result)
        if not result["passed"]:
            return {
                "passed": False,
                "error_type": result["error_type"],
                "results": results,
            }
    return {"passed": True, "error_type": None, "results": results}
