# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/swebench_docker/evaluate_instance.py

from __future__ import annotations
import base64
import json
import logging
import os
import re
import subprocess
import sys
import pickle

# install coverage
try:
    # Run the pip command to install the coverage package quietly
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "coverage", "--quiet"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("'coverage' package installed successfully.")

except subprocess.CalledProcessError as e:
    print(f"Failed to install 'coverage': {e.stderr.decode().strip()}")

# install pytest
try:
    # Run the pip command to install the coverage package quietly
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "pytest", "--quiet"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("'pytest' package installed successfully.")

except subprocess.CalledProcessError as e:
    print(f"Failed to install 'coverage': {e.stderr.decode().strip()}")

from typing import Any, Dict, List, Optional, Tuple
from coverage import CoverageData

from swebench_docker.constants import (
    KEY_BASELINES,
    KEY_MODEL,
    KEY_ID,
    KEY_INSTANCE_ID,
    KEY_PREDICTIONS,
    KEY_TEST_FILE_PATH,
    MAP_REPO_TO_TEST_FRAMEWORK,
    SETTING_PROMPT_MAP,
    TESTS_CONFIG,
    TESTS_FAILED,
    UNFILTERED_TESTS_FAILED,
    UNFILTERED_TESTS_PASSED,
    PatchType,
)
from swebench_docker.context_manager import TaskEnvContextManager
from swebench_docker.swebench_utils import get_test_directives

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("evaluate_instance")

import ast
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class BlockSpan:
    kind: str  # "class" | "function" | "async_function"
    qualname: str  # e.g., "MyClass.method(x, y=?)", "helper(a)", "Outer.Inner"
    start: int  # inclusive (1-based)
    end: int  # inclusive (1-based)
    depth: int  # number of name components


def _node_span(node: ast.AST) -> Tuple[int, int]:
    """Compute [start, end] lines (1-based, inclusive) for class/def/async def nodes.
    Includes decorator lines if present. Requires Python 3.8+ for end_lineno.
    """
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)

    # Include decorators (if any)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for dec in getattr(node, "decorator_list", []):
            dln = getattr(dec, "lineno", None)
            if isinstance(dln, int):
                start = min(start, dln) if isinstance(start, int) else dln

    # Fallback if end_lineno missing
    if not isinstance(end, int):
        end = int(start) if isinstance(start, int) else 1
        for child in ast.walk(node):
            e = getattr(child, "end_lineno", None)
            if isinstance(e, int):
                end = max(end, e)

    return int(start), int(end)


def _arg_name(a: ast.arg) -> str:
    # Just the identifier; we ignore annotations in the signature key
    return a.arg


def _normalize_signature(args: ast.arguments) -> str:
    """Return a normalized signature string that distinguishes argument shapes.

    - Shows positional-only params and "/" if any (PEP 570)
    - Shows var-positional as "*name" if present, else lone "*" before kw-only (PEP 3102)
    - Shows kw-only params (after "*" marker or *var)
    - Shows var-keyword as "**name"
    - Marks presence of a default with '=?' (without revealing default value)
    """
    parts: List[str] = []

    # Positional-only
    posonly = getattr(args, "posonlyargs", [])
    for a in posonly:
        parts.append(_arg_name(a))
    if posonly:
        parts.append("/")  # marker after the last pos-only

    # Positional-or-keyword (args.args) with defaults
    # Defaults align to the last N of args.args
    defaults = list(args.defaults or [])
    n_args = len(args.args)
    n_def = len(defaults)
    def_start = n_args - n_def
    for i, a in enumerate(args.args):
        name = _arg_name(a)
        if i >= def_start:
            parts.append(f"{name}=?")
        else:
            parts.append(name)

    # Var-positional
    if args.vararg is not None:
        parts.append(f"*{_arg_name(args.vararg)}")
        star_already = True
    else:
        star_already = False

    # Kw-only (with defaults in kw_defaults)
    for name, dflt in zip(args.kwonlyargs or [], args.kw_defaults or []):
        n = _arg_name(name)
        if dflt is None:
            parts.append(n)
        else:
            parts.append(f"{n}=?")

    # If there are kw-only args but no *vararg, we need a bare "*" marker in front
    if (args.kwonlyargs or []) and not star_already:
        # Insert "*" before the first kw-only item; find its index
        first_kw_idx = 0
        # find where kw-only params started (after pos params)
        # We placed posonly + "/" (optional) + args.args
        # So kw-only start index is len(parts) - len(kwonlyargs)
        first_kw_idx = len(parts) - len(args.kwonlyargs)
        parts.insert(first_kw_idx, "*")

    # Var-keyword
    if args.kwarg is not None:
        parts.append(f"**{_arg_name(args.kwarg)}")

    return "(" + ", ".join(parts) + ")"


def detect_loops(script_code: str):
    """
    Detects all 'for' and 'while' loops in the given Python script.

    Args:
        script_code (str): The Python source code as a string.

    Returns:
        dict: A dictionary with 'for' and 'while' as keys and lists of line numbers as values.
    """
    tree = ast.parse(script_code)
    for_lines = []
    while_lines = []

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            for_lines.append(node.lineno)
        elif isinstance(node, ast.While):
            while_lines.append(node.lineno)

    return {"for": sorted(for_lines), "while": sorted(while_lines)}


def _display_name(n: ast.AST) -> Optional[str]:
    """Return display name for a block:
    - class: its name
    - function/async function: name + normalized signature
    """
    if isinstance(n, ast.ClassDef):
        return n.name
    if isinstance(n, ast.FunctionDef):
        return f"{n.name}{_normalize_signature(n.args)}"
    if isinstance(n, ast.AsyncFunctionDef):
        return f"{n.name}{_normalize_signature(n.args)}"
    return None


def _walk_blocks(source: str) -> List[BlockSpan]:
    tree = ast.parse(source)
    blocks: List[BlockSpan] = []
    stack: List[str] = []

    def visit(n: ast.AST):
        disp = _display_name(n)
        kind: Optional[str] = None
        if isinstance(n, ast.ClassDef):
            kind = "class"
        elif isinstance(n, ast.FunctionDef):
            kind = "function"
        elif isinstance(n, ast.AsyncFunctionDef):
            kind = "async_function"

        if disp and kind:
            qual_parts = stack + [disp]
            qualname = ".".join(qual_parts)
            start, end = _node_span(n)
            blocks.append(BlockSpan(kind, qualname, start, end, len(qual_parts)))

            stack.append(disp)
            for child in ast.iter_child_nodes(n):
                visit(child)
            stack.pop()
        else:
            for child in ast.iter_child_nodes(n):
                visit(child)

    visit(tree)
    return blocks


def _find_loop_starts(source: str) -> Dict[int, str]:
    """Return {lineno: loop_type} where loop_type in {"for", "while"}.
    Includes AsyncFor as "for".
    """
    tree = ast.parse(source)
    loop_starts: Dict[int, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            ln = getattr(node, "lineno", None)
            if isinstance(ln, int):
                loop_starts[ln] = "for"
        elif isinstance(node, ast.While):
            ln = getattr(node, "lineno", None)
            if isinstance(ln, int):
                loop_starts[ln] = "while"
    return loop_starts


def classify_lines(source: str) -> Dict[int, Dict[str, object]]:
    """Return a dict keyed by line number (1-based)."""
    lines = source.splitlines()
    n = len(lines)

    # Initialize all as module
    result: Dict[int, Dict[str, object]] = {
        i
        + 1: {
            "block": "module",
            "kind": "module",
            "is_loop_start": False,
            "loop_type": None,
        }
        for i in range(n)
    }

    # Paint blocks (deeper overrides)
    blocks = _walk_blocks(source)
    blocks.sort(key=lambda b: (b.depth, b.start, -b.end))
    for b in blocks:
        for ln in range(max(1, b.start), min(n, b.end) + 1):
            cell = result[ln]
            cell["block"] = b.qualname
            cell["kind"] = b.kind

    # Mark loop starts
    loop_starts = _find_loop_starts(source)
    for ln, ltype in loop_starts.items():
        if 1 <= ln <= n:
            result[ln]["is_loop_start"] = True
            result[ln]["loop_type"] = ltype

    return result


def _end_lineno_fallback(node: ast.AST) -> int:
    """
    Best-effort fallback to compute an end line number if the node lacks
    the 'end_lineno' attribute (older Python or unusual nodes).
    We take the maximum end/line number of all descendants, or the node's
    own line number if nothing else is present.
    """
    best = getattr(node, "end_lineno", None)
    if isinstance(best, int):
        return best

    best = getattr(node, "lineno", 0)
    for child in ast.walk(node):
        end_ln = getattr(child, "end_lineno", None)
        ln = getattr(child, "lineno", None)
        if isinstance(end_ln, int):
            best = max(best, end_ln)
        elif isinstance(ln, int):
            best = max(best, ln)
    return best or 0


def _node_span(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", None) or 0
    end = getattr(node, "end_lineno", None)
    if not isinstance(end, int):
        end = _end_lineno_fallback(node)
    return start, end


def _docstring_of(node: ast.AST) -> Optional[Tuple[int, int, str]]:
    """
    If 'node' (Module, ClassDef, FunctionDef/AsyncFunctionDef) has a docstring,
    return (start_line, end_line, text). Otherwise None.
    """
    body = getattr(node, "body", None)
    if not body or not isinstance(body, list) or not body:
        return None

    first = body[0]
    # In Python 3.8+, docstring appears as ast.Expr(value=ast.Constant(str))
    if isinstance(first, ast.Expr):
        val = first.value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            s, e = _node_span(first)
            return s, e, val.value
        # Older forms (rare today): ast.Str
        if hasattr(ast, "Str") and isinstance(val, ast.Str):  # type: ignore[attr-defined]
            s, e = _node_span(first)
            return s, e, val.s  # type: ignore[attr-defined]
    return None


def _extract_decorators(node: ast.AST, source: str) -> List[str]:
    """
    Extract decorator names/expressions from a function or class definition.
    Returns a list of decorator strings as they appear in the source.
    """
    decorators = []
    decorator_list = getattr(node, "decorator_list", [])

    for dec in decorator_list:
        # Try to extract a meaningful representation
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            # Handle chained attributes like @obj.method
            parts = []
            current = dec
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            decorators.append(".".join(reversed(parts)))
        elif isinstance(dec, ast.Call):
            # Decorator with arguments like @decorator(arg)
            func = dec.func
            if isinstance(func, ast.Name):
                decorators.append(f"{func.id}(...)")
            elif isinstance(func, ast.Attribute):
                parts = []
                current = func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                decorators.append(".".join(reversed(parts)) + "(...)")
            else:
                decorators.append("(...)")
        else:
            # Fallback for complex expressions
            decorators.append("<complex>")

    return decorators


def _qualname(name_stack: List[str], leaf: Optional[str]) -> Optional[str]:
    if leaf is None:
        return None
    parts = [*name_stack, leaf] if name_stack else [leaf]
    return ".".join(parts)


class SpanCollector(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.items: List[Dict[str, Any]] = []
        self.stack: List[str] = []  # for qualified names

    def record_entity(self, kind: str, name: Optional[str], node: ast.AST):
        start, end = _node_span(node)
        decorators = _extract_decorators(node, self.source)

        entity = {
            "kind": kind,  # "class" | "function" | "async_function"
            "name": _qualname(self.stack, name) if name else None,
            "start_line": start,
            "end_line": end,
        }

        # Only add decorators field if there are decorators
        if decorators:
            entity["decorators"] = decorators

        self.items.append(entity)

    def record_docstring(
        self, owner_kind: str, owner_name: Optional[str], s: int, e: int, text: str
    ):
        self.items.append(
            {
                "kind": "docstring",
                "of": owner_kind,  # "module" | "class" | "function"
                "name": _qualname(self.stack, owner_name) if owner_name else None,
                "start_line": s,
                "end_line": e,
                "text": text,
            }
        )

    # Module
    def visit_Module(self, node: ast.Module):
        ds = _docstring_of(node)
        if ds:
            s, e, text = ds
            self.record_docstring("module", None, s, e, text)
        self.generic_visit(node)

    # Class
    def visit_ClassDef(self, node: ast.ClassDef):
        self.record_entity("class", node.name, node)
        # push for qualname of nested members
        self.stack.append(node.name)
        # class docstring
        ds = _docstring_of(node)
        if ds:
            s, e, text = ds
            self.record_docstring("class", node.name, s, e, text)
        self.generic_visit(node)
        self.stack.pop()

    # Function (sync)
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.record_entity("function", node.name, node)
        # push for nested defs
        self.stack.append(node.name)
        ds = _docstring_of(node)
        if ds:
            s, e, text = ds
            self.record_docstring("function", node.name, s, e, text)
        self.generic_visit(node)
        self.stack.pop()

    # Function (async)
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.record_entity("async_function", node.name, node)
        self.stack.append(node.name)
        ds = _docstring_of(node)
        if ds:
            s, e, text = ds
            self.record_docstring("function", node.name, s, e, text)
        self.generic_visit(node)
        self.stack.pop()


def analyze_file(code: str) -> List[Dict[str, Any]]:
    tree = ast.parse(code, filename="<string>")
    collector = SpanCollector(code)
    collector.visit(tree)
    # Sort by start_line for a stable, readable output
    collector.items.sort(
        key=lambda x: (x["start_line"], x["end_line"], x.get("kind", ""))
    )
    return collector.items


def is_name_main_check(node):
    """Check if node is: if __name__ == "__main__": or similar"""
    if not isinstance(node, ast.If):
        return False

    test = node.test

    # Check for: __name__ == "__main__"
    if isinstance(test, ast.Compare):
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True

        # Check for: "__main__" == __name__
        if (
            isinstance(test.left, ast.Constant)
            and test.left.value == "__main__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == "__name__"
        ):
            return True

    return False


def get_executed_lines(source_code: str):
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return []

    num_lines = len(source_code.split("\n"))
    executed_line_numbers = set()

    for node in tree.body:
        # Skip if __name__ == "__main__" blocks
        if is_name_main_check(node):
            continue

        # Skip function and class definitions entirely (including decorators)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        # Everything else at top level executes
        if node.lineno and node.end_lineno:
            for line_num in range(node.lineno, node.end_lineno + 1):
                executed_line_numbers.add(line_num)

    # Return sorted list of (line_number, line_content)
    result = []
    for line_num in sorted(executed_line_numbers):
        if line_num <= num_lines:
            result.append(line_num)

    return result


def indent_text(text, indent_level):
    return "\n".join(
        " " * indent_level + line if line.strip() else line for line in text.split("\n")
    )


def extract_preamble_classes_and_functions(code, tcm):
    class_pattern = re.compile(
        r"(^(\s*@[\w\.\(\)\', ]+\s*)*^\s*class ([\w]+)\([^)]+\):)", re.MULTILINE
    )
    # Capture methods with or without decorators
    test_method_pattern = re.compile(
        r"(^(\s*@.*\s*)*^\s*def\s+test\w+\(.*\):)", re.MULTILINE
    )

    # Capture functions with or without decorators
    test_function_pattern = re.compile(
        r"(^(\s*@.*\s*)*^\s*def\s+test\w+\(.*\):)", re.MULTILINE
    )

    preamble = ""
    classes = []
    test_functions = []

    current_position = 0

    def extract_class_body(code: str, start_index: int) -> Tuple[str, int]:
        """
        Extracts the body of a class from the given code starting from the specified index.
        Returns the class body and the end index of the class body.
        """
        if not code or start_index < 0 or start_index >= len(code):
            raise ValueError("Invalid code or start index")

        # Split the code into lines
        lines = code[start_index:].split("\n")
        class_body_lines = []

        # Find the starting indentation level of the class definition
        class_start_line = lines[0]
        start_indent = len(class_start_line) - len(class_start_line.lstrip())

        inside_multiline_comment = False
        end_index = start_index
        for i, line in enumerate(lines[1:], start=1):
            stripped_line = line.strip()
            current_indent = len(line) - len(line.lstrip())

            # Handle multiline comments or docstrings
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                if inside_multiline_comment:
                    inside_multiline_comment = False
                else:
                    inside_multiline_comment = True

            if not inside_multiline_comment:
                # Stop when we reach a line with less indentation than the class definition
                if current_indent <= start_indent and stripped_line:
                    break

            # Add lines that are part of the class body
            class_body_lines.append(line)
            # Update the end index to the current line end
            end_index = start_index + len("\n".join(lines[: i + 1])) + 1

        return code[start_index:end_index], end_index

    while current_position < len(code):
        class_match = class_pattern.search(code, current_position)
        method_match = test_function_pattern.search(code, current_position)

        if class_match and (
            not method_match or class_match.start() < method_match.start()
        ):
            class_name = class_match.group(0)
            class_body, end_idx = extract_class_body(code, class_match.end())
            current_position = end_idx

            methods = []
            class_prefix = class_name
            set_prefix = False
            for method_match in test_method_pattern.finditer(class_body):
                method_name = method_match.group()
                method_start = method_match.start()
                if not set_prefix:
                    class_prefix = class_name + class_body[:method_start]
                    set_prefix = True
                next_method = test_method_pattern.search(
                    class_body, method_start + len(method_name)
                )
                method_body = (
                    class_body[method_start : next_method.start()]
                    if next_method
                    else class_body[method_start:]
                )
                methods.append((method_name, method_body))

            if methods:
                classes.append((class_prefix, methods, class_match.start()))
            else:
                preamble += class_name + class_body

        elif method_match:
            function_name = method_match.group(0)
            start_idx = method_match.start()
            next_function = test_function_pattern.search(
                code, start_idx + len(function_name)
            )
            function_body = (
                code[start_idx : next_function.start()]
                if next_function
                else code[start_idx:]
            )
            test_functions.append((function_body, start_idx))
            current_position = method_match.end()

        else:
            break

    if classes and test_functions:
        preamble = code[: min(classes[0][2], test_functions[0][1])]
    else:
        preamble = (
            code[: classes[0][2]]
            if classes
            else code[: test_functions[0][1]] if test_functions else code
        )

    return preamble.strip(), classes, test_functions


def postprocess_tests(
    task_instance,
    preamble,
    class_name,
    methods,
    successful_tests,
    tcm,
    setting,
    translated=-1,
):
    repo = task_instance["repo"]
    django_repo = repo == "django/django"

    def needs_django_harness(preamble):
        no_django_test = "TestCase" not in preamble
        no_unittest = "unittest" not in preamble
        no_simple_test_case = "SimpleTestCase" not in preamble
        return no_django_test and no_unittest and no_simple_test_case

    if django_repo and needs_django_harness(preamble):
        preamble = "from django.test import SimpleTestCase\n" + preamble
        preamble += "\n\nclass TestsHarness(SimpleTestCase):\n"
        added_class = True
    else:
        added_class = False

    for method_name, test_case in methods:
        if django_repo and added_class:
            if "(self):" not in test_case:
                test_case = test_case.replace("():", "(self):", 1)

        class_content = f"{class_name}\n{test_case}\n"

        with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
            test_content = preamble + "\n\n" + class_content
            f.write(test_content)

        _, success = tcm.run_tests_task(
            task_instance, log_data=False, skip_mutation=True
        )

        # check if .corverage exist
        if "test_case" in setting:
            if os.path.exists(".coverage") == False:
                raise Exception("Coverage file not found")

            data = CoverageData(
                basename=".coverage",
                suffix=None,
                warn=None,
                debug=None,
            )
            data.read()
            prefix = os.getcwd()
            code_file_name = os.path.join(prefix, task_instance["code_file"])
            logger.info(f"Testing for code file: {code_file_name}")
            logger.info(f"Dir: {os.getcwd()} {os.listdir()}")
            logger.info(f"Coverage data: {data._file_map}")

            arcs = data.arcs(filename=code_file_name)
            if arcs is None:
                logger.info(f"Arcs not found")
            else:
                branches = []
                visited = []
                for e in arcs:
                    if e[0] < 0:
                        continue
                    if e[1] < 0:
                        continue
                    if e[0] in visited:
                        for i, branch in enumerate(branches):
                            if e[0] in branch:
                                branches[i].append(e[1])
                                visited.append(e[1])
                    else:
                        branches.append([e[0], e[1]])
                        visited.append(e[0])
                        visited.append(e[1])
                if translated == -1:
                    task_instance["branches"][setting] = branches
                else:
                    task_instance[f"branch_translate_{translated}"][setting] = branches

        if os.path.exists(".coverage"):
            logger.info("Removing coverage")
            os.remove(".coverage")

        if success:
            successful_tests.append((class_name, method_name, test_case))


def postprocess_functions(
    task_instance,
    preamble,
    test_functions,
    successful_tests,
    tcm,
    setting,
    translated=-1,
):
    repo = task_instance["repo"]
    django_repo = repo == "django/django"

    def needs_django_harness(preamble):
        no_django_test = "TestCase" not in preamble
        no_unittest = "unittest" not in preamble
        no_simple_test_case = "SimpleTestCase" not in preamble
        return no_django_test and no_unittest and no_simple_test_case

    added_class = False
    if django_repo and needs_django_harness(preamble):
        preamble = "from django.test import SimpleTestCase\n" + preamble
        class_wrapper_start = "\n\nclass TestsHarness(SimpleTestCase):\n"
        preamble += class_wrapper_start
        added_class = True

    class_content = ""
    for test_function, start in test_functions:
        with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
            if django_repo and added_class:
                if "(self):" not in test_function:
                    test_function = test_function.replace("():", "(self):", 1)
                test_content = preamble + "\n\n" + indent_text(test_function, 4)
            else:
                test_content = preamble + "\n\n" + test_function
            f.write(test_content)

        _, success = tcm.run_tests_task(
            task_instance, log_data=False, skip_mutation=True
        )

        if "test_case" in setting:
            if os.path.exists(".coverage") == False:
                raise Exception("Coverage file not found")

            data = CoverageData(
                basename=".coverage",
                suffix=None,
                warn=None,
                debug=None,
            )
            data.read()
            prefix = os.getcwd()
            code_file_name = os.path.join(prefix, task_instance["code_file"])
            logger.info(f"Testing for code file: {code_file_name}")
            logger.info(f"Dir: {os.getcwd()} {os.listdir()}")
            logger.info(f"Coverage data: {data._file_map}")

            arcs = data.arcs(filename=code_file_name)
            if arcs is None:
                logger.info(f"\n\nArcs not found\n\n")
            else:
                branches = []
                visited = []
                for e in arcs:
                    if e[0] < 0:
                        continue
                    if e[1] < 0:
                        continue
                    if e[0] in visited:
                        for i, branch in enumerate(branches):
                            if e[0] in branch:
                                branches[i].append(e[1])
                                visited.append(e[1])
                    else:
                        branches.append([e[0], e[1]])
                        visited.append(e[0])
                        visited.append(e[1])
                if translated == -1:
                    task_instance["branches"][setting] = branches
                else:
                    task_instance[f"branch_translate_{translated}"][setting] = branches
                logger.info(f"====================== Branches: {branches}")

        if os.path.exists(".coverage"):
            logger.info("Removing coverage")
            os.remove(".coverage")

        if success:
            if django_repo and added_class:
                class_content += indent_text(test_function, 4) + "\n"
            else:
                successful_tests.append((None, test_function))

            # check if .corverage exist

    if django_repo and class_content:
        successful_tests.append((None, class_wrapper_start + class_content))


def full_processing(
    prompt_list, tcm, task_instance, tranlsated, skip_mutation, setting: str
):

    for prompt in prompt_list:
        preamble, classes, test_functions = extract_preamble_classes_and_functions(
            prompt, tcm
        )

        # store the extracted preamble, classes, and functions
        extracted_preamble = {
            "preamble": preamble,
            "classes": classes,
            "test_functions": test_functions,
        }

        with open(
            os.path.join(
                tcm.log_dir,
                f"{task_instance[KEY_ID]}_setting_{setting}_extracted.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(extracted_preamble, f)

        successful_tests = []

        if classes:
            for class_name, methods, start in classes:
                postprocess_tests(
                    task_instance,
                    preamble,
                    class_name,
                    methods,
                    successful_tests,
                    tcm,
                    setting,
                    translated=tranlsated,
                )

        if test_functions:
            postprocess_functions(
                task_instance,
                preamble,
                test_functions,
                successful_tests,
                tcm,
                setting,
                translated=tranlsated,
            )

        # save task_instance
        # tmpfile_path = tempfile.mktemp(suffix=".json")
        with open(
            os.path.join(
                tcm.log_dir, f"{task_instance[KEY_ID]}_setting_{setting}.json"
            ),
            "w",
        ) as f:
            json.dump(task_instance, f)

        tcm.log.write(f"{TESTS_CONFIG}full pred\n")
        if len(successful_tests) > 0:
            success_tests = []
            class_definitions = {}
            for item in successful_tests:
                if item[0]:  # It's a class method
                    class_def, method_name, method_content = item
                    if class_def not in class_definitions:
                        class_definitions[class_def] = [method_content]
                    else:
                        class_definitions[class_def].append(method_content)
                else:  # It's a standalone function
                    success_tests.append(item[1])

            for class_def, methods in class_definitions.items():
                class_content = f"{class_def}\n" + "\n".join(methods)
                success_tests.append(class_content)

            success_tests_str = "\n\n".join(success_tests)

            with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
                f.write(preamble + "\n" + success_tests_str)

            _, success = tcm.run_tests_task(task_instance, skip_mutation=skip_mutation)

            total_tests = len(test_functions) + sum(
                len(methods) for _, methods, _ in classes
            )
            if success and len(successful_tests) == total_tests:
                tcm.log.write(UNFILTERED_TESTS_PASSED)
            else:
                tcm.log.write(UNFILTERED_TESTS_FAILED)
        else:
            tcm.log.write("TestsTime: 0.0")
            tcm.log.write(TESTS_FAILED)
            tcm.log.write(UNFILTERED_TESTS_FAILED)


def completion_processing(
    prompt_list, tcm, setting, task_instance, only_baseline, skip_mutation
):
    i = 0
    for prompt_ind in range(len(prompt_list)):
        prompt = prompt_list[prompt_ind]
        skip_prompt = False
        tcm.log.write(
            f"{TESTS_CONFIG}{setting} {'baseline' if only_baseline else 'pred'}\n"
        )
        if only_baseline:
            with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
                f.write(prompt)
        else:
            file_content = task_instance["preds_context"][SETTING_PROMPT_MAP[setting]]

            full_prompt = file_content + "\n" + prompt

            if (
                "assert" not in prompt
                and ".raises" not in prompt
                and "Error" not in prompt
            ) or "def" not in prompt:
                skip_prompt = True
            else:
                with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
                    f.write(full_prompt)

        if not skip_prompt:
            tcm.run_tests_task(task_instance, skip_mutation=True)
        else:
            tcm.log.write(TESTS_FAILED)


def test_case_processing(
    prompt_list, tcm, task_instance, translated, skip_mutation, setting: str
):
    successful_tests = []
    for prompt in prompt_list:
        with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
            test_content = prompt
            f.write(test_content)

        _, success = tcm.run_tests_task(
            task_instance, log_data=False, skip_mutation=True
        )

        # check if .corverage exist
        if os.path.exists(".coverage") == False:
            raise Exception("Coverage file not found")

        data = CoverageData(
            basename=".coverage",
            suffix=None,
            warn=None,
            debug=None,
        )
        data.read()
        prefix = os.getcwd()
        code_file_name = os.path.join(prefix, task_instance["code_file"])
        logger.info(f"Testing for code file: {code_file_name}")
        logger.info(f"Dir: {os.getcwd()} {os.listdir()}")
        logger.info(f"Coverage data: {data._file_map}")

        arcs = data.arcs(filename=code_file_name)
        if arcs is None:
            logger.info(f"\n\nArcs not found\n\n")
        else:
            init_lines = get_executed_lines(source_code=task_instance["code_src"])
            res = analyze_file(code=task_instance["code_src"])
            line_exclude = []
            for l in res:
                if l["kind"] == "docstring":
                    line_exclude.append(l["start_line"])
                    line_exclude.append(l["end_line"])
                elif l["kind"] == "class":
                    line_exclude.append(l["start_line"])
                elif (l["kind"] == "function") or (l["kind"] == "async_function"):
                    print(l)
                    if "decorators" in l.keys():
                        i = 1
                        for decor in l["decorators"]:
                            line_exclude.append(l["start_line"] - i)
                            i += 1
                    line_exclude.append(l["start_line"])

            clean_arcs = []
            start_arcs = []

            for arc in arcs:
                if arc[1] in line_exclude:
                    continue
                if arc[0] < 0:
                    if arc[1] == -1 * arc[0]:
                        continue
                    else:
                        start_arcs.append((-1 * arc[0], arc[1]))
                elif arc[0] in line_exclude:
                    if arc[1] in line_exclude:
                        continue
                    elif arc[1] < 0:
                        continue
                    else:
                        clean_arcs.append(arc)

                else:
                    clean_arcs.append(arc)

            clean_arcs = sorted(clean_arcs)
            init_arcs = []
            remain_arcs = []
            for i, arc in enumerate(clean_arcs):
                if arc[0] in init_lines and arc[1] in init_lines:
                    init_arcs.append(arc)
                else:
                    remain_arcs.append(arc)

            init_arcs = sorted(init_arcs)
            rows = classify_lines(source=task_instance["code_src"])

            branches = []

            init_branch = []
            for arc in init_arcs:
                if arc[0] not in init_branch:
                    init_branch.append(arc[0])
                if arc[1] not in init_branch:
                    init_branch.append(arc[1])

            remain_arcs = sorted(remain_arcs)
            branch = []
            seen_loop = []
            current_arc = ""

            for arc in remain_arcs:
                arc_0 = arc[0] if arc[0] > 0 else -1 * arc[0]
                new_arc = rows[arc_0]["block"]

                if current_arc == "":
                    current_arc = new_arc
                else:
                    if new_arc != current_arc:
                        # end of branch
                        branches.append(branch)
                        branch = []
                        current_arc = new_arc

                if arc[1] < 0:

                    if arc[0] == -1 * arc[1]:
                        continue

                    if arc[0] not in branch:
                        branch.append(arc[0])

                    if branch[0] != -1 * arc[1]:
                        branch = [-1 * arc[1]] + branch
                    if ("is_loop_start" in rows[arc[0]].keys()) and (
                        rows[arc[0]]["is_loop_start"] == True
                    ):
                        if arc[0] not in seen_loop:
                            seen_loop.append(arc[0])
                else:
                    if arc[0] in branch:
                        branch.append(arc[1])
                    else:
                        branch.append(arc[0])
                        branch.append(arc[1])

            if len(branches) > 0:
                branches = [init_branch] + branches

            if translated == -1:
                task_instance["branches"][setting] = branches
                task_instance["arcs"][setting] = arcs
            else:
                task_instance[f"branch_translate_{translated}"][setting] = branches

            if os.path.exists(".coverage"):
                logger.info("Removing coverage")
                os.remove(".coverage")

        if success:
            successful_tests.append(prompt)

    with open(
        os.path.join(tcm.log_dir, f"{task_instance[KEY_ID]}_setting_{setting}.json"),
        "w",
    ) as f:
        json.dump(task_instance, f)

    tcm.log.write(f"{TESTS_CONFIG}full pred\n")
    if len(successful_tests) > 0:
        success_tests = []
        class_definitions = {}
        for item in successful_tests:
            success_tests.append(item)

        success_tests_str = "\n\n===========================\n\n".join(success_tests)

        with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
            f.write(success_tests_str)

        _, success = tcm.run_tests_task(task_instance, skip_mutation=skip_mutation)

        total_tests = len(successful_tests)
        if success and len(successful_tests) == total_tests:
            tcm.log.write(UNFILTERED_TESTS_PASSED)
        else:
            tcm.log.write(UNFILTERED_TESTS_FAILED)
    else:
        tcm.log.write("TestsTime: 0.0")
        tcm.log.write(TESTS_FAILED)
        tcm.log.write(UNFILTERED_TESTS_FAILED)


def main(
    task_instance: dict,
    testbed_name: str,
    setting: str,
    repo_dir: str,
    log_dir: str,
    timeout: Optional[int],
    translated: int = -1,
    raw: int = 0,
    image_type: str = "conda",
    only_baseline: bool = False,
    skip_mutation: bool = False,
):
    logger.info(
        "Instance ID: "
        + task_instance["instance_id"]
        + "\nID: "
        + task_instance["id"]
        + "\nTestbed: "
        + testbed_name
        + "\nLog dir: "
        + log_dir
        + "\nTest case: "
        + setting
    )
    logger.info(f"Only Baseline: {only_baseline}")

    if only_baseline:
        task_instance[KEY_MODEL] = "baseline"
        test_type = MAP_REPO_TO_TEST_FRAMEWORK[task_instance["repo"]]
        test_directives = get_test_directives(task_instance)
        test_cmd = f"{test_type} {' '.join(test_directives)}"
        task_instance["test_directives"] = test_directives
        task_instance["test_cmd"] = test_cmd

    with TaskEnvContextManager(
        task_instance,
        setting,
        testbed_name,
        repo_dir,
        log_dir,
        timeout=timeout,
        mutation_timeout=3600,
        image_type=image_type,
    ) as tcm:
        test_patch = task_instance["test_patch"]
        if not tcm.apply_patch(
            task_instance["patch"], patch_type=PatchType.PATCH_GOLD.value
        ) or (
            test_patch
            and not tcm.apply_patch(test_patch, patch_type=PatchType.PATCH_TEST.value)
        ):
            logger.warning("Evaluation failed")
            sys.exit(1)

        # Make baselines a list so the loop below works
        if "test_case" not in setting:
            prompt_list = (
                [task_instance[KEY_BASELINES][setting]]
                if only_baseline
                else [task_instance[KEY_PREDICTIONS][setting]]
            )
            print(f"Running {setting} with prompt list length:", len(prompt_list))
        else:
            if translated != -1:
                prompt_list = [task_instance[f"translate_{translated}"][setting]]
            else:
                if raw == 1:
                    prompt_list = [task_instance["test_cases"][setting]["code"]]
                else:
                    prompt_list = [task_instance["test_cases"][setting]]
        if setting == "full":
            print(
                "Running full processing - with prompt list length:", len(prompt_list)
            )
            full_processing(
                prompt_list,
                tcm,
                task_instance,
                tranlsated=translated,
                skip_mutation=skip_mutation,
                setting=setting,
            )
        elif "test_case" in setting:
            test_case_processing(
                prompt_list=prompt_list,
                tcm=tcm,
                task_instance=task_instance,
                skip_mutation=skip_mutation,
                setting=setting,
                translated=translated,
            )
        elif "eval_branch" in setting:
            prompt_list = task_instance[KEY_PREDICTIONS][setting]
            test_case_processing(
                prompt_list=prompt_list,
                tcm=tcm,
                task_instance=task_instance,
                skip_mutation=skip_mutation,
                setting=setting,
                translated=translated,
            )
        else:
            completion_processing(
                prompt_list, tcm, setting, task_instance, only_baseline, skip_mutation
            )

        logger.info("Evaluation succeeded")


if __name__ == "__main__":
    # os.chmod("/home/swe-bench/task_instance.json", 0o777)
    TASK_INSTANCE_JSON = "/home/swe-bench/task_instance.json"
    if os.path.exists(TASK_INSTANCE_JSON):
        with open(TASK_INSTANCE_JSON, "r") as f:
            task_instance = json.load(f)
    else:
        instance_encoded = os.getenv("INSTANCE")
        if instance_encoded is None:
            raise ValueError("INSTANCE environment variable is not set")
        task_instance = json.loads(base64.b64decode(instance_encoded).decode("utf-8"))
    log_dir = os.getenv("LOG_DIR")
    if log_dir is None:
        raise ValueError("LOG_DIR environment variable is not set")

    testbed_name = os.getenv("TESTBED_NAME")
    if testbed_name is None:
        raise ValueError("TESTBED_NAME environment variable is not set")

    repo_dir = os.getenv("REPO_DIR") if os.getenv("REPO_DIR") else os.getenv("TESTBED")
    if repo_dir is None:
        raise ValueError("REPO_DIR environment variable is not set")

    timeout = os.getenv("TIMEOUT")
    int_timeout: Optional[int] = None
    if timeout is not None:
        try:
            int_timeout = int(timeout)
        except ValueError:
            raise ValueError("TIMEOUT environment variable must be an integer or None")

    setting = os.getenv("SETTING")
    if setting is None:
        raise ValueError("SETTING environment variable is not set")

    translated = os.getenv("TRANSLATED")
    if translated is None:
        raise ValueError("TRANSLATED environment variable is not set")
    translated = int(translated)

    raw = os.getenv("RAW")
    if raw is None:
        raise ValueError("RAW environment variable is not set")
    raw = int(raw)

    main(
        task_instance=task_instance,
        testbed_name=testbed_name,
        repo_dir=repo_dir,
        log_dir=log_dir,
        timeout=int_timeout,
        translated=translated,
        raw=raw,
        setting=setting,
        image_type=os.getenv("IMAGE_TYPE", "conda"),
        only_baseline=os.getenv("ONLY_BASELINE") == "True",
        skip_mutation=os.getenv("SKIP_MUTATION") == "True",
    )
