import re
import ast
from rich import console
from yapf.yapflib.yapf_api import FormatCode
from typing import Set, Dict, List, Tuple

from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from typing import Dict

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "key": "bold yellow",
        "value": "white",
    }
)

console = Console(theme=custom_theme)


def log_table(dct: Dict, name: str):

    table = Table(title=name)

    table.add_column("Property.", style="key")
    table.add_column("Values", style="value")

    for key in dct.keys():
        table.add_row(key, dct[key])
    console.log(table)


def extract_preamble_classes_and_functions(code: str) -> Tuple[str, list, list]:
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
            test_functions.append((function_name, function_body, start_idx))
            current_position = method_match.end()

        else:
            break

    if classes and test_functions:
        preamble = code[: min(classes[0][2], test_functions[0][2])]
    else:
        preamble = (
            code[: classes[0][2]]
            if classes
            else code[: test_functions[0][2]] if test_functions else code
        )

    return preamble.strip(), classes, test_functions


def indent_text(text, indent_level):
    return "\n".join(
        " " * indent_level + line if line.strip() else line for line in text.split("\n")
    )


def postprocess_tests(
    repo: str,
    preamble: str,
    class_name: str,
    methods: List[Tuple[str, str]],
    test_cases: Dict[str, str],
) -> Dict[str, str]:
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

    test_id = len(test_cases.keys())
    # print(f"Processing with {len(methods)} methods, id begins at {test_id}")
    for method_name, test_case in methods:
        if django_repo and added_class:
            if "(self):" not in test_case:
                test_case = test_case.replace("():", "(self):", 1)

        class_content = f"{class_name}\n{test_case}\n"
        test_content = preamble + "\n\n" + class_content

        try:
            trimmed_test_content = trim_test_cases(
                source_code=test_content, target=f"{class_name}.{method_name}"
            )
            trimmed_test_content = FormatCode(
                trimmed_test_content, style_config="pep8"
            )[0]
        except Exception as e:
            console.log(
                f"Error trimming test cases for {class_name}.{method_name}: {e}"
            )
            continue

        test_cases[f"test_case_{test_id}"] = trimmed_test_content
        # print(f"Added test case {test_id}")
        test_id += 1

    return test_cases


def postprocess_functions(
    repo: str,
    preamble: str,
    test_functions: List[Tuple[str, str]],
    test_cases: Dict[str, str],
) -> Dict[str, str]:
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

    test_id = len(test_cases.keys())

    class_content = ""
    for test_function_name, test_function, start in test_functions:
        if django_repo and added_class:
            if "(self):" not in test_function:
                test_function = test_function.replace("():", "(self):", 1)
            test_content = preamble + "\n\n" + indent_text(test_function, 4)
        else:
            test_content = preamble + "\n\n" + test_function

        fun_name = extract_function_names_from_code(code=test_function)
        trimmed_test_content = trim_test_cases(
            source_code=test_content, target=fun_name[0]
        )
        trimmed_test_content = FormatCode(trimmed_test_content, style_config="pep8")[0]

        # try:
        #     fun_name = extract_function_names_from_code(code=test_function)
        #     trimmed_test_content = trim_test_cases(
        #         source_code=test_content, target=fun_name[0]
        #     )
        #     trimmed_test_content = FormatCode(
        #         trimmed_test_content, style_config="pep8"
        #     )[0]
        # except Exception as e:
        #     console.log(f"Error trimming test cases for {test_function_name}: {e}")
        #     continue

        test_cases[f"test_case_{test_id}"] = trimmed_test_content
        test_id += 1

    return test_cases


class DependencyCollector(ast.NodeVisitor):
    def __init__(self):
        self.required_names: Set[str] = set()
        self.used_import_names: Set[str] = set()
        self.visited: Set[str] = set()

        self.function_defs: Dict[str, ast.FunctionDef] = {}
        self.class_defs: Dict[str, ast.ClassDef] = {}
        self.assignments: Dict[str, ast.stmt] = {}
        self.imports: List[ast.stmt] = []

        self.nodes_to_keep: List[ast.stmt] = []

    def visit_Module(self, node: ast.Module):
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function_defs[stmt.name] = stmt
            elif isinstance(stmt, ast.ClassDef):
                self.class_defs[stmt.name] = stmt
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self.imports.append(stmt)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                target_names = self._get_assign_targets(stmt)
                for name in target_names:
                    self.assignments[name] = stmt

    def resolve_dependencies(self, name: str):
        if name in self.visited:
            return
        self.visited.add(name)

        if name in self.function_defs:
            func_node = self.function_defs[name]
            self.nodes_to_keep.append(func_node)
            self._handle_decorators(func_node.decorator_list)
            self.generic_visit(func_node)

        elif name in self.class_defs:
            self._include_entire_class(name)

        elif name in self.assignments:
            assign_node = self.assignments[name]
            self.nodes_to_keep.append(assign_node)
            self.generic_visit(assign_node)

    def resolve_class_method(self, class_name: str, method_name: str):
        if class_name not in self.class_defs:
            print(f"Class '{class_name}' not found.")
            return

        class_node = self.class_defs[class_name]
        self.nodes_to_keep.append(class_node)

        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == method_name:
                    self._handle_decorators(item.decorator_list)
                    self.generic_visit(item)

    def _include_entire_class(self, class_name: str):
        class_node = self.class_defs[class_name]
        if class_node not in self.nodes_to_keep:
            self.nodes_to_keep.append(class_node)
            self.generic_visit(class_node)

    def _handle_decorators(self, decorators):
        for decorator in decorators:
            if isinstance(decorator, ast.Name):
                self.resolve_dependencies(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                self.visit(decorator)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.required_names.add(node.id)
            self.used_import_names.add(node.id)
            self.resolve_dependencies(node.id)

    def visit_Attribute(self, node):
        self.visit(node.value)

    def visit_Call(self, node):
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def _get_assign_targets(self, node):
        targets = []
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets.append(node.target.id)
        return targets

    def collect(self, source_code: str):
        # console.log("Here's the source code:")
        # console.log(source_code)
        # console.log("=" * 100)
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            console.log(f"Syntax error: {e}")
            return None
        self.visit(tree)
        return tree

    def filter_imports(self):
        kept = []
        for imp in self.imports:
            if isinstance(imp, ast.Import):
                for alias in imp.names:
                    if alias.asname:
                        name = alias.asname
                    else:
                        name = alias.name.split(".")[0]
                    if name in self.used_import_names:
                        kept.append(imp)
                        break
            elif isinstance(imp, ast.ImportFrom):
                if imp.module is None:
                    continue
                for alias in imp.names:
                    name = alias.asname if alias.asname else alias.name
                    if name in self.used_import_names:
                        kept.append(imp)
                        break
        return kept

    def reconstruct_code(self):
        final_imports = self.filter_imports()
        needed = final_imports + sorted(
            self.nodes_to_keep, key=lambda n: getattr(n, "lineno", 0)
        )
        return "\n\n".join(ast.unparse(n) for n in needed)


class LineSliceTrimmer(ast.NodeVisitor):
    def __init__(self, target_lines: List[List[int]]):
        self.target_lines = {line for group in target_lines for line in group}
        self.keep_nodes: List[ast.stmt] = []
        self.function_defs: Dict[str, ast.FunctionDef] = {}
        self.class_defs: Dict[str, ast.ClassDef] = {}
        self.assignments: Dict[str, ast.stmt] = {}
        self.imports: List[ast.stmt] = []

        self.required_names: Set[str] = set()
        self.visited_names: Set[str] = set()

    def visit_Module(self, node: ast.Module):
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function_defs[stmt.name] = stmt
            elif isinstance(stmt, ast.ClassDef):
                self.class_defs[stmt.name] = stmt
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self.imports.append(stmt)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                for name in self._get_assign_targets(stmt):
                    self.assignments[name] = stmt

        for stmt in node.body:
            if self._stmt_in_target(stmt):
                self.keep_nodes.append(stmt)
                self.generic_visit(stmt)

    def _stmt_in_target(self, stmt: ast.stmt) -> bool:
        return any(getattr(stmt, "lineno", -1) in group for group in self.target_lines)

    def _get_assign_targets(self, node):
        targets = []
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets.append(node.target.id)
        return targets

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            name = node.id
            self.required_names.add(name)
            self._resolve_name(name)

    def visit_Attribute(self, node):
        self.visit(node.value)

    def visit_Call(self, node):
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def _resolve_name(self, name):
        if name in self.visited_names:
            return
        self.visited_names.add(name)

        if name in self.function_defs:
            func_node = self.function_defs[name]
            self.keep_nodes.append(func_node)
            self.generic_visit(func_node)

        elif name in self.class_defs:
            class_node = self.class_defs[name]
            self.keep_nodes.append(class_node)
            self.generic_visit(class_node)

        elif name in self.assignments:
            assign_node = self.assignments[name]
            self.keep_nodes.append(assign_node)
            self.generic_visit(assign_node)

    def filter_imports(self):
        used = set(self.required_names)
        filtered = []

        for imp in self.imports:
            if isinstance(imp, ast.Import):
                if any(
                    alias.asname or alias.name.split(".")[0] in used
                    for alias in imp.names
                ):
                    filtered.append(imp)
            elif isinstance(imp, ast.ImportFrom):
                if any(alias.asname or alias.name in used for alias in imp.names):
                    filtered.append(imp)

        return filtered

    def reconstruct_code(self):
        all_nodes = self.filter_imports() + sorted(
            self.keep_nodes, key=lambda n: getattr(n, "lineno", 0)
        )
        return "\n\n".join(ast.unparse(n) for n in all_nodes)


def trim_code_by_branch(source_code: str, line_groups: List[List[int]]):

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return ""

    trimmer = LineSliceTrimmer(line_groups)
    trimmer.visit(tree)
    trimmed_code = trimmer.reconstruct_code()

    return trimmed_code


def trim_test_cases(source_code, target):

    collector = DependencyCollector()
    tree = collector.collect(source_code)

    if "." in target:
        class_name, method_name = target.split(".")
        collector.resolve_class_method(class_name, method_name)
    else:
        collector.resolve_dependencies(target)

    trimmed_code = collector.reconstruct_code()
    return trimmed_code


def extract_function_names_from_code(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return []

    function_names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
    return function_names
