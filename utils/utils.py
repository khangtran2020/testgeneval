import re
import sys
import ast
from rich import console
from yapf.yapflib.yapf_api import FormatCode
from typing import Set, Dict, List, Tuple, Union

from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from typing import Dict, Optional

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


# class DependencyCollector(ast.NodeVisitor):
#     def __init__(self):
#         # Store all top-level objects
#         self.class_defs: Dict[str, List[ast.ClassDef]] = {}
#         self.func_defs: Dict[str, ast.FunctionDef] = {}
#         self.async_func_defs: Dict[str, ast.AsyncFunctionDef] = {}
#         self.assigns: Dict[str, ast.Assign] = {}
#         self.imports: List[ast.stmt] = []
#         self.used_names: Set[str] = set()
#         self.all_names: Set[str] = set()
#         self.nodes_to_keep: List[ast.stmt] = []
#         self.visited: Set[str] = set()
#         self.current_class: List[str] = []  # Stack for nested class support

#     def visit_Import(self, node):
#         self.imports.append(node)
#         for alias in node.names:
#             self.all_names.add(alias.asname or alias.name.split(".")[0])

#     def visit_ImportFrom(self, node):
#         self.imports.append(node)
#         for alias in node.names:
#             self.all_names.add(alias.asname or alias.name)

#     def visit_ClassDef(self, node):
#         self.class_defs.setdefault(node.name, []).append(node)
#         self.all_names.add(node.name)
#         self.current_class.append(node.name)
#         self.generic_visit(node)  # Now descend into methods
#         self.current_class.pop()

#     def visit_FunctionDef(self, node):
#         if self.current_class:
#             key = f"{self.current_class[-1]}.{node.name}"
#         else:
#             key = f"global.{node.name}"
#         self.func_defs[key] = node
#         self.all_names.add(key)

#     def visit_AsyncFunctionDef(self, node):
#         if self.current_class:
#             key = f"{self.current_class[-1]}.{node.name}"
#         else:
#             key = f"global.{node.name}"
#         self.async_func_defs[key] = node
#         self.all_names.add(key)

#     def visit_Assign(self, node):
#         for t in node.targets:
#             if isinstance(t, ast.Name):
#                 self.assigns[t.id] = node
#                 self.all_names.add(t.id)

#     def collect(self, tree: ast.Module):
#         self.visit(tree)

#     def find_names_in_node(self, node: ast.AST) -> Set[str]:
#         """Recursively collect all variable names used in a node."""
#         names = set()

#         class NameVisitor(ast.NodeVisitor):
#             def visit_Name(self, n):
#                 names.add(n.id)

#             def visit_Attribute(self, n):
#                 if isinstance(n.value, ast.Name):
#                     names.add(n.value.id)
#                 self.generic_visit(n)

#         NameVisitor().visit(node)
#         return names

#     def resolve(self, name: str, function: str, class_name: str):

#         if (function is not None) and (class_name is None):
#             self.resolve_function(function)
#         elif (function is None) and (class_name is not None):
#             self.resolve_class_method(class_name, name)
#         elif (name is not None) and (name in self.assigns):
#             node = self.assigns[name]
#             self.nodes_to_keep.append(node)
#             self._resolve_node(node)
#         # (Imports handled separately)
#         else:
#             pass

#     def resolve_function(self, function):
#         key = f"global.{function}"
#         if key in self.visited:
#             return
#         self.visited.add(f"global.{function}")
#         # Functions
#         if key in self.func_defs:
#             node = self.func_defs[key]
#             self.nodes_to_keep.append(node)
#             self._resolve_node(node)
#         elif key in self.async_func_defs:
#             node = self.async_func_defs[key]
#             self.nodes_to_keep.append(node)
#             self._resolve_node(node)

#     def resolve_class_method(self, class_name: str, method_name: str):
#         # Get ALL class defs with this name
#         class_nodes = self.class_defs.get(class_name, [])
#         # Search each for the target method
#         for class_node in class_nodes:
#             for item in class_node.body:
#                 if (
#                     isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
#                     and item.name == method_name
#                 ):
#                     # Found the method in this class!
#                     if class_node not in self.nodes_to_keep:
#                         self.nodes_to_keep.append(class_node)
#                         # Also resolve bases and decorators
#                         for base in class_node.bases:
#                             for n in ast.walk(base):
#                                 if isinstance(n, ast.Name):
#                                     self.resolve(n.id)
#                         for deco in class_node.decorator_list:
#                             for n in ast.walk(deco):
#                                 if isinstance(n, ast.Name):
#                                     self.resolve(n.id)
#                                 self.generic_visit(item)
#                     return

#         if class_nodes:
#             class_node = class_nodes[-1]
#             if class_node not in self.nodes_to_keep:
#                 self.nodes_to_keep.append(class_node)
#                 # Also resolve bases and decorators
#                 for base in class_node.bases:
#                     for n in ast.walk(base):
#                         if isinstance(n, ast.Name):
#                             self.resolve(n.id)
#                 for deco in class_node.decorator_list:
#                     for n in ast.walk(deco):
#                         if isinstance(n, ast.Name):
#                             self.resolve(n.id)
#                         self.generic_visit(item)

#     def _resolve_node(self, node):
#         """Given any node, resolve all referenced names recursively."""
#         for n in ast.walk(node):
#             if isinstance(n, ast.Name):
#                 # Try both global and class-qualified keys
#                 candidates = [f"global.{n.id}"] + [
#                     k for k in self.all_names if k.endswith(f".{n.id}")
#                 ]
#                 for candidate in candidates:
#                     if candidate in self.all_names:
#                         self.resolve(candidate, None, None)

#     def filter_imports(self):
#         """Only keep needed imports."""
#         needed_imports = []
#         needed_names = set()
#         for node in self.nodes_to_keep:
#             for name in self.find_names_in_node(node):
#                 needed_names.add(name)
#         for imp in self.imports:
#             if isinstance(imp, ast.Import):
#                 for alias in imp.names:
#                     if alias.asname and alias.asname in needed_names:
#                         needed_imports.append(imp)
#                         break
#                     elif alias.name.split(".")[0] in needed_names:
#                         needed_imports.append(imp)
#                         break
#             elif isinstance(imp, ast.ImportFrom):
#                 for alias in imp.names:
#                     if (alias.asname and alias.asname in needed_names) or (
#                         alias.name in needed_names
#                     ):
#                         needed_imports.append(imp)
#                         break
#         return needed_imports

#     def get_minimal_code(self):
#         imports = self.filter_imports()
#         code_blocks = sorted(
#             imports + self.nodes_to_keep, key=lambda n: getattr(n, "lineno", 0)
#         )
#         return "\n\n".join(ast.unparse(n) for n in code_blocks)


# def extract_minimal_test(script: str, target: str, id: str):
#     """
#     target:
#       - For a function: 'test_func_name'
#       - For a class method: 'ClassName|class_method_split|method_name'
#     """
#     tree = ast.parse(script)
#     collector = DependencyCollector()
#     collector.collect(tree)

#     if "|class_method_split|" in target:
#         class_name, method_name = target.split("|class_method_split|")
#         # Find the class
#         if class_name not in collector.class_defs:
#             raise ValueError(f"Class '{class_name}' not found in file.")

#         class_nodes = collector.class_defs[class_name]
#         # Find the method
#         method_node = None
#         class_node = None
#         for class_node in class_nodes:
#             for item in class_node.body:
#                 if (
#                     isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
#                     and item.name == method_name
#                 ):
#                     method_node = item
#                     class_name = class_node.name
#                     break
#         if not method_node:
#             raise ValueError(
#                 f"Method '{method_name}' not found in class '{class_name}' at id {id}."
#             )
#         # Resolve method dependencies
#         collector.resolve(name=None, function=method_name, class_name=class_name)
#         # Reconstruct a class with just the docstring (if present) and the test method
#         new_body = []
#         if (
#             class_node.body
#             and isinstance(class_node.body[0], ast.Expr)
#             and isinstance(class_node.body[0].value, ast.Constant)
#             and isinstance(class_node.body[0].value.value, str)
#         ):
#             new_body.append(class_node.body[0])
#         new_body.append(method_node)
#         new_class = ast.ClassDef(
#             name=class_node.name,
#             bases=class_node.bases,
#             keywords=getattr(class_node, "keywords", []),
#             decorator_list=class_node.decorator_list,
#             body=new_body,
#             lineno=class_node.lineno,
#             col_offset=class_node.col_offset,
#         )
#         # Remove any previous full class in nodes_to_keep
#         collector.nodes_to_keep = [
#             n
#             for n in collector.nodes_to_keep
#             if not (isinstance(n, ast.ClassDef) and n.name == class_name)
#         ]
#         collector.nodes_to_keep.append(new_class)
#     else:
#         # It's a function, just resolve it
#         collector.resolve(target)
#     return collector.get_minimal_code()


# class DependencyCollector(ast.NodeVisitor):
#     def __init__(self):
#         # Store all top-level objects
#         self.class_defs: Dict[str, List[ast.ClassDef]] = {}
#         self.func_defs: Dict[str, ast.FunctionDef] = {}
#         self.async_func_defs: Dict[str, ast.AsyncFunctionDef] = {}
#         self.assigns: Dict[str, ast.Assign] = {}
#         self.imports: List[ast.stmt] = []
#         self.used_names: Set[str] = set()
#         self.all_names: Set[str] = set()
#         self.nodes_to_keep: List[ast.stmt] = []
#         self.visited: Set[str] = set()
#         self.current_class: List[str] = []  # Stack for nested class support

#     def visit_Import(self, node):
#         self.imports.append(node)
#         for alias in node.names:
#             self.all_names.add(alias.asname or alias.name.split(".")[0])

#     def visit_ImportFrom(self, node):
#         self.imports.append(node)
#         for alias in node.names:
#             self.all_names.add(alias.asname or alias.name)

#     def visit_ClassDef(self, node):
#         self.class_defs.setdefault(node.name, []).append(node)
#         self.all_names.add(node.name)
#         self.current_class.append(node.name)
#         for item in node.body:
#             if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                 # Register class method with key
#                 key = f"{node.name}.{item.name}"
#                 if isinstance(item, ast.FunctionDef):
#                     self.func_defs[key] = item
#                 else:
#                     self.async_func_defs[key] = item
#                 self.all_names.add(key)
#         self.current_class.pop()

#     def visit_FunctionDef(self, node):
#         # Only for top-level functions (not methods)
#         if not self.current_class:
#             key = f"global.{node.name}"
#             self.func_defs[key] = node
#             self.all_names.add(key)

#     def visit_AsyncFunctionDef(self, node):
#         if not self.current_class:
#             key = f"global.{node.name}"
#             self.async_func_defs[key] = node
#             self.all_names.add(key)

#     def visit_Assign(self, node):
#         for t in node.targets:
#             if isinstance(t, ast.Name):
#                 self.assigns[t.id] = node
#                 self.all_names.add(t.id)

#     def collect(self, tree: ast.Module):
#         self.visit(tree)

#     def find_names_in_node(self, node: ast.AST) -> Set[str]:
#         """Recursively collect all variable names used in a node."""
#         names = set()

#         class NameVisitor(ast.NodeVisitor):
#             def visit_Name(self, n):
#                 names.add(n.id)

#         NameVisitor().visit(node)
#         return names

#     def resolve(self, key: str, function: str = None, class_name: str = None):
#         # `key` is either a fully qualified function/method name, or assignment variable
#         if key is None:
#             # Should not happen, safety
#             return
#         if key in self.visited:
#             return
#         self.visited.add(key)

#         # Function or method
#         if key in self.func_defs:
#             node = self.func_defs[key]
#             self.nodes_to_keep.append(node)
#             self._resolve_node(node)
#         elif key in self.async_func_defs:
#             node = self.async_func_defs[key]
#             self.nodes_to_keep.append(node)
#             self._resolve_node(node)
#         elif key in self.assigns:
#             node = self.assigns[key]
#             self.nodes_to_keep.append(node)
#             self._resolve_node(node)
#         elif "." in key:
#             # Might be a method that exists only in a class; try to keep the class minimally
#             class_name, method_name = key.split(".", 1)
#             self.resolve_class_method(class_name, method_name)
#         else:
#             # It could be a class definition
#             if key in self.class_defs:
#                 for class_node in self.class_defs[key]:
#                     self.nodes_to_keep.append(class_node)

#     def resolve_class_method(self, class_name: str, method_name: str):
#         # For class methods, reconstruct a minimal class node with just that method (and optional docstring)
#         class_nodes = self.class_defs.get(class_name, [])
#         for class_node in class_nodes:
#             # Find docstring if present
#             docstring = None
#             new_body = []
#             if (
#                 class_node.body
#                 and isinstance(class_node.body[0], ast.Expr)
#                 and isinstance(class_node.body[0].value, ast.Constant)
#                 and isinstance(class_node.body[0].value.value, str)
#             ):
#                 docstring = class_node.body[0]
#             # Find the method
#             method_node = None
#             for item in class_node.body:
#                 if (
#                     isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
#                     and item.name == method_name
#                 ):
#                     method_node = item
#                     break
#             if method_node is not None:
#                 if docstring:
#                     new_body.append(docstring)
#                 new_body.append(method_node)
#                 # Reconstruct minimal class node
#                 new_class = ast.ClassDef(
#                     name=class_node.name,
#                     bases=class_node.bases,
#                     keywords=getattr(class_node, "keywords", []),
#                     decorator_list=class_node.decorator_list,
#                     body=new_body,
#                     lineno=class_node.lineno,
#                     col_offset=class_node.col_offset,
#                 )
#                 # Remove any previous full class in nodes_to_keep
#                 self.nodes_to_keep = [
#                     n
#                     for n in self.nodes_to_keep
#                     if not (isinstance(n, ast.ClassDef) and n.name == class_name)
#                 ]
#                 self.nodes_to_keep.append(new_class)
#                 # Recursively resolve method's dependencies
#                 self._resolve_node(method_node)
#                 # Optionally, resolve base classes
#                 for base in class_node.bases:
#                     if isinstance(base, ast.Name):
#                         self.resolve(base.id)
#                 return
#         # Fallback: keep the full class definition if something failed
#         if class_nodes:
#             self.nodes_to_keep.append(class_nodes[-1])

#     def _resolve_node(self, node):
#         """Given any node, resolve all referenced names recursively."""
#         for n in ast.walk(node):
#             if isinstance(n, ast.Name):
#                 # Try both global and class-qualified keys
#                 candidates = [f"global.{n.id}"] + [
#                     k for k in self.all_names if k.endswith(f".{n.id}")
#                 ]
#                 found = False
#                 for candidate in candidates:
#                     if candidate in self.all_names:
#                         self.resolve(candidate)
#                         found = True
#                 # If it's an assignment/global/class, also try as-is
#                 if not found and n.id in self.all_names:
#                     self.resolve(n.id)

#     def filter_imports(self):
#         """Only keep needed imports."""
#         needed_imports = []
#         needed_names = set()
#         for node in self.nodes_to_keep:
#             for name in self.find_names_in_node(node):
#                 needed_names.add(name)
#         for imp in self.imports:
#             if isinstance(imp, ast.Import):
#                 for alias in imp.names:
#                     if alias.asname and alias.asname in needed_names:
#                         needed_imports.append(imp)
#                         break
#                     elif alias.name.split(".")[0] in needed_names:
#                         needed_imports.append(imp)
#                         break
#             elif isinstance(imp, ast.ImportFrom):
#                 for alias in imp.names:
#                     if (alias.asname and alias.asname in needed_names) or (
#                         alias.name in needed_names
#                     ):
#                         needed_imports.append(imp)
#                         break
#         return needed_imports

#     def get_minimal_code(self):
#         imports = self.filter_imports()
#         code_blocks = sorted(
#             imports + self.nodes_to_keep, key=lambda n: getattr(n, "lineno", 0)
#         )
#         return "\n\n".join(ast.unparse(n) for n in code_blocks)


# def extract_minimal_test(script: str, target: str, id: str):
#     """
#     target:
#       - For a function: 'test_func_name'
#       - For a class method: 'ClassName|class_method_split|method_name'
#     """
#     tree = ast.parse(script)
#     collector = DependencyCollector()
#     collector.collect(tree)

#     if "|class_method_split|" in target:
#         class_name, method_name = target.split("|class_method_split|")
#         method_key = f"{class_name}.{method_name}"
#         if class_name not in collector.class_defs:
#             raise ValueError(f"Class '{class_name}' not found in file.")
#         # Check if method exists
#         found = False
#         for class_node in collector.class_defs[class_name]:
#             for item in class_node.body:
#                 if (
#                     isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
#                     and item.name == method_name
#                 ):
#                     found = True
#         if not found:
#             raise ValueError(
#                 f"Method '{method_name}' not found in class '{class_name}' at id {id}."
#             )
#         # Resolve by full key
#         collector.resolve(method_key)
#     else:
#         # It's a function, resolve as global
#         key = f"global.{target}"
#         collector.resolve(key)
#     return collector.get_minimal_code()


class DependencyCollector(ast.NodeVisitor):
    def __init__(self):
        # Store all top-level objects
        self.class_defs: Dict[str, List[ast.ClassDef]] = {}
        self.func_defs: Dict[str, ast.FunctionDef] = {}
        self.async_func_defs: Dict[str, ast.AsyncFunctionDef] = {}
        self.assigns: Dict[str, ast.Assign] = {}
        self.imports: List[ast.stmt] = []
        self.used_names: Set[str] = set()
        self.all_names: Set[str] = set()
        self.nodes_to_keep: List[ast.stmt] = []
        self.visited: Set[str] = set()
        self.current_class: List[str] = []  # Stack for nested class support

    def visit_Import(self, node):
        self.imports.append(node)
        for alias in node.names:
            self.all_names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        self.imports.append(node)
        for alias in node.names:
            self.all_names.add(alias.asname or alias.name)

    def visit_ClassDef(self, node):
        self.class_defs.setdefault(node.name, []).append(node)
        self.all_names.add(node.name)
        self.current_class.append(node.name)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Register class method with key
                key = f"{node.name}.{item.name}"
                if isinstance(item, ast.FunctionDef):
                    self.func_defs[key] = item
                else:
                    self.async_func_defs[key] = item
                self.all_names.add(key)
        self.current_class.pop()

    def visit_FunctionDef(self, node):
        # Only for top-level functions (not methods)
        if not self.current_class:
            key = f"global.{node.name}"
            self.func_defs[key] = node
            self.all_names.add(key)

    def visit_AsyncFunctionDef(self, node):
        if not self.current_class:
            key = f"global.{node.name}"
            self.async_func_defs[key] = node
            self.all_names.add(key)

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.assigns[t.id] = node
                self.all_names.add(t.id)

    def collect(self, tree: ast.Module):
        self.visit(tree)

    def find_names_in_node(self, node: ast.AST) -> Set[str]:
        """Recursively collect all variable names used in a node."""
        names = set()

        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, n):
                names.add(n.id)

        NameVisitor().visit(node)
        return names

    def is_test_class(self, class_node: ast.ClassDef) -> bool:
        """Check if a class is a Django test case class."""
        # Check if it inherits from TestCase, SimpleTestCase, etc.
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                if base.id in (
                    "TestCase",
                    "SimpleTestCase",
                    "TransactionTestCase",
                    "LiveServerTestCase",
                ):
                    return True
            elif isinstance(base, ast.Attribute):
                # Handle cases like django.test.TestCase
                if base.attr in (
                    "TestCase",
                    "SimpleTestCase",
                    "TransactionTestCase",
                    "LiveServerTestCase",
                ):
                    return True
        return False

    def resolve(self, key: str, function: str = None, class_name: str = None):
        # `key` is either a fully qualified function/method name, or assignment variable
        if key is None:
            # Should not happen, safety
            return
        if key in self.visited:
            return
        self.visited.add(key)

        # Function or method
        if key in self.func_defs:
            node = self.func_defs[key]
            self.nodes_to_keep.append(node)
            self._resolve_node(node)
        elif key in self.async_func_defs:
            node = self.async_func_defs[key]
            self.nodes_to_keep.append(node)
            self._resolve_node(node)
        elif key in self.assigns:
            node = self.assigns[key]
            self.nodes_to_keep.append(node)
            self._resolve_node(node)
        elif "." in key:
            # Might be a method that exists only in a class; try to keep the class minimally
            class_name, method_name = key.split(".", 1)
            self.resolve_class_method(class_name, method_name)
        else:
            # It could be a class definition
            if key in self.class_defs:
                for class_node in self.class_defs[key]:
                    self.nodes_to_keep.append(class_node)

    def resolve_test_class(self, class_name: str, test_method_name: str = None):
        """
        For Django test classes, keep the entire class with all its methods,
        or optionally just a specific test method if specified.
        """
        class_nodes = self.class_defs.get(class_name, [])
        for class_node in class_nodes:
            if not self.is_test_class(class_node):
                # Not a test class, fall back to regular method resolution
                if test_method_name:
                    self._resolve_regular_class_method(
                        class_node, class_name, test_method_name
                    )
                else:
                    self.nodes_to_keep.append(class_node)
                    self._resolve_node(class_node)
                return

            # For test classes, we have two strategies:
            if test_method_name:
                # Strategy 1: Extract just the specific test method with setUp/tearDown
                self._resolve_test_class_with_method(class_node, test_method_name)
            else:
                # Strategy 2: Keep the entire test class
                self.nodes_to_keep.append(class_node)
                self._resolve_node(class_node)

            # Resolve base classes (TestCase, etc.)
            for base in class_node.bases:
                if isinstance(base, ast.Name):
                    self.resolve(base.id)

    def _resolve_test_class_with_method(
        self, class_node: ast.ClassDef, test_method_name: str
    ):
        """
        Create a minimal test class with just the specified test method,
        plus any setUp/tearDown methods and class docstring.
        """
        new_body = []

        # Always include class docstring if present
        if (
            class_node.body
            and isinstance(class_node.body[0], ast.Expr)
            and isinstance(class_node.body[0].value, ast.Constant)
            and isinstance(class_node.body[0].value.value, str)
        ):
            new_body.append(class_node.body[0])

        # Find and include setUp, tearDown, setUpClass, tearDownClass methods
        essential_methods = {
            "setUp",
            "tearDown",
            "setUpClass",
            "tearDownClass",
            "setUpTestData",
            "setUpModule",
            "tearDownModule",
        }

        target_method = None
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == test_method_name:
                    target_method = item
                elif item.name in essential_methods:
                    new_body.append(item)
                    self._resolve_node(item)

        if target_method is None:
            raise ValueError(
                f"Test method '{test_method_name}' not found in class '{class_node.name}'"
            )

        new_body.append(target_method)
        self._resolve_node(target_method)

        # Include any class variables/attributes that might be needed
        for item in class_node.body:
            if isinstance(item, ast.Assign):
                new_body.append(item)
                self._resolve_node(item)

        # Create minimal class
        new_class = ast.ClassDef(
            name=class_node.name,
            bases=class_node.bases,
            keywords=getattr(class_node, "keywords", []),
            decorator_list=class_node.decorator_list,
            body=new_body,
            lineno=class_node.lineno,
            col_offset=class_node.col_offset,
        )

        # Remove any previous class definition with same name
        self.nodes_to_keep = [
            n
            for n in self.nodes_to_keep
            if not (isinstance(n, ast.ClassDef) and n.name == class_node.name)
        ]
        self.nodes_to_keep.append(new_class)

    def _resolve_regular_class_method(
        self, class_node: ast.ClassDef, class_name: str, method_name: str
    ):
        """
        Handle regular (non-test) class method resolution.
        """
        # Find docstring if present
        docstring = None
        new_body = []
        if (
            class_node.body
            and isinstance(class_node.body[0], ast.Expr)
            and isinstance(class_node.body[0].value, ast.Constant)
            and isinstance(class_node.body[0].value.value, str)
        ):
            docstring = class_node.body[0]

        # Find the method
        method_node = None
        for item in class_node.body:
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == method_name
            ):
                method_node = item
                break

        if method_node is not None:
            if docstring:
                new_body.append(docstring)
            new_body.append(method_node)

            # Reconstruct minimal class node
            new_class = ast.ClassDef(
                name=class_node.name,
                bases=class_node.bases,
                keywords=getattr(class_node, "keywords", []),
                decorator_list=class_node.decorator_list,
                body=new_body,
                lineno=class_node.lineno,
                col_offset=class_node.col_offset,
            )

            # Remove any previous full class in nodes_to_keep
            self.nodes_to_keep = [
                n
                for n in self.nodes_to_keep
                if not (isinstance(n, ast.ClassDef) and n.name == class_name)
            ]
            self.nodes_to_keep.append(new_class)

            # Recursively resolve method's dependencies
            self._resolve_node(method_node)

            # Resolve base classes
            for base in class_node.bases:
                if isinstance(base, ast.Name):
                    self.resolve(base.id)

    def resolve_class_method(self, class_name: str, method_name: str):
        # Check if this is a test class first
        class_nodes = self.class_defs.get(class_name, [])
        for class_node in class_nodes:
            if self.is_test_class(class_node):
                self.resolve_test_class(class_name, method_name)
                return

        # For non-test classes, use the regular resolution logic
        for class_node in class_nodes:
            self._resolve_regular_class_method(class_node, class_name, method_name)
            return

        # Fallback: keep the full class definition if something failed
        if class_nodes:
            self.nodes_to_keep.append(class_nodes[-1])

    def _resolve_node(self, node):
        """Given any node, resolve all referenced names recursively."""
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                # Try both global and class-qualified keys
                candidates = [f"global.{n.id}"] + [
                    k for k in self.all_names if k.endswith(f".{n.id}")
                ]
                found = False
                for candidate in candidates:
                    if candidate in self.all_names:
                        self.resolve(candidate)
                        found = True
                # If it's an assignment/global/class, also try as-is
                if not found and n.id in self.all_names:
                    self.resolve(n.id)

    def filter_imports(self):
        """Only keep needed imports."""
        needed_imports = []
        needed_names = set()
        for node in self.nodes_to_keep:
            for name in self.find_names_in_node(node):
                needed_names.add(name)

        # Always include common Django test imports
        django_test_names = {
            "TestCase",
            "SimpleTestCase",
            "TransactionTestCase",
            "LiveServerTestCase",
            "Client",
        }
        needed_names.update(django_test_names)

        for imp in self.imports:
            if isinstance(imp, ast.Import):
                for alias in imp.names:
                    if alias.asname and alias.asname in needed_names:
                        needed_imports.append(imp)
                        break
                    elif alias.name.split(".")[0] in needed_names:
                        needed_imports.append(imp)
                        break
            elif isinstance(imp, ast.ImportFrom):
                for alias in imp.names:
                    if (alias.asname and alias.asname in needed_names) or (
                        alias.name in needed_names
                    ):
                        needed_imports.append(imp)
                        break
        return needed_imports

    def get_minimal_code(self):
        imports = self.filter_imports()
        code_blocks = sorted(
            imports + self.nodes_to_keep, key=lambda n: getattr(n, "lineno", 0)
        )
        return "\n\n".join(ast.unparse(n) for n in code_blocks)


def extract_minimal_test(script: str, target: str, id: str):
    """
    target:
      - For a function: 'test_func_name'
      - For a class method: 'ClassName|class_method_split|method_name'
      - For a test class: 'TestClassName' (extracts entire test class)
      - For a specific test method: 'TestClassName|class_method_split|test_method_name'
    """
    tree = ast.parse(script)
    collector = DependencyCollector()
    collector.collect(tree)

    if "|class_method_split|" in target:
        class_name, method_name = target.split("|class_method_split|")

        # Check if the class exists
        if class_name not in collector.class_defs:
            raise ValueError(f"Class '{class_name}' not found in file.")

        # Check if it's a test class
        class_nodes = collector.class_defs[class_name]
        is_test_class = any(collector.is_test_class(node) for node in class_nodes)

        if is_test_class:
            # For test classes, use the enhanced test class resolution
            collector.resolve_test_class(class_name, method_name)
        else:
            # For regular classes, use the original method resolution
            method_key = f"{class_name}.{method_name}"
            # Check if method exists
            found = False
            for class_node in class_nodes:
                for item in class_node.body:
                    if (
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == method_name
                    ):
                        found = True
            if not found:
                raise ValueError(
                    f"Method '{method_name}' not found in class '{class_name}' at id {id}."
                )
            # Resolve by full key
            collector.resolve(method_key)
    else:
        # Check if it's a test class (entire class)
        if target in collector.class_defs:
            class_nodes = collector.class_defs[target]
            is_test_class = any(collector.is_test_class(node) for node in class_nodes)
            if is_test_class:
                collector.resolve_test_class(target)
            else:
                collector.resolve(target)
        else:
            # It's a function, resolve as global
            key = f"global.{target}"
            collector.resolve(key)

    return collector.get_minimal_code()


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


# def trim_test_cases(source_code, target):

#     collector = DependencyCollector()
#     tree = collector.collect(source_code)

#     if "." in target:
#         class_name, method_name = target.split(".")
#         collector.resolve_class_method(class_name.strip(), method_name.strip())
#     else:
#         collector.resolve_dependencies(target)

#     trimmed_code = collector.reconstruct_code()
#     return trimmed_code


# def trim_test_cases(source_code, target):

#     class_name = None
#     method_name = None
#     function_name = None
#     collector = DependencyCollector()
#     tree = collector.collect(source_code)

#     if "|class_method_split|" in target:
#         class_name, method_name = target.split("|class_method_split|")
#     else:
#         function_name = target

#     if function_name is not None:
#         collector.resolve_dependencies(function_name)

#     if (class_name is not None) and (method_name is not None):
#         collector.resolve_class_method(class_name, method_name)

#     trimmed_code = collector.reconstruct_code()
#     return trimmed_code


def extract_function_names_from_code(code: str):
    # try:
    #     tree = ast.parse(code)
    # except SyntaxError as e:
    #     print(f"Syntax error: {e}")
    #     return []
    tree = ast.parse(code)
    function_names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
    return function_names


def find_tests_in_script(
    script: str,
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    tree = ast.parse(script)

    test_functions = []
    test_classes = {}

    # Known base classes (for legacy use / optional enforcement)
    known_test_bases: Set[str] = {
        "unittest.TestCase",
        "django.test.TestCase",
        "django.test.SimpleTestCase",
        "TestCase",
        "SimpleTestCase",
    }

    def is_test_function(node):
        return isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test")

    def walk_with_parents(node, parents=None):
        if parents is None:
            parents = []
        yield node, parents
        for child in ast.iter_child_nodes(node):
            yield from walk_with_parents(child, parents + [node])

    test_class_nodes = set()

    # Step 1: Collect all test classes and their test methods
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            method_names = [
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("test")
            ]
            if method_names:
                test_classes[node.name] = method_names
                test_class_nodes.add(node)

    # Step 2: Collect all top-level test functions
    for node in tree.body:
        if is_test_function(node):
            test_functions.append(node.name)

    return {"test_functions": test_functions, "test_classes": test_classes}
