import os
import ast
import json
import asyncio
import aiofiles
import datetime
import tempfile
import subprocess
from tqdm.asyncio import tqdm
from utils.utils import (
    # extract_preamble_classes_and_functions,
    # postprocess_functions,
    # postprocess_tests,
    extract_minimal_test,
    find_tests_in_script,
)
from datasets import load_dataset, load_from_disk, concatenate_datasets

# typing
from rich.console import Console
from typing import Dict


class RemoveImportOfName(ast.NodeTransformer):
    def __init__(self, target_name: str):
        self.target = target_name

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Keep only aliases that aren't the target object
        kept = [a for a in node.names if a.name != self.target]
        if not kept:
            return None  # drop entire statement
        if len(kept) != len(node.names):
            node = ast.ImportFrom(module=node.module, names=kept, level=node.level)
        return node

    def visit_Import(self, node: ast.Import):
        # Remove plain "import <object_name>" if present
        kept = [a for a in node.names if a.name != self.target]
        if not kept:
            return None
        if len(kept) != len(node.names):
            node = ast.Import(names=kept)
        return node


def remove_import(src: str, object_name: str):
    tree = ast.parse(src)

    new_tree = RemoveImportOfName(object_name).visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_src = ast.unparse(new_tree)  # Python 3.9+
    except AttributeError:
        raise SystemExit("Python 3.9+ required (ast.unparse not available).")

    if not new_src.endswith("\n"):
        new_src += "\n"
    return new_src


def get_importables(code):
    try:
        tree = ast.parse(code)
        importables = []

        for node in tree.body:  # Only iterates through top-level statements
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    importables.append(node.name)

            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    importables.append(node.name)

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if not target.id.startswith("_"):
                            importables.append(target.id)

        return importables
    except:
        return []


def extract_imports(code):
    try:
        tree = ast.parse(code)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                names = ", ".join(alias.name for alias in node.names)
                imports.append((module, names))
        return imports
    except:
        return []


class Data(object):

    def __init__(
        self,
        data_name: str,
        data_path: str = None,
        save_path: str = None,
        num_processes: int = 1,
        console: Console = None,
    ) -> None:

        if (
            data_name
            not in ["kjain14/testgeneval", "kjain14/testgenevallite", "codamosa"]
            and data_path is None
        ):
            raise ValueError(
                "Invalid data name without data path, please provide data path"
            )
        self.data_name = data_name
        self.data_path = data_path
        self.save_path = save_path
        self.console = console
        self.num_processes = num_processes

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def load_raw_data(self) -> None:
        if "testgeneval" in self.data_name:
            if self.data_path is None:
                dataset = load_dataset(self.data_name, split="test")
            else:
                dataset = load_from_disk(self.data_path)
            self.dataset = dataset
            self.console.log(f"Data {self.data_name} loaded successfully")
        elif self.data_name == "codamosa":
            with open(self.data_path, "r") as f:
                dataset = [json.loads(line) for line in f.readlines()]
            self.dataset = dataset
            self.console.log(f"Data {self.data_name} loaded successfully")
        else:
            raise ValueError("Invalid data name")

    def process_data(self) -> None:
        if os.path.exists(
            os.path.join(self.save_path, f"{self.data_name.split('/')[-1]}.jsonl")
        ):
            self.console.log(f"Data {self.data_name} already processed")
            return
        if self.data_name in ["kjain14/testgeneval", "kjain14/testgenevallite"]:
            asyncio.run(self.process_raw_data_testgeneval())
        elif self.data_name == "codamosa":
            asyncio.run(self.process_raw_data_codamosa())

    async def process_raw_data_testgeneval(self) -> None:
        data_list = []
        num_test_cases = 0
        num_data_point = 0

        semaphore = asyncio.Semaphore(self.num_processes)

        tasks = [
            self.process_one_raw(self.dataset[i], semaphore)
            for i in range(len(self.dataset))
        ]
        await asyncio.gather(*tasks)

    async def process_one_raw(self, data: Dict, semaphore) -> Dict:

        async with semaphore:
            repo = data["repo"]
            commit_id = data["base_commit"]
            version = data["version"]
            instance_id = data["instance_id"]
            patch = data["patch"]
            test_patch = data["test_patch"]
            preds_context = data["preds_context"]
            code_src = data["code_src"]
            test_src = data["test_src"]
            code_file = data["code_file"]
            test_file = data["test_file"]
            local_imports = data["local_imports"]
            idx = data["id"]
            baseline_covs = data["baseline_covs"]

            self.console.log(f"[blue]Working on instance id: {instance_id}[/blue]")

            # preamble, classes, test_functions = extract_preamble_classes_and_functions(
            #     code=test_src
            # )
            test_dict = find_tests_in_script(test_src)

            # self.console.log(f"==================== {idx} ====================")
            test_cases = {}
            test_id = 0

            test_cases_exist = []

            for tar_func in test_dict["test_functions"]:
                trimmed_code = extract_minimal_test(
                    script=test_src, target=tar_func, id=instance_id
                )

                if trimmed_code is not None:
                    trimmed_code = remove_docstrings(trimmed_code)
                    if repo == "django/django":
                        trimmed_code = handle_django_testcase(
                            trimmed_code=trimmed_code, test_name=tar_func
                        )
                    try:
                        # Create a temporary file (not auto-deleted)
                        temp = tempfile.NamedTemporaryFile(delete=False)
                        temp_name = temp.name

                        os.chmod(temp_name, 0o666)
                        temp.write(trimmed_code.encode("utf-8"))
                        temp.flush()
                        try:
                            result = subprocess.run(
                                [
                                    "ruff",
                                    "check",
                                    temp_name,
                                    "--select",
                                    "F401",
                                    "--fix",
                                ],
                                check=True,
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"Error occurred while running ruff: {e}")
                            continue

                        temp.seek(0)
                        with open(temp_name, "r") as f:
                            cleaned_code = f.read()
                        trimmed_code = cleaned_code
                    finally:
                        # Always remove the temp file after processing
                        try:
                            temp.close()
                            os.remove(temp_name)
                            print(f"Temporary file {temp_name} removed.")
                        except Exception as e:
                            print(f"Error removing temp file: {e}")

                    # imports = extract_imports(code=trimmed_code)
                    # module_path = code_file.replace("/", ".").split(".py")[0]
                    # module_code = code_src
                    # importables = get_importables(code=module_code)

                    # is_directly_imported = False
                    # for imp in imports:
                    #     if isinstance(imp, tuple):
                    #         module = imp[0]
                    #         if module == module_path:
                    #             is_directly_imported = True
                    #             break
                    #         else:
                    #             if imp[1] in importables:
                    #                 removed_code = remove_import(
                    #                     src=trimmed_code,
                    #                     object_name=imp[1],
                    #                 )
                    #                 trimmed_code = (
                    #                     f"from {module_path} import {imp[1]}\n"
                    #                     + removed_code
                    #                 )
                    #     else:
                    #         if module_path in imp:
                    #             is_directly_imported = True
                    #             break

                    # if not is_directly_imported:
                    #     continue
                    if trimmed_code in test_cases_exist:
                        continue

                    test_cases[f"test_case_{test_id}"] = {
                        "target": tar_func,
                        "code": trimmed_code,
                    }
                    test_id += 1

            for class_name in test_dict["test_classes"]:
                for method in test_dict["test_classes"][class_name]:
                    trimmed_code = extract_minimal_test(
                        script=test_src,
                        target=f"{class_name}|class_method_split|{method}",
                        id=instance_id,
                    )
                    if trimmed_code is not None:
                        trimmed_code = remove_docstrings(trimmed_code)
                        if repo == "django/django":
                            trimmed_code = handle_django_testcase(
                                trimmed_code=trimmed_code, test_name=method
                            )
                        try:
                            # Create a temporary file (not auto-deleted)
                            temp = tempfile.NamedTemporaryFile(delete=False)
                            temp_name = temp.name

                            os.chmod(temp_name, 0o666)
                            temp.write(trimmed_code.encode("utf-8"))
                            temp.flush()
                            try:
                                result = subprocess.run(
                                    [
                                        "ruff",
                                        "check",
                                        temp_name,
                                        "--select",
                                        "F401",
                                        "--fix",
                                    ],
                                    check=True,
                                )
                            except subprocess.CalledProcessError as e:
                                print(f"Error occurred while running ruff: {e}")
                                continue

                            temp.seek(0)
                            with open(temp_name, "r") as f:
                                cleaned_code = f.read()
                            trimmed_code = cleaned_code
                        finally:
                            # Always remove the temp file after processing
                            try:
                                temp.close()
                                os.remove(temp_name)
                                print(f"Temporary file {temp_name} removed.")
                            except Exception as e:
                                print(f"Error removing temp file: {e}")

                        # imports = extract_imports(code=trimmed_code)
                        # module_path = code_file.replace("/", ".").split(".py")[0]
                        # module_code = code_src
                        # importables = get_importables(code=module_code)

                        # is_directly_imported = False
                        # for imp in imports:
                        #     if isinstance(imp, tuple):
                        #         module = imp[0]
                        #         if module == module_path:
                        #             is_directly_imported = True
                        #             break
                        #         # else:
                        #         #     if imp[1] in importables:
                        #         #         is_directly_imported = True
                        #         #         break
                        #     else:
                        #         if module_path in imp:
                        #             is_directly_imported = True
                        #             break

                        # if not is_directly_imported:
                        #     continue
                        if trimmed_code in test_cases_exist:
                            continue

                        test_cases[f"test_case_{test_id}"] = {
                            "target": f"{class_name}.{method}",
                            "code": trimmed_code,
                        }
                        test_id += 1

            branches = {}
            arcs = {}
            for key in test_cases.keys():
                branches[key] = []
                arcs[key] = []

            processed_data = {
                "repo": repo,
                "base_commit": commit_id,
                "version": version,
                "instance_id": instance_id,
                "patch": patch,
                "test_patch": test_patch,
                "preds_context": preds_context,
                "code_src": code_src,
                "test_src": test_src,
                "code_file": code_file,
                "test_file": test_file,
                "local_imports": local_imports,
                "id": idx,
                "baseline_covs": baseline_covs,
                "test_cases": test_cases,
                "branches": branches,
                "arcs": arcs,
            }

            if len(processed_data["test_cases"].keys()) == 0:
                self.console.log(f"No test cases found for {idx}")

            async with aiofiles.open(
                os.path.join(self.save_path, f"{self.data_name.split('/')[-1]}.jsonl"),
                "a",
            ) as f:
                await f.write(json.dumps(processed_data) + "\n")

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.console.log(f"Done {idx} at {current_time}")
            return len(processed_data["test_cases"].keys())

    async def process_raw_data_codamosa(self) -> None:
        semaphore = asyncio.Semaphore(self.num_processes)

        tasks = [
            self.process_one_raw_codamosa(self.dataset[i], semaphore)
            for i in range(len(self.dataset))
        ]
        await asyncio.gather(*tasks)

    async def process_one_raw_codamosa(self, data: Dict, semaphore) -> Dict:

        async with semaphore:

            repo = data["repo_name"]
            code_src = data["source_code"]
            code_file = data["module_path"]
            test_file = "./src_test.py"
            classes = [cls["name"] for cls in data["classes"]]
            functions = data["functions"]
            local_imports = (
                "from "
                + data["module_name"]
                + " import "
                + ", ".join([*classes, *functions])
                + "\n"
            )
            idx = data["module_name"]
            self.console.log(f"[blue]Working on module: {idx}[/blue]")
            processed_data = {
                "repo": repo,
                "code_src": code_src,
                "code_file": code_file,
                "test_file": test_file,
                "local_imports": local_imports,
                "id": idx,
            }

            async with aiofiles.open(
                os.path.join(self.save_path, f"{self.data_name.split('/')[-1]}.jsonl"),
                "a",
            ) as f:
                await f.write(json.dumps(processed_data) + "\n")

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.console.log(f"Done {idx} at {current_time}")
            return len(processed_data["test_cases"].keys())


def handle_django_testcase(trimmed_code: str, test_name: str):
    # repo = task_instance["repo"]
    # django_repo = repo == "django/django"

    # extract everything before the test case
    preamble, func_code, decorator_found = extract_preamble(
        test_src=trimmed_code, test_name=test_name
    )

    def needs_django_harness(preamble):
        no_django_test = "TestCase" not in preamble
        no_unittest = "unittest" not in preamble
        no_simple_test_case = "SimpleTestCase" not in preamble
        return no_django_test and no_unittest and no_simple_test_case

    added_class = False
    if needs_django_harness(preamble):
        preamble = "from django.test import SimpleTestCase\n" + preamble
        class_wrapper_start = "\n\nclass TestsHarness(SimpleTestCase):\n"
        preamble += class_wrapper_start
        added_class = True

    class_content = ""
    if added_class:
        if not decorator_found:
            func_code = "\n\n" + indent_text(code=func_code, num_spaces=4)
        test_content = preamble + indent_text(code=func_code, num_spaces=4)
    else:
        if not decorator_found:
            func_code = "\n\n" + func_code
        test_content = preamble + func_code

    return test_content


def extract_preamble(test_src: str, test_name: str) -> str:
    lines = test_src.split("\n")
    preamble_lines = []
    decorator_found = False
    for line in lines:
        if test_name in line:
            # check for decorators
            for rline_index in range(len(preamble_lines) - 1, -1, -1):
                rline = preamble_lines[rline_index]
                if rline.strip().startswith("@"):
                    preamble_lines.pop(rline_index)
                    decorator_found = True
                if rline.strip() == "":
                    break
        preamble_lines.append(line)

    code_lines = [line for line in lines if line not in preamble_lines]
    preamble = "\n".join(preamble_lines)
    code = "\n".join(code_lines)
    return preamble, code, decorator_found


def indent_text(code: str, num_spaces: int) -> str:

    code_lines = code.split("\n")
    first_indent = len(code_lines[0]) - len(code_lines[0].lstrip())

    results = []
    if first_indent == 0:
        indent = " " * num_spaces
        for line in code_lines:
            if line.strip() == "":
                results.append(line)
            else:
                results.append(indent + line)
    else:
        for line in code_lines:
            if line.strip() == "":
                results.append(line)
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent >= first_indent:
                supp_line = line[first_indent:]
                supp_line = indent + supp_line
                results.append(supp_line)
            else:
                results.append(line)

    return "\n".join(results)


class DocstringRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node


def remove_docstrings(source_code):
    tree = ast.parse(source_code)
    tree = DocstringRemover().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
