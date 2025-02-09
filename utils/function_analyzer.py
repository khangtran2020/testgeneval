import ast
import json
from utils.utils import indent_text
from swebench_docker.constants import KEY_ID
from swebench_docker.utils import get_test_tasks


class TestFunctionExtractor(ast.NodeVisitor):
    def __init__(self, source_code):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.test_functions = []

    def visit_FunctionDef(self, node):
        if node.name.startswith("test_"):
            # Extract the function content using line numbers
            start_line = node.lineno - 1  # Adjusting for zero-based index
            end_line = self._find_end_line(node)

            function_code = "\n".join(self.lines[start_line:end_line])
            self.test_functions.append(function_code)

        self.generic_visit(node)

    def _find_end_line(self, node):
        # Traverse the function body to find the last line
        last_node = node.body[-1]
        while isinstance(
            last_node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.With,
                ast.Try,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            last_node = last_node.body[-1]
        return (
            last_node.end_lineno
            if hasattr(last_node, "end_lineno")
            else last_node.lineno + 1
        )


def extract_test_functions_from_code(source_code: str):
    tree = ast.parse(source_code)
    extractor = TestFunctionExtractor(source_code)
    extractor.visit(tree)
    return extractor.test_functions


def combine_translate_one_task(task_instance: dict, num_try: int) -> None:

    preamble = task_instance["preds_context"]["preamble"]
    for i in range(num_try):
        for key in task_instance[f"translate_{i}"].keys():
            test_content = combine_translate_and_preamble(
                preamble=preamble,
                translated=task_instance[f"translate_{i}"][key],
                repo=task_instance["repo"],
            )
            task_instance[f"translate_{i}"][key] = test_content


def combine_translate_all(data_path: str, num_try: int) -> int:

    try:
        out_path = data_path.replace(".jsonl", "_combined.jsonl")
        tasks = get_test_tasks(data_path)
        task_dict = {task[KEY_ID]: task for task in tasks}
        for key in task_dict.keys():
            combine_translate_one_task(task_instance=task_dict[key], num_try=num_try)
        with open(out_path, "w") as f:
            for item in task_dict.values():
                f.write(json.dumps(item) + "\n")
        print(f"Combine complete")
        return 1
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


def combine_translate_and_preamble(preamble: str, translated: str, repo: str) -> str:

    if translated == "":
        return ""

    # extract test function
    test_function = extract_test_functions_from_code(source_code=translated)[0]

    # extract imports
    import_list = []
    for line in translated.split("\n"):
        if line.strip().startswith("import "):
            import_list.append(line.strip())

    for import_item in import_list:
        if import_item in preamble:
            import_list.remove(import_item)
    imports = "\n".join(import_list)
    preamble = f"{imports}\n\n{preamble}"

    if repo == "django/django":
        if "(self):" not in test_function:
            test_function = test_function.replace("():", "(self):", 1)
        test_content = preamble + "\n\n" + indent_text(test_function, 4)
    else:
        test_content = preamble + "\n\n" + test_function
    return test_content
