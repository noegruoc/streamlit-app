"""Generate a JSON file containing the structure of all Python code in the project."""

import ast
import json
from pathlib import Path
from typing import Any


def extract_structure(file_path: Path) -> dict[str, Any]:
    """Extract classes, functions, and variables from a Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return {"error": "Could not parse file"}

    structure: dict[str, list[Any]] = {
        "classes": [],
        "functions": [],
        "variables": [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info: dict[str, Any] = {
                "name": node.name,
                "line": node.lineno,
                "methods": [],
                "attributes": [],
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = {
                        "name": item.name,
                        "line": item.lineno,
                        "args": [arg.arg for arg in item.args.args],
                    }
                    class_info["methods"].append(method_info)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            class_info["attributes"].append(target.id)
            structure["classes"].append(class_info)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = {
                "name": node.name,
                "line": node.lineno,
                "args": [arg.arg for arg in node.args.args],
            }
            structure["functions"].append(func_info)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    structure["variables"].append(
                        {
                            "name": target.id,
                            "line": node.lineno,
                        }
                    )

    return structure


def generate_project_structure() -> dict[str, Any]:
    """Generate structure for all Python files in the project."""
    project_root = Path(__file__).parent.parent
    python_files = list(project_root.rglob("*.py"))

    # Exclude virtual environments and cache directories
    excluded_dirs = {"venv", "env", ".venv", "__pycache__", ".git", "node_modules"}
    python_files = [
        f
        for f in python_files
        if not any(excluded in f.parts for excluded in excluded_dirs)
    ]

    project_structure = {}
    for file_path in sorted(python_files):
        relative_path = file_path.relative_to(project_root)
        project_structure[str(relative_path)] = extract_structure(file_path)

    return project_structure


def main() -> None:
    """Generate and save the project structure to a JSON file."""
    project_root = Path(__file__).parent.parent
    structure = generate_project_structure()

    output_path = project_root / "code_structure.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=2)

    print(f"Code structure saved to {output_path}")


if __name__ == "__main__":
    main()
