"""Make the from-scratch restriction part of the test suite."""

import ast
import sys
from pathlib import Path


def test_core_imports_only_numpy_and_the_standard_library():
    base = Path(__file__).resolve().parents[2] / "src" / "nyquist_research"
    for file in base.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots = [name.name.split(".")[0] for name in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            assert set(roots) <= set(sys.stdlib_module_names) | {"numpy"}, (file, roots)
