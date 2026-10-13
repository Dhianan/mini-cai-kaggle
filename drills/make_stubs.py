"""Regenerate drills/stubs/ (and fresh drills/exercises/) from drills/solutions/.

Keeps imports, docstrings and signatures; replaces every function body with
`raise NotImplementedError`. Run after editing a solution:
    python drills/make_stubs.py [--overwrite-exercises]
"""
import ast
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent


class Stub(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        doc = ast.get_docstring(node)
        body = [ast.Expr(ast.Constant(doc))] if doc else []
        node.body = body + [ast.Raise(ast.Call(ast.Name("NotImplementedError", ast.Load()), [], []), None)]
        return node


def stub(src: str) -> str:
    tree = Stub().visit(ast.parse(src))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def main() -> None:
    (HERE / "stubs").mkdir(exist_ok=True)
    (HERE / "exercises").mkdir(exist_ok=True)
    for f in sorted((HERE / "solutions").glob("d*.py")):
        (HERE / "stubs" / f.name).write_text(stub(f.read_text()))
        ex = HERE / "exercises" / f.name
        if not ex.exists() or "--overwrite-exercises" in sys.argv:
            shutil.copy(HERE / "stubs" / f.name, ex)
        print(f"stub {f.name}")


if __name__ == "__main__":
    main()
