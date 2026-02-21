"""Minimal notebook executor for environments without Jupyter tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import CodeType
from typing import Any


def _load_notebook(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Notebook payload is not a mapping: {path}")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Notebook is missing cells list: {path}")
    return payload


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    raise ValueError("Unsupported cell source format")


def _compile_cell(source: str, *, notebook_path: Path, cell_index: int) -> CodeType:
    filename = f"{notebook_path}#cell-{cell_index}"
    return compile(source, filename, "exec")


def execute_notebook(path: Path) -> None:
    """Execute notebook code cells in order."""

    notebook = _load_notebook(path)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    code_cells = 0

    for index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = _cell_source(cell)
        if not source.strip():
            continue
        code = _compile_cell(source, notebook_path=path, cell_index=index)
        exec(code, namespace, namespace)

    print(f"notebook-exec: ok ({path}, code_cells={code_cells})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute notebook code cells without Jupyter")
    parser.add_argument("notebook_path", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    execute_notebook(args.notebook_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
