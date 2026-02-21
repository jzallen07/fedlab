from __future__ import annotations

import json
from pathlib import Path

from scripts.execute_notebook import execute_notebook


def test_execute_notebook_runs_code_cells(tmp_path: Path) -> None:
    marker_path = tmp_path / "marker.txt"
    notebook_path = tmp_path / "sample.ipynb"

    notebook_payload = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# sample"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    f"Path({str(marker_path)!r}).write_text('ok', encoding='utf-8')\n",
                ],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook_payload), encoding="utf-8")

    execute_notebook(notebook_path)

    assert marker_path.read_text(encoding="utf-8") == "ok"
