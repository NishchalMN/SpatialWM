"""Single append-only results sink.

Every run appends one row to ``results/summary.csv``; this is the project's
single source of truth for experiment outcomes.
"""

from __future__ import annotations

import csv
import os


def append_result(row: dict, path: str = "results/summary.csv") -> None:
    """Append ``row`` to the CSV at ``path``, creating parent + header as needed.

    The header is the union of any existing header and the new row's keys, so
    runs that report different metrics still land in the same file. Rows written
    before a column existed simply leave that cell blank.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    existing_rows: list[dict] = []
    existing_header: list[str] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_header = list(reader.fieldnames or [])
            existing_rows = list(reader)

    header = list(existing_header)
    for k in row:
        if k not in header:
            header.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        writer.writerow(row)
