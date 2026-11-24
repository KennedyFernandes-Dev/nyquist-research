"""Shared experiment paths, input models and output encoders."""

import csv
import json
from pathlib import Path

import numpy as np

from nyquist_research.transfer_function import TransferFunction

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CONFIG = ROOT / "experiments" / "configurations"


def systems() -> dict[str, TransferFunction]:
    items = json.loads((CONFIG / "systems.json").read_text())
    return {
        label: TransferFunction(
            item["numerator"], item["denominator"], item["gain"], item["delay"]
        )
        for label, item in items.items()
    }


def json_result(path: str, result: object) -> None:
    target = RESULTS / path
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_result(path: str, rows: list[dict]) -> None:
    target = RESULTS / path
    target.parent.mkdir(exist_ok=True, parents=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def complex_pairs(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]
