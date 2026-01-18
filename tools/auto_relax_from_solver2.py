import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from tools.auto_relax_qty_minus import relax_qty_minus

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "log" / "aps-loom.log"


def run_solver_without_qty_minus(input_path: Path) -> dict[str, Any]:
    """Run `python run.py` with APPLY_QTY_MINUS=false as a separate process.

    Returns {"status": str, "zeros_total": int} based on aps-loom.log.
    """
    # Clean old log
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    env = os.environ.copy()
    env["CALC_TEST_DATA"] = "true"
    env["TEST_INPUT_FILE"] = str(input_path)
    env["APPLY_QTY_MINUS"] = "false"

    proc = subprocess.run(
        ["python", "run.py"],
        cwd=str(ROOT_DIR),
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"run.py exited with code {proc.returncode}")

    if not LOG_PATH.exists():
        raise RuntimeError(f"{LOG_PATH} not found after run.py")

    status = "UNKNOWN"
    zeros_total = 0

    status_re = re.compile(r"Статус решения:\s+(\w+)")
    # Ищем строку с продуктом 0 и qty=...
    prod0_re = re.compile(r"Продукт\s+0\(")
    qty_re = re.compile(r"qty=(\d+)")

    # Лог может писаться в системной кодировке (Windows-1251 и т.п.),
    # поэтому читаем его с utf-8 и игнорируем ошибки декодирования.
    with LOG_PATH.open(encoding="utf8", errors="ignore") as f:
        for line in f:
            m = status_re.search(line)
            if m:
                status = m.group(1)
            if prod0_re.search(line):
                mq = qty_re.search(line)
                if mq:
                    zeros_total = int(mq.group(1))

    return {"status": status, "zeros_total": zeros_total}


def compute_deficit_from_input(data: dict[str, Any], zeros_total: int) -> tuple[int, int, int, int]:
    """Compute deficit based on input JSON and zeros_total from solver run."""
    machines_raw = data["machines"]
    products_raw = data["products"]
    cleans_raw = data.get("cleans", [])
    count_days = int(data["count_days"])

    M = len(machines_raw)
    D = count_days
    C = len(cleans_raw)

    machine_days_total = M * D
    machine_days_work = machine_days_total - C

    required = 0
    for p in products_raw:
        idx = int(p["idx"])
        if idx == 0:
            continue
        qty = int(p.get("qty", 0))
        qty_minus = int(p.get("qty_minus", 0))
        qty_minus_min = int(p.get("qty_minus_min", 0))

        if qty > 0:
            if qty_minus == 0:
                required += qty
            elif qty_minus_min > 0:
                required += qty_minus_min

    prod_capacity = machine_days_work - zeros_total
    deficit = required - prod_capacity

    return deficit, machine_days_total, machine_days_work, zeros_total


def auto_relax_with_deficit(input_path: Path, top_k: int = 5) -> tuple[dict[str, Any], int, int]:
    """Run solver without qty_minus, estimate deficit, and relax top-K products.

    Returns (new_data, deficit, step_minus).
    """
    with input_path.open(encoding="utf8") as f:
        data = json.load(f)

    solver_info = run_solver_without_qty_minus(input_path)
    status = solver_info["status"]
    zeros_total = solver_info["zeros_total"]

    if status not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(
            f"Solver status {status} when APPLY_QTY_MINUS=false; cannot derive deficit"
        )

    deficit, machine_days_total, machine_days_work, zeros_total = compute_deficit_from_input(
        data, zeros_total
    )

    if deficit <= 0:
        # No relaxation required from capacity point of view.
        return data, deficit, 0

    # Collect top-K products by qty
    products_raw = data["products"]
    candidates = [
        p for p in products_raw
        if int(p.get("idx", 0)) > 0 and int(p.get("qty", 0)) > 0
    ]
    candidates.sort(key=lambda p: int(p.get("qty", 0)), reverse=True)
    candidates = candidates[:top_k]

    sum_lday = sum(int(p.get("lday", 0)) for p in candidates)
    if sum_lday <= 0:
        step_minus = 1
    else:
        step_minus = math.ceil(deficit / sum_lday)
        if step_minus < 1:
            step_minus = 1

    new_data = relax_qty_minus(data, step_minus=step_minus, top_k=top_k)
    return new_data, deficit, step_minus


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Auto-relax qty_minus based on deficit (via external solver run without qty_minus)."
        )
    )
    parser.add_argument(
        "input",
        help="Path to input JSON file (e.g. example/middle_test_in.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output JSON file (default: overwrite input)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Relax only top-K products by qty (default: 5)",
    )

    args = parser.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    new_data, deficit, step_minus = auto_relax_with_deficit(in_path, top_k=args.top_k)

    with out_path.open("w", encoding="utf8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(
        f"Auto-relax completed. deficit={deficit}, step_minus={step_minus}, "
        f"top_k={args.top_k}. Saved to {out_path}"
    )


if __name__ == "__main__":
    main()
