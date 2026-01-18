import json
from pathlib import Path
from typing import Any


def relax_qty_minus(
    data: dict[str, Any],
    step_minus: int = 1,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Relax qty_minus for biggest products.

    Heuristic: for products with largest qty (idx > 0), set
      qty_minus = 1
      qty_minus_min = max(qty - lday * step_minus, 0)

    If top_k is provided, only the top_k products by qty are relaxed.
    """

    products: list[dict[str, Any]] = data.get("products", [])

    # Соберём кандидатов: реальные продукты (idx > 0) с qty > 0
    candidates: list[dict[str, Any]] = [
        p for p in products
        if int(p.get("idx", 0)) > 0 and int(p.get("qty", 0)) > 0
    ]

    # Отсортируем по убыванию qty
    candidates.sort(key=lambda p: int(p.get("qty", 0)), reverse=True)

    if top_k is not None:
        candidates = candidates[:top_k]

    # Мэп idx -> продукт для быстрого доступа
    by_idx: dict[int, dict[str, Any]] = {
        int(p.get("idx", 0)): p for p in products
    }

    for p in candidates:
        idx = int(p.get("idx", 0))
        qty = int(p.get("qty", 0))
        lday = int(p.get("lday", 0))

        # Новый минимум: qty - lday * step_minus
        new_min = qty - lday * step_minus
        if new_min < 0:
            new_min = 0

        target = by_idx.get(idx)
        if target is None:
            continue

        # Переводим в режим qty_minus=1 и задаём новый минимум
        target["qty_minus"] = 1
        target["qty_minus_min"] = new_min

    return data


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Relax qty_minus for largest-qty products: "
            "qty_minus=1, qty_minus_min=qty-lday*step_minus"
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
        "--step-minus",
        type=int,
        default=1,
        help="Step multiplier for lday in qty_minus_min = qty - lday * step_minus (default: 1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Relax only top-K products by qty (default: all with qty>0)",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    with in_path.open(encoding="utf8") as f:
        data = json.load(f)

    data_relaxed = relax_qty_minus(data, step_minus=args.step_minus, top_k=args.top_k)

    with out_path.open("w", encoding="utf8") as f:
        json.dump(data_relaxed, f, ensure_ascii=False, indent=2)

    print(
        f"Updated file saved to {out_path}. "
        f"step_minus={args.step_minus}, top_k={args.top_k}"
    )


if __name__ == "__main__":
    main()
