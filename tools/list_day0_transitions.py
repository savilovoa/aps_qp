import json
from pathlib import Path

from src.loom.schedule_loom import create_schedule_init


def list_day0_transitions(input_path: Path) -> None:
    with input_path.open(encoding="utf8") as f:
        data = json.load(f)

    machines = data["machines"]
    products = data["products"]
    cleans = data["cleans"]
    count_days = int(data["count_days"])
    max_daily_prod_zero = int(data["max_daily_prod_zero"])

    schedule, objective_value, deviation_proportion, count_product_zero = create_schedule_init(
        machines=machines,
        products=products,
        cleans=cleans,
        count_days=count_days,
        max_daily_prod_zero=max_daily_prod_zero,
    )

    print(f"Input: {input_path}")
    print(f"  objective_value={objective_value}, deviation_proportion={deviation_proportion}, count_product_zero={count_product_zero}")

    day0_transitions = []
    for m_idx, row in enumerate(schedule):
        p0 = row[0]
        if p0 == 0:  # PRODUCT_ZERO на первый день
            m = machines[m_idx]
            day0_transitions.append((m_idx, m.get("name", ""), m.get("product_idx", None)))

    print("  Machines with PRODUCT_ZERO on day 0 (greedy init):")
    if not day0_transitions:
        print("    NONE")
    else:
        for m_idx, name, product_idx in day0_transitions:
            print(f"    machine_idx={m_idx}, name='{name}', initial_product_idx={product_idx}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="List machines that have PRODUCT_ZERO on day 0 in greedy init schedule.",
    )
    parser.add_argument(
        "input",
        help="Path to input JSON file (e.g. example/middle_test_in.json or example/test_in.json)",
    )

    args = parser.parse_args()
    list_day0_transitions(Path(args.input))


if __name__ == "__main__":
    main()
