import json
from pathlib import Path

from src.loom.schedule_loom import create_schedule_init


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run greedy create_schedule_init on input JSON and print stats.",
    )
    parser.add_argument(
        "input",
        help="Path to input JSON file (e.g. example/middle_test_in.json or example/test_in.json)",
    )

    args = parser.parse_args()
    in_path = Path(args.input)

    with in_path.open(encoding="utf8") as f:
        data = json.load(f)

    schedule, objective_value, deviation_proportion, count_product_zero = create_schedule_init(
        machines=data["machines"],
        products=data["products"],
        cleans=data["cleans"],
        count_days=data["count_days"],
        max_daily_prod_zero=data["max_daily_prod_zero"],
    )

    print(f"Greedy init on {in_path}:")
    print(f"  objective_value={objective_value}")
    print(f"  deviation_proportion={deviation_proportion}")
    print(f"  count_product_zero={count_product_zero}")

    # Посмотрим, не нарушаем ли дневной лимит по простоям
    D = int(data["count_days"])
    zeros_per_day = [0] * D
    for m_idx, row in enumerate(schedule):
        for d, p in enumerate(row):
            if p == 0:
                zeros_per_day[d] += 1

    print(f"  max_daily_prod_zero={data['max_daily_prod_zero']}")
    print("  zeros per day (first 30 days):")
    for d, z in list(enumerate(zeros_per_day))[:30]:
        print(f"    day {d}: zeros={z}")


if __name__ == "__main__":
    main()
