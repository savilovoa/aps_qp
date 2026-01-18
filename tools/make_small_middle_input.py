import json
import copy
from pathlib import Path

SRC = Path("example/middle_test_in.json")
DST = Path("example/middle_small_in.json")

MAX_MACHINES = 4
MAX_DAYS = 21


def main() -> None:
    with SRC.open(encoding="utf8") as f:
        data = json.load(f)

    small = copy.deepcopy(data)

    small["machines"] = [
        m for m in data["machines"] if m.get("idx", 0) < MAX_MACHINES
    ]

    small["cleans"] = [
        c
        for c in data.get("cleans", [])
        if c.get("machine_idx", 0) < MAX_MACHINES and c.get("day_idx", 0) < MAX_DAYS
    ]

    small["count_days"] = MAX_DAYS

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf8") as f:
        json.dump(small, f, ensure_ascii=False, indent=2)

    print(
        f"created {DST} machines={len(small['machines'])}, days={small['count_days']}"
    )


if __name__ == "__main__":
    main()
