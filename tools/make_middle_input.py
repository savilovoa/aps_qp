import json
import copy
from pathlib import Path

SRC = Path("example/test_in.json")
DST = Path("example/middle_test_in.json")

MAX_MACHINES = 12
MAX_DAYS = 42


def main() -> None:
    with SRC.open(encoding="utf8") as f:
        data = json.load(f)

    middle = copy.deepcopy(data)

    middle["machines"] = [
        m for m in data["machines"] if m.get("idx", 0) < MAX_MACHINES
    ]

    middle["cleans"] = [
        c
        for c in data["cleans"]
        if c.get("machine_idx", 0) < MAX_MACHINES and c.get("day_idx", 0) < MAX_DAYS
    ]

    middle["count_days"] = MAX_DAYS

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf8") as f:
        json.dump(middle, f, ensure_ascii=False, indent=2)

    print(
        f"created {DST} machines={len(middle['machines'])}, days={middle['count_days']}"
    )


if __name__ == "__main__":
    main()
