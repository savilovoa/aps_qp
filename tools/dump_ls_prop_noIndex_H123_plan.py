from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настройки: LS_PROP в LONG_SIMPLE, qty_minus включен, INDEX_UP выключен только в SIMPLE,
    # эвристики H1+H2+H3 включены.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 600

    settings.APPLY_QTY_MINUS = True

    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True
    settings.APPLY_STRATEGY_PENALTY = True

    settings.APPLY_INDEX_UP = True
    settings.SIMPLE_DISABLE_INDEX_UP = True

    settings.SIMPLE_DEBUG_H_START = True
    settings.SIMPLE_DEBUG_H_MODE = "H123"

    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    if isinstance(res, dict):
        status = res.get("status", -1)
        status_str = res.get("status_str", "")
        schedule = res.get("schedule", [])
    else:
        status = res.status
        status_str = res.status_str
        schedule = res.schedule

    print(f"status={status} ({status_str})")

    # Определяем горизонт в модельных днях (по schedule).
    if not schedule:
        print("Пустое расписание")
        return

    max_day = max(s["day_idx"] for s in schedule)
    num_days = max_day + 1
    num_machines = len(machines)

    # Построим матрицу machine x day -> product_idx (внутренний индекс SIMPLE).
    table: list[list[int | None]] = [[None for _ in range(num_days)] for _ in range(num_machines)]
    for s in schedule:
        m = s["machine_idx"]
        d = s["day_idx"]
        p = s["product_idx"]
        if 0 <= m < num_machines and 0 <= d < num_days:
            table[m][d] = p

    # Выведем строки: машина, тип, цех(div), начальный индекс, затем индексы по дням (01,01,03,...)
    print("\nМашина\ttype\tdiv\tinit_idx\tдни (индексы продуктов через запятую)")

    for m_idx in range(num_machines):
        m_data = data["machines"][m_idx]
        name = m_data.get("name", f"m{m_idx}")
        m_type = m_data.get("type", 0)
        m_div = m_data.get("div", 1)
        init_idx = m_data.get("product_idx", 0)

        row = table[m_idx]
        idx_strs: list[str] = []
        for p in row:
            if p is None:
                idx_strs.append("00")
            else:
                idx_strs.append(f"{int(p):02d}")
        seq = ",".join(idx_strs)
        print(f"{name}\t{m_type}\t{m_div}\t{init_idx:02d}\t{seq}")


if __name__ == "__main__":  # pragma: no cover
    main()
