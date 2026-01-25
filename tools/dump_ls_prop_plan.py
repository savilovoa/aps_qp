from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настраиваем режим LS_PROP
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True

    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    schedule = res["schedule"]  # list of dicts: machine_idx, day_idx, product_idx, ...

    if not schedule:
        print("EMPTY SCHEDULE")
        return

    num_machines = max(rec["machine_idx"] for rec in schedule) + 1
    num_days = max(rec["day_idx"] for rec in schedule) + 1

    # строим матрицу product_idx по (m,d)
    mat: list[list[int | None]] = [[None for _ in range(num_days)] for _ in range(num_machines)]
    for rec in schedule:
        m = rec["machine_idx"]
        d = rec["day_idx"]
        p = rec["product_idx"]
        mat[m][d] = p

    # Вывод в формате: name,div,type,start_idx: d1,d2,...
    for m in range(num_machines):
        m_data = data["machines"][m]
        name = m_data.get("name", f"loom {m}")
        div = m_data.get("div", 1)
        m_type = m_data.get("type", 0)
        start_idx = int(m_data.get("product_idx", 0))

        # форматируем индексы как двухзначные с ведущим нулём (01, 02, ...)
        def fmt_idx(x: int | None) -> str:
            if x is None:
                return "00"  # чистки/пустые дни
            return f"{int(x):02d}"

        day_str = ",".join(fmt_idx(mat[m][d]) for d in range(num_days))
        print(f"{name},{div},{m_type},{fmt_idx(start_idx)}: {day_str}")


if __name__ == "__main__":  # pragma: no cover
    main()
