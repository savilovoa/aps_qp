from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def run_ls_scenario(label: str, mode: str, h_mode: str | None) -> None:
    """Запускает один LS-сценарий (LS_OVER или LS_PROP) в режиме LONG_SIMPLE
    с заданной эвристикой h_mode: None, "H1", "H2" или "H3".
    """
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настройки горизонта и времени
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 360

    # Отключаем все объёмные ограничения/цели, оставляем только переходы и индекс up.
    settings.APPLY_QTY_MINUS = False
    settings.APPLY_PROP_OBJECTIVE = False
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.APPLY_STRATEGY_PENALTY = False
    settings.APPLY_INDEX_UP = True

    # Отдельно настраиваем SIMPLE_USE_PROP_MULT для LS_PROP
    if mode == "LS_OVER":
        settings.SIMPLE_USE_PROP_MULT = False
    elif mode == "LS_PROP":
        settings.SIMPLE_USE_PROP_MULT = True

    # Включаем/выключаем эвристику начального плана в SIMPLE.
    if h_mode is None:
        settings.__dict__["SIMPLE_DEBUG_H_START"] = False
        settings.__dict__["SIMPLE_DEBUG_H_MODE"] = None
    else:
        settings.__dict__["SIMPLE_DEBUG_H_START"] = True
        settings.__dict__["SIMPLE_DEBUG_H_MODE"] = h_mode

    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    # schedule_loom_calc в норме возвращает LoomPlansOut, но при ранней
    # ошибке может вернуть dict; поддержим оба варианта.
    if isinstance(res, dict):
        status = res.get("status", -1)
        status_str = res.get("status_str", "")
        objective = res.get("objective_value", 0)
        error_str = res.get("error_str", "")
    else:
        status = res.status
        status_str = res.status_str
        objective = res.objective_value
        error_str = res.error_str

    print(f"\n=== {label} (mode={mode}, H={h_mode or 'NONE'}) ===")
    print(f"status={status} ({status_str}), objective={objective}")
    if error_str:
        print(f"error_str={error_str}")


def main() -> None:
    # H1 и H2 теперь включены всегда в модели. Здесь проверяем H3 (крупные
    # продукты, забивающие станок) поверх H1+H2 для LS_OVER и LS_PROP в
    # режиме LONG_SIMPLE, без объёмных ограничений, только переходы + index_up.
    for mode in ("LS_OVER", "LS_PROP"):
        run_ls_scenario("H_START_H3", mode, "H3")


if __name__ == "__main__":  # pragma: no cover
    main()
