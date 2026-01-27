from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def run_ls_scenario(label: str, mode: str, h_mode: str | None) -> None:
    """Запускает один LS-сценарий (LS_OVER или LS_PROP) в режиме LONG_SIMPLE
    с включённым APPLY_QTY_MINUS и заданной комбинацией эвристик H1–H3.

    h_mode управляет SIMPLE_DEBUG_H_MODE:
      - None / "H123": все H1, H2, H3 включены
      - "NONE": все H1–H3 выключены
      - "H1": только H1
      - "H2": только H2
      - "H3": только H3
      - "H12": H1 и H2
    """
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настройки горизонта и времени
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 120

    # Включаем qty_minus, чтобы воспроизвести рабочий режим.
    settings.APPLY_QTY_MINUS = True

    # Внутренняя пропорциональная цель SIMPLE сейчас не используется, но
    # для совместимости оставляем флаги пропорций/стратегий такими же,
    # как в compare_long_vs_simple для LS_OVER / LS_PROP.
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.APPLY_STRATEGY_PENALTY = True
    settings.APPLY_INDEX_UP = True

    if mode == "LS_OVER":
        settings.SIMPLE_USE_PROP_MULT = False
    elif mode == "LS_PROP":
        settings.SIMPLE_USE_PROP_MULT = True

    # Управляем эвристиками H1–H3 через debug-флаги.
    if h_mode is None:
        settings.__dict__["SIMPLE_DEBUG_H_START"] = False
        settings.__dict__["SIMPLE_DEBUG_H_MODE"] = None
        h_label = "H123 (all)"
    else:
        settings.__dict__["SIMPLE_DEBUG_H_START"] = True
        settings.__dict__["SIMPLE_DEBUG_H_MODE"] = h_mode
        h_label = h_mode

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

    print(f"\n=== {label} (mode={mode}, H={h_label}) ===")
    print(f"status={status} ({status_str}), objective={objective}")
    if error_str:
        print(f"error_str={error_str}")


def main() -> None:
    # Прогоняем LS_OVER и LS_PROP при различных комбинациях H1–H3 с включённым
    # APPLY_QTY_MINUS, чтобы локализовать источник невыполнимости.
    h_modes = [
        None,       # продакшен: H1+H2+H3
        "NONE",    # все эвристики выкл.
        "H1",      # только H1
        "H2",      # только H2
        "H3",      # только H3
        "H12",     # H1+H2
        "H123",    # H1+H2+H3 явно
    ]

    for mode in ("LS_OVER", "LS_PROP"):
        for h_mode in h_modes:
            label = f"QTYM_{mode}_{h_mode or 'ALL'}"
            run_ls_scenario(label, mode, h_mode)


if __name__ == "__main__":  # pragma: no cover
    main()
