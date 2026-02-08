# This Python file uses the following encoding: utf-8
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from os import getcwd, path, makedirs
import logging
from logging.config import fileConfig

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str
    VERSION: str = Field(default="0.0.1.2")
    DEBUG: bool = Field(default=False)
    DEBUG_SCHEDULE: bool = Field(default=False, description="Подробные отладочные логи расписания (lday, батчи и т.п.)")
    PORT: int
    SERVER_NAME: str = Field(default="0.0.0.0")
    BASE_DIR: str = Field(default=getcwd())
    LOOM_MAX_TIME: int = Field(default=200)
    LOOM_NUM_WORKERS: int = Field(default=1)
    CALC_TEST_DATA: bool = Field(default=False)
    TEST_INPUT_FILE: str | None = Field(default=None, description="Путь к файлу тестовых данных для CALC_TEST_DATA")
    SOLVER_ENUMERATE: bool = Field(default=False)
    SOLVER_ENUMERATE_COUNT: int = Field(default=3)
    APPLY_QTY_MINUS: bool = Field(default=True)
    APPLY_INDEX_UP: bool = Field(default=True)

    # Учитывать ли разделение по цехам (div) в моделях.
    # Если False, ограничения совместимости по div и выбор единственного цеха
    # для flex-продуктов (div=0) в CP-моделях отключаются, но разбиение по div
    # в отчётах (HTML) остаётся.
    APPLY_DIV_CONSTRAINTS: bool = Field(
        default=True,
        description="Включать ли ограничения по цехам (div) в моделях FULL/LONG/LONG_SIMPLE",
    )

    # В SIMPLE можно отдельно отключать INDEX_UP, не трогая LONG/полную модель.
    SIMPLE_DISABLE_INDEX_UP: bool = Field(
        default=False,
        description="Если True, в LONG_SIMPLE игнорируем APPLY_INDEX_UP и не накладываем индекс только вверх",
    )
    APPLY_DOWNTIME_LIMITS: bool = Field(default=True)

    # Режим горизонта планирования: FULL (по умолчанию), LONG (упрощённый 84 смены), SHORT (детализированный 21 смена)
    HORIZON_MODE: str = Field(default="FULL")

    # Мастер-флаг бизнес-логики переходов (двухдневный переход, запрет 3-го нуля и т.п.)
    APPLY_TRANSITION_BUSINESS_LOGIC: bool = Field(
        default=True,
        description="Включать ли полную бизнес-логику переходов (двухдневный переход, запрет 3-го нуля и пр.)",
    )

    # Детальная настройка ограничений по простоям (PRODUCT_ZERO)
    # Если APPLY_DOWNTIME_LIMITS=False, оба ограничения полностью отключаются.
    APPLY_ZERO_PER_DAY_LIMIT: bool = Field(
        default=True,
        description="Ограничивать количество PRODUCT_ZERO по всем машинам в каждый день (max_daily_prod_zero)",
    )
    APPLY_ZERO_PER_MACHINE_LIMIT: bool = Field(
        default=True,
        description="Ограничивать суммарное количество PRODUCT_ZERO на машину за период (не более одного перехода в неделю)",
    )
    APPLY_THIRD_ZERO_BAN: bool = Field(
        default=True,
        description="Запрещать ли третий подряд PRODUCT_ZERO после двухдневного перехода",
    )
    APPLY_PROP_OBJECTIVE: bool = Field(default=True, description="Включать ли штраф за отклонение от пропорций qty в целевую функцию")
    APPLY_STRATEGY_PENALTY: bool = Field(default=True, description="Включать ли штрафы по стратегиям (--, -, =, +, ++)")
    KFZ_DOWNTIME_PENALTY: int = Field(default=10)
    USE_GREEDY_HINT: bool = Field(default=False, description="Использовать ли жадный предварительный план как hint для CP-SAT")

    # Альтернативная цель: линейный штраф за превышение плана вместо пропорций
    APPLY_OVERPENALTY_INSTEAD_OF_PROP: bool = Field(
        default=False,
        description="Если True, в LONG/LONG_SIMPLE используем линейный штраф за превышение плана вместо пропорциональной цели",
    )

    # Для LONG_SIMPLE: использовать ли "полный" алгоритм пропорций (через умножение),
    # вместо упрощённого линейного штрафа. Работает только при APPLY_PROP_OBJECTIVE=True.
    SIMPLE_USE_PROP_MULT: bool = Field(
        default=False,
        description="В LONG_SIMPLE использовать алгоритм пропорций с умножением вместо линейного штрафа",
    )

    # Отладка эвристик начального плана H1–H3 в LONG_SIMPLE.
    SIMPLE_DEBUG_H_START: bool = Field(
        default=False,
        description="Включить отладочный режим управления эвристиками H1–H3 в LONG_SIMPLE",
    )
    SIMPLE_DEBUG_H_MODE: str | None = Field(
        default=None,
        description=(
            "Комбинация эвристик H1–H3 в debug-режиме: NONE, H1, H2, H3, H12, H123. "
            "В production-режиме (SIMPLE_DEBUG_H_START=False) игнорируется."
        ),
    )

    # Отладочный фильтр для блока APPLY_QTY_MINUS в LONG_SIMPLE: если задан, то
    # нижние границы по qty_minus применяются только к этим индексам продуктов.
    SIMPLE_QTY_MINUS_SUBSET: set[int] | None = Field(
        default=None,
        description="Подмножество индексов продуктов для применения qty_minus в LONG_SIMPLE (отладка)",
    )

    # Отладка минимально достижимого объёма для конкретного продукта в LONG_SIMPLE:
    # только базовые домены jobs[m,d], связь с product_counts[p] и тип/div.
    SIMPLE_DEBUG_SUPER_SIMPLE: bool = Field(
        default=False,
        description=(
            "Если True, в LONG_SIMPLE строится максимально простая модель без "
            "qty_minus, монотонности и ограничений по переходам (для отладки)."
        ),
    )

    # Отладка минимально достижимого объёма для конкретного продукта в LONG_SIMPLE:
    # если задан индекс, то в SIMPLE вместо обычной цели минимизируем
    # product_counts[idx], чтобы оценить нижнюю границу по объёму с учётом всех
    # текущих ограничений.
    SIMPLE_DEBUG_MINIMIZE_PRODUCT_IDX: int | None = Field(
        default=None,
        description=(
            "Если не None, в LONG_SIMPLE используем цель Minimize(product_counts[idx]) "
            "для оценки минимально достижимого объёма по этому продукту."
        ),
    )

    # Отладочные верхние границы для отдельных продуктов в LONG_SIMPLE:
    # если заданы, то добавляем ограничения product_counts[p] <= cap_days.
    SIMPLE_DEBUG_PRODUCT_UPPER_CAPS: dict[int, int] | None = Field(
        default=None,
        description=(
            "Словарь {product_idx: cap_days} для введения точечных верхних "
            "границ на product_counts[p] в LONG_SIMPLE (отладка перепроизводства)."
        ),
    )

    # Отладка: дамп линейных ограничений для конкретного продукта в LONG_SIMPLE.
    # SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX задаётся во внешних idx (как в JSON),
    # SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX_INTERNAL – соответствующий внутренний idx
    # (заполняется в schedule_loom_calc после переотображения idx).
    SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX: int | None = Field(
        default=None,
        description="Внешний idx продукта, для которого нужно сделать отладочный дамп линейных ограничений в LONG_SIMPLE",
    )
    SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX_INTERNAL: int | None = Field(
        default=None,
        description="Внутренний idx продукта для дампа ограничений (служебное поле, заполняется в runtime)",
    )

    # Включать ли монотонность по C[p,d] (машино-дни по продукту в день) в LONG_SIMPLE.
    # При SIMPLE_USE_MONOTONE_COUNTS=False блок монотонности в create_model_simple
    # отключается, что может помочь диагностировать конфликты с нижними границами
    # по qty_minus.
    SIMPLE_USE_MONOTONE_COUNTS: bool = Field(
        default=True,
        description=(
            "Если True, в LONG_SIMPLE накладывается монотонность C[p,d] по дням; "
            "если False — монотонность C[p,d] отключена."
        ),
    )

    # Максимальное число балансирующих продуктов qty_minus!=0 на один цех (div)
    # в LONG_SIMPLE. Эти продукты могут свободнее отклоняться от плана, все
    # остальные получают жёсткий коридор вокруг планового объёма.
    SIMPLE_QTY_MINUS_MAX_BALANCE_PER_DIV: int = Field(
        default=1,
        description=(
            "Максимальное количество балансирующих продуктов qty_minus!=0 на один "
            "цех (div) в LONG_SIMPLE. Остальные flex-продукты получают жёсткий "
            "коридор вокруг плана в днях."
        ),
    )

    # Дополнительная крупная эвристика для LONG_SIMPLE: большие продукты без
    # стартовых машин (machines.product_idx == p) могут полностью занимать одну
    # подходящую машину на весь горизонт. В этом случае продукт p запрещается на
    # остальных машинах. Включена опционально, чтобы можно было легко сравнивать
    # поведение модели с/без этой эвристики.
    # Флаг для крупной эвристики big-no-start. Храним как строку/None, чтобы
    # избежать жёсткого bool-parsing от pydantic для разных значений env.
    # Флаг эвристики big-no-start в LONG_SIMPLE. Если непустая строка,
    # считается включённой (True); если None или пусто — выключена.
    SIMPLE_ENABLE_BIG_NOSTART_HEURISTIC: str | None = Field(
        default=None,
        description=(
            "Флаг эвристики big-no-start в LONG_SIMPLE. Если непустая строка, "
            "считается включённой (True); если None или пусто — выключена."
        ),
    )

    # Эвристика H6: фиксация продуктов, которые занимают полные машины.
    # Если True, находим продукты, которые есть на старте и чей план
    # заполняет целые машины, и фиксируем их.
    SIMPLE_ENABLE_HEURISTIC_H6: bool = Field(
        default=True,
        description="Включить эвристику H6 (фиксация полных машин для продуктов со старта)",
    )

    # Минимальная длина партии (в модельных днях) при переходе на новый продукт.
    # Работает только при SIMPLE_USE_MONOTONE_COUNTS=True (use_machine_contiguity).
    SIMPLE_MIN_BATCH_DAYS: int = Field(
        default=3,
        description="Минимальная непрерывная длительность работы продукта на машине (в днях) после старта",
    )
    
    # Путь для сохранения результата в JSON (для интеграции Two-Phase)
    # Если None, не сохраняем.
    SAVE_RESULT_JSON_PATH: str | None = Field(
        default=None,
        description="Путь к файлу для сохранения результата расчета (JSON). Используется в Two-Phase."
    )

settings = Settings()

log_path = settings.BASE_DIR + "/log"
if not path.exists(log_path):
    try:
        makedirs(log_path)
        print(f"Каталог '{log_path}' успешно создан.")
    except OSError as e:
        print(f"Ошибка при создании каталога '{log_path}': {e}")
        raise

fileConfig(settings.BASE_DIR + r'/logging.ini')
logger = logging.getLogger(settings.PROJECT_NAME)
