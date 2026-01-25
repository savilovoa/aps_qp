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
