from pydantic import BaseModel, Field
from datetime import date
from typing import Optional

class Machine(BaseModel):
    idx: int = Field(description="индекс")
    name: str
    product_idx: int = Field(description="индекс продукта на машине")
    id: str
    type: int
    remain_day: int = Field(default=0)
    reserve: bool = Field(default=False, description="признак резервирования машины")

class Remain(BaseModel):
    idx: int
    name: str
    qty: int  = Field(description="количество остатка на начало")


class Resources(BaseModel):
    resource_idx: int = Field(description="индекс остатков")
    qty: int = Field(description="Потребность в ресурсах")

class Product(BaseModel):
    idx: int
    name: str
    qty: int = Field(description="количество запланировать")
    id: str
    machine_type: int
    resource: list[Resources] = Field(default=[])
    qty_minus: int = Field(default=0)
    lday: int = Field(description="количество смен после перехода в основе")
    src_root: int = Field(default=-1, description="индекс главного сырья из таблицы remains")
    qty_minus_min: int = Field(default=0, description="минимальное количество планирования")
    sr: bool = Field(default=False, description="признак специального продукта")
    strategy: str = Field(default="--", description="стратегия планирования (--, -, =, +, ++)")

class Clean(BaseModel):
    day_idx: int
    machine_idx: int

class DataLoomIn(BaseModel):
    machines: list[Machine]
    remains: list[Remain]
    products: list[Product]
    cleans: list[Clean]
    max_daily_prod_zero: int = Field(description="Максимальное количество перезаправок в день")
    count_days: int = Field(description="Количество дней планирования")
    dt_begin: date
    apply_qty_minus: Optional[bool] = Field(description="Применить ограничение на количество минус", default=None)
    apply_index_up: Optional[bool] = Field(description="Применить ограничение на индекс up", default=None)

class LoomPlan(BaseModel):
    machine_idx: int
    day_idx: int
    product_idx: int | None
    days_in_batch: int | None
    prev_lday: int | None

class LoomPlansOut(BaseModel):
    status: int = Field(default=0)
    status_str: str = Field(default="")
    error_str: str = Field(default="")
    schedule: list[LoomPlan] = Field(default=[])
    objective_value: int = Field(default=0)
    proportion_diff: int = Field(default=0)
    res_html: str = Field(default="")

class LoomPlansViewIn(BaseModel):
    machines: list[Machine]
    products: list[Product]
    count_days: int = Field(description="Количество дней планирования")
    schedule: list[LoomPlan]
    dt_begin: date

class LoomPlansViewOut(BaseModel):
    res_html: str
    error_str: str = Field(default="")

class LoomPlansViewByIdIn(BaseModel):
    id: str
