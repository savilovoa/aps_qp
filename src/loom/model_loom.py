from pydantic import BaseModel, Field
from datetime import date

class Machine(BaseModel):
    idx: int = Field(description="индекс")
    name: str
    product_idx: int = Field(description="индекс продукта на машине")
    id: str
    type: int

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

class LoomPlan(BaseModel):
    machine_idx: int
    day_idx: int
    product_idx: int | None

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
