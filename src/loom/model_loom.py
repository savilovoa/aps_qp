from pydantic import BaseModel, Field

class Machine(BaseModel):
    idx: int = Field(description="индекс")
    name: str
    product_idx: int = Field(description="индекс продукта на машине")

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
    resource: list[Resources] = Field(default=[])

class DataLoomIn(BaseModel):
    machines: list[Machine]
    remains: list[Remain]
    products: list[Product]
    max_daily_prod_zero: int = Field(description="Максимальное количество перезаправок в день")
    count_days: int = Field(description="Количество дней планирования")

class LoomPlan(BaseModel):
    machine_idx: int
    day_idx: int
    product_idx: int

class ProductPlan(BaseModel):
    product_idx: int
    qty: int
    penalty: int

class DayZero(BaseModel):
    day_idx: int
    count_zero: int

class LoomPlansOut(BaseModel):
    status: int = Field(default=0)
    status_str: str = Field(default="")
    error_str: str = Field(default="")
    schedule: list[LoomPlan]
    products: list[ProductPlan]
    zeros: list[DayZero]
    objective_value: int = Field(default=0)
    proportion_diff: int = Field(default=0)





