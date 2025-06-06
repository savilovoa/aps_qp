import datetime

from .config import settings, logger
import uvicorn
from fastapi import FastAPI
from .routers import router
import json
from src.loom.schedule_loom import schedule_loom_calc
from example.loom_plan_view import view_schedule

app = FastAPI()

app.include_router(router, prefix="/api")

def main():
    global app
    if not settings.CALC_TEST_DATA:
        logger.info(f"Start server {settings.SERVER_NAME}")
        uvicorn.run(app, host=settings.SERVER_NAME, port=settings.PORT)
    else:
        calc_test_data_from()


def calc_test_data_from():
    # Исходные данные
    with open("example/test_in.json", encoding="utf8") as f:
        test_in = f.read()

    data = json.loads(test_in)
    machines = [(d["name"], d["product_idx"]) for d in data["machines"]]
    products = [(d["name"], d["qty"], []) for d in data["products"]]
    result = schedule_loom_calc(machines=machines, products=products, remains=[],
                                max_daily_prod_zero=data["max_daily_prod_zero"],
                                count_days=data["count_days"])
    if result["error_str"]:
        print(result["error_str"])
        return
    days = [str(d) for d in range(data["count_days"])]
    machines = [d["name"] for d in data["machines"]]
    products = [d["name"] for d in data["products"]]
    title_text = f"{result['status_str']} оптим. значение {result['objective_value']}"

    view_schedule(machines=machines, products=products, days=days, schedules=result["schedule"], title_text=title_text)



if __name__ == "__main__":
    print(settings.CALC_TEST_DATA)
    if not settings.CALC_TEST_DATA:
        main()
    else:
        calc_test_data_from()
