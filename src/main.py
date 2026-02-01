import datetime

from .config import settings, logger
import uvicorn
from fastapi import FastAPI
from .routers import router
import json
from src.loom.schedule_loom import schedule_loom_calc
#from example.loom_plan_view import view_schedule
from src.loom.loom_plan_html import schedule_to_html, aggregated_schedule_to_html
import sys
from datetime import datetime

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
    input_file = settings.TEST_INPUT_FILE or "example/test_in_new.json"
    logger.info(f"Загрузка тестовых данных из файла: {input_file}")
    with open(input_file, encoding="utf8") as f:
        test_in = f.read()

    data = json.loads(test_in)

    # Низкоуровневые структуры, как в старом коде (использует remains из JSON как есть)
    machines = [
        (d["name"], d["product_idx"], d["id"], d["type"], d["remain_day"], d["reserve"])
        for d in data["machines"]
    ]
    products = [
        (
            d["name"],
            d["qty"],
            d["id"],
            d["machine_type"],
            d["qty_minus"],
            d["lday"],
            d["src_root"],
            d["qty_minus_min"],
            d["sr"],
            d["strategy"],
        )
        for d in data["products"]
    ]
    cleans = [(d["machine_idx"], d["day_idx"]) for d in data["cleans"]]
    remains = data["remains"]

    result_calc = schedule_loom_calc(
        machines=machines,
        products=products,
        remains=remains,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )
    if result_calc["error_str"]:
        print(result_calc["error_str"])
        return

    title_text = f"{result_calc['status_str']} оптимизационное значение {result_calc['objective_value']}"
    dt_begin = datetime.strptime(data["dt_begin"], "%Y-%m-%dT%H:%M:%S").date()

    # Выбор HTML-представления в зависимости от режима горизонта
    horizon_mode = getattr(settings, "HORIZON_MODE", "FULL").upper()
    if horizon_mode in ("LONG_SIMPLE", "LONG_SIMPLE_HINT", "LONG_TWOLEVEL"):
        # Агрегируем помашинный план в long_schedule и рендерим таблицу (дни × продукты по цехам)
        counts: dict[tuple[int, int], int] = {}
        for s in result_calc["schedule"]:
            p = s["product_idx"]
            d = s["day_idx"]
            if p is None or p <= 0:
                continue
            key = (d, p)
            counts[key] = counts.get(key, 0) + 1
        long_schedule = [
            {"day_idx": d, "product_idx": p, "machine_count": c}
            for (d, p), c in sorted(counts.items())
        ]
        result_html = aggregated_schedule_to_html(
            machines=data["machines"],
            schedule=result_calc["schedule"],
            products=data["products"],
            long_schedule=long_schedule,
            dt_begin=dt_begin,
            title_text=title_text,
        )
    else:
        # Детальное представление по машинам/сменам
        days = [d for d in range(data["count_days"])]
        machines_names = [d["name"] for d in data["machines"]]
        products_names = [d["name"] for d in data["products"]]
        result_html = schedule_to_html(
            machines=machines_names,
            products=products_names,
            days=days,
            schedules=result_calc["schedule"],
            dt_begin=datetime.strptime(data["dt_begin"], "%Y-%m-%dT%H:%M:%S"),
            title_text=title_text,
        )

    f_name = "example/res.html"
    with open(f_name, "w", encoding="utf8") as f:
        f.write(result_html)

    # original_stdout = sys.stdout  # Store original stdout
    # with open("example/view.txt", encoding="utf8", mode="w+") as f:
    #     sys.stdout = f  # Redirect stdout to the file
    #     print(machines)
    #     print(products)
    #     print(days)
    #     print(result["schedule"])
    # sys.stdout = original_stdout  # Restore original stdout

    #view_schedule(machines=machines, products=products, days=days, schedules=result["schedule"], title_text=title_text)



if __name__ == "__main__":
    print(settings.CALC_TEST_DATA)
    if not settings.CALC_TEST_DATA:
        main()
    else:
        calc_test_data_from()
