import pandas as pd
import json

with open("test_in.json", encoding="utf8") as f:
    test_in = f.read()

data = json.loads(test_in)

# --- Инициализация данных ---
machines_df = pd.DataFrame(data['machines'])
products_df = pd.DataFrame(data['products'])
count_days = data["count_days"]
max_daily_prod_zero = data["max_daily_prod_zero"]
num_machines = len(machines_df)
days = list(range(count_days))

# Инициализация расписания
schedule = [[None for _ in range(count_days)] for _ in range(num_machines)]

# Заполнение дней очистки
for clean in data.get('cleans', []):
    machine_idx = clean['machine_idx']
    day_idx = clean['day_idx']
    if 0 <= machine_idx < num_machines and 0 <= day_idx < count_days:
        schedule[machine_idx][day_idx] = -2


# Функция проверки возможности размещения перехода
def can_place_zero(day, zeros_per_day, max_daily_prod_zero):
    return zeros_per_day.get(day, 0) < max_daily_prod_zero


# Счетчик переходов по дням
zeros_per_day = {day: 0 for day in range(count_days)}

# Добавление индекса и количества продукта
machines_df.reset_index(inplace=True)
machines_df.rename(columns={'index': 'original_index'}, inplace=True)
product_quantities = products_df['qty']
machines_df['product_qty'] = machines_df['product_idx'].map(product_quantities)

# --- Первая часть алгоритма ---
for _, machine in machines_df.iterrows():
    machine_idx = machine['idx']
    product_idx = machine['product_idx']
    qty = machine['product_qty']

    # Проверяем, нужно ли планировать начальный продукт (если qty >= count_days/2)
    if qty >= count_days / 2:
        days_planned = 0
        day = 0

        # Планируем продукт
        while days_planned < qty and day < count_days:
            if schedule[machine_idx][day] is None:  # Если день свободен
                schedule[machine_idx][day] = product_idx
                days_planned += 1
            day += 1

        # Планируем 2 дня перехода, если есть место
        if days_planned > 0 and day + 1 < count_days:
            zero_days_needed = 2
            zero_days_placed = 0
            start_day = day

            while zero_days_placed < zero_days_needed and day < count_days:
                if schedule[machine_idx][day] is None and can_place_zero(day, zeros_per_day, max_daily_prod_zero):
                    schedule[machine_idx][day] = 0
                    zeros_per_day[day] = zeros_per_day.get(day, 0) + 1
                    zero_days_placed += 1
                elif schedule[machine_idx][day] is None:
                    # Если нельзя поставить переход, продолжаем планировать тот же продукт
                    schedule[machine_idx][day] = product_idx
                    days_planned += 1
                day += 1

            # Если не удалось разместить 2 перехода, продолжаем планировать продукт
            while zero_days_placed < zero_days_needed and day < count_days:
                if schedule[machine_idx][day] is None:
                    schedule[machine_idx][day] = product_idx
                    days_planned += 1
                day += 1

        # Уменьшаем qty в products_df
        products_df.loc[products_df['idx'] == product_idx, 'qty'] -= days_planned

# --- Вторая часть алгоритма ---

# Сортировка машин: по убыванию типа и по возрастанию необходимого продукта
machines_df_sorted = machines_df.sort_values(by=['type', 'product_qty'], ascending=[False, True])

# Сортировка продуктов: по убыванию типа машины и убыванию qty
products_df_sorted = products_df.sort_values(by=['machine_type', 'qty'], ascending=[False, False])

# Список обработанных машин
processed_machines = set()

for _, machine in machines_df_sorted.iterrows():
    machine_idx = machine['idx']
    machine_type = machine['type']
    initial_product_idx = machine['product_idx']

    if machine_idx in processed_machines:
        continue

    # Проверяем, была ли машина запланирована в первой части
    is_scheduled = any(schedule[machine_idx][day] is not None and schedule[machine_idx][day] != -2
                       for day in range(count_days))

    if not is_scheduled:
        # Планируем начальный продукт
        days_planned = 0
        day = 0
        while day < count_days and days_planned < \
                products_df.loc[products_df['idx'] == initial_product_idx, 'qty'].iloc[0]:
            if schedule[machine_idx][day] is None:
                schedule[machine_idx][day] = initial_product_idx
                days_planned += 1
            day += 1

        # Планируем 2 дня перехода
        zero_days_needed = 2
        zero_days_placed = 0
        start_day = day

        while zero_days_placed < zero_days_needed and day < count_days:
            if schedule[machine_idx][day] is None and can_place_zero(day, zeros_per_day, max_daily_prod_zero):
                schedule[machine_idx][day] = 0
                zeros_per_day[day] = zeros_per_day.get(day, 0) + 1
                zero_days_placed += 1
            elif schedule[machine_idx][day] is None:
                # Если нельзя поставить переход, продолжаем планировать начальный продукт
                schedule[machine_idx][day] = initial_product_idx
                days_planned += 1
            day += 1

        # Обновляем qty для начального продукта
        products_df.loc[products_df['idx'] == initial_product_idx, 'qty'] -= days_planned

    # Планируем продукты из отсортированного списка
    for _, product in products_df_sorted.iterrows():
        product_idx = product['idx']
        product_type = product['machine_type']
        qty_needed = product['qty']

        # Проверяем совместимость типов
        if product_type == 1 and machine_type != 1:
            continue

        if qty_needed <= 0:
            continue

        # Находим первый свободный день после переходов
        day = 0
        while day < count_days and schedule[machine_idx][day] is not None:
            day += 1

        # Планируем продукт до конца дней или до достижения qty
        days_planned = 0
        while day < count_days and days_planned < qty_needed:
            if schedule[machine_idx][day] is None:
                schedule[machine_idx][day] = product_idx
                days_planned += 1
            day += 1

        # Обновляем qty
        products_df.loc[products_df['idx'] == product_idx, 'qty'] -= days_planned

        # Если запланировали достаточно или больше, переходим к следующей машине
        if days_planned >= qty_needed:
            processed_machines.add(machine_idx)
            break

    # Если машина полностью запланирована, добавляем в обработанные
    if all(schedule[machine_idx][day] is not None for day in range(count_days)):
        processed_machines.add(machine_idx)

# Считаем

# Вывод результата
print("Расписание:")
for machine_idx in range(num_machines):
    print(f"Машина {machine_idx}: {schedule[machine_idx]}")