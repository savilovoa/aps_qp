import pandas as pd
import json
import numpy as np

def create_schedule_init(machines, products, cleans, count_days, max_daily_prod_zero):
    machines_df = pd.DataFrame(machines)
    products_df = pd.DataFrame(products)
    days = [item for item in range(count_days)]
    num_machines = len(machines_df)
    num_products = len(products_df)

    schedule = [[None for _ in range(count_days)] for _ in range(num_machines)]

    # Предварительно заполняем дни очистки (cleans) - это жесткие ограничения
    for clean in cleans:
        machine_idx = clean['machine_idx']
        day_idx = clean['day_idx']
        if 0 <= machine_idx < num_machines and 0 <= day_idx < count_days:
            schedule[machine_idx][day_idx] = -2

    # Подсчет рабочих дней без чисток
    work_days = count_days * len(machines) - sum(
        1 for clean in cleans
    )
    # Функция для проверки возможности установки перехода (prod_zero) в день
    def can_place_zero(day, zeros_per_day, max_daily_prod_zero):
        return zeros_per_day.get(day, 0) < max_daily_prod_zero

    # Счетчик переходов по дням
    zeros_per_day = {day: 0 for day in range(count_days)}

    # --- Шаг 3: Добавление колонки с индексом объекта ---
    machines_df.reset_index(inplace=True)
    machines_df.rename(columns={'index': 'original_index'}, inplace=True)

    # --- Шаг 4: Добавление колонки 'product_qty' ---
    product_quantities = products_df['qty']
    machines_df['product_qty'] = machines_df['product_idx'].map(product_quantities)

    # Правим количество пропорционально
    # Считаем мин и мак чисток
    next_count_min = 0
    for idx, machine in machines_df.iterrows():
        if machine['product_qty'] < count_days / 2:
            next_count_min += 1
    next_count_max = round(count_days / 2)
    if next_count_max < next_count_min:
        next_count = next_count_max
    else:
        next_count = next_count_min * round((next_count_max - next_count_min) / 2)
    work_days = work_days - next_count * 2
    count_qty = 0
    for qty in products_df[products_df['idx'] > 0]['qty']:
        count_qty += qty
    # Считаем коэф увеличения/уменьшения
    kf_count = 0.9 * work_days / count_qty
    for product_idx in range(1, num_products):
        qty = products_df.at[product_idx, 'qty']
        if qty <= 0:
            continue
        qty = round(qty * kf_count)
        products_df.at[product_idx, 'qty'] = qty
    product_quantities = products_df['qty']
    machines_df['product_qty'] = machines_df['product_idx'].map(product_quantities)

    # --- Шаг 5: Добавление колонки 'day_remains' ---
    machines_df['day_remains'] = count_days
    for machine_idx in range(num_machines):
        clean_days = sum(1 for clean in cleans if clean['machine_idx'] == machine_idx)
        machines_df.at[machine_idx, 'day_remains'] -= clean_days

    # --- Первая часть алгоритма ---
    for machine_idx in range(num_machines):
        product_idx = machines_df.at[machine_idx, 'product_idx']
        qty_needed = machines_df.at[machine_idx, 'product_qty']
        if qty_needed >= count_days / 2:
            days_to_plan = min(int(qty_needed), machines_df.at[machine_idx, 'day_remains'])
            day_idx = 0
            days_planned = 0
            while days_planned < days_to_plan and day_idx < count_days:
                if schedule[machine_idx][day_idx] is None:  # Проверяем, что день не занят чисткой
                    schedule[machine_idx][day_idx] = int(product_idx)
                    days_planned += 1
                    machines_df.at[machine_idx, 'day_remains'] -= 1
                    products_df.at[product_idx, 'qty'] -= 1
                    machines_df.at[machine_idx, 'product_qty'] -= 1
                day_idx += 1

            # Планируем переход (2 дня с product_idx = 0)
            if days_planned >= 1 and day_idx + 1 < count_days:
                zero_days = 0
                start_day = day_idx
                while zero_days < 2 and day_idx < count_days:
                    if schedule[machine_idx][day_idx] is None:
                        if can_place_zero(day_idx, zeros_per_day, max_daily_prod_zero):
                            schedule[machine_idx][day_idx] = 0
                            zeros_per_day[day_idx] = zeros_per_day.get(day_idx, 0) + 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                            zero_days += 1
                        else:
                            # Если нельзя поставить переход, продолжаем планировать тот же продукт
                            schedule[machine_idx][day_idx] = int(product_idx)
                            products_df.at[product_idx, 'qty'] -= 1
                            machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                    day_idx += 1
                # Если осталось менее 3 дней до конца, продолжаем планировать тот же продукт
                if day_idx >= count_days - 2:
                    while day_idx < count_days:
                        if schedule[machine_idx][day_idx] is None:
                            schedule[machine_idx][day_idx] = int(product_idx)
                            products_df.at[product_idx, 'qty'] -= 1
                            machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                        day_idx += 1

    # --- Вторая часть алгоритма ---
    def schedule_remaining_days(machine_type_filter=None):
        # Сортировка машин
        machines_to_schedule = machines_df.copy()
        if machine_type_filter is not None:
            machines_to_schedule = machines_to_schedule[machines_to_schedule['type'] == machine_type_filter]
        machines_to_schedule = machines_to_schedule.sort_values(
            by=['type', 'product_qty', 'day_remains'], ascending=[False, True, True]
        )

        # Сортировка продуктов
        products_to_schedule = products_df[products_df['qty'] > 0].copy()
        if machine_type_filter is not None:
            products_to_schedule = products_to_schedule[products_to_schedule['machine_type'] == machine_type_filter]
        products_to_schedule = products_to_schedule.sort_values(
            by=['machine_type', 'qty'], ascending=[False, False]
        )

        while not machines_to_schedule.empty and not products_to_schedule.empty:
            product_idx = products_to_schedule.iloc[0]['idx']
            qty_needed = products_to_schedule.iloc[0]['qty']
            product_machine_type = products_to_schedule.iloc[0]['machine_type']

            # Проверяем, есть ли машина с совпадающим начальным продуктом
            matching_machines = machines_to_schedule[machines_to_schedule['product_idx'] == product_idx]
            if not matching_machines.empty:
                machine_idx = matching_machines.iloc[0]['original_index']
            else:
                # Выбираем первую машину из отсортированного списка
                machine_idx = machines_to_schedule.iloc[0]['original_index']

            machine_type = machines_df[machines_df['original_index'] == machine_idx]['type'].iloc[0]
            # Проверяем совместимость типов
            if product_machine_type == 1 and machine_type != 1:
                products_to_schedule = products_to_schedule.iloc[1:]
                continue

            # Находим первый свободный день
            start_day = 0
            for day in range(count_days):
                if schedule[machine_idx][day] is not None:
                    start_day = day + 1
                else:
                    break

            # Если начальный продукт совпадает, планируем его до конца
            if machines_df[machines_df['original_index'] == machine_idx]['product_idx'].iloc[0] == product_idx:
                days_planned = 0
                for day in range(start_day, count_days):
                    if schedule[machine_idx][day] is None:
                        schedule[machine_idx][day] = int(product_idx)
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[machine_idx, 'product_qty'] -= 1
                        machines_df.at[machine_idx, 'day_remains'] -= 1
                        days_planned += 1
                machines_to_schedule = machines_to_schedule[machines_to_schedule['original_index'] != machine_idx]
            else:
                # Планируем начальный продукт, если он есть в products_df
                initial_product_idx = machines_df[machines_df['original_index'] == machine_idx]['product_idx'].iloc[0]
                initial_qty = products_df[products_df['idx'] == initial_product_idx]['qty']
                if not initial_qty.empty and initial_qty.iloc[0] > 0:
                    days_planned = 0
                    day_idx = start_day
                    while day_idx < count_days and days_planned < initial_qty.iloc[0]:
                        if schedule[machine_idx][day_idx] is None:
                            schedule[machine_idx][day_idx] = int(initial_product_idx)
                            products_df.at[initial_product_idx, 'qty'] -= 1
                            machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                            days_planned += 1
                        day_idx += 1
                    start_day = day_idx

                # Планируем переход (2 дня с product_idx = 0)
                zero_days = 0
                day_idx = start_day
                while zero_days < 2 and day_idx < count_days:
                    if schedule[machine_idx][day_idx] is None:
                        if can_place_zero(day_idx, zeros_per_day, max_daily_prod_zero):
                            schedule[machine_idx][day_idx] = 0
                            zeros_per_day[day_idx] = zeros_per_day.get(day_idx, 0) + 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                            zero_days += 1
                        else:
                            # Планируем начальный продукт, если переход невозможен
                            schedule[machine_idx][day_idx] = int(initial_product_idx)
                            if products_df.at[initial_product_idx, 'qty'] > 0:
                                products_df.at[initial_product_idx, 'qty'] -= 1
                                machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                        day_idx += 1
                    else:
                        day_idx += 1
                    start_day = day_idx

                # Планируем выбранный продукт до конца дней
                for day in range(start_day, count_days):
                    if schedule[machine_idx][day] is None:
                        schedule[machine_idx][day] = int(product_idx)
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[machine_idx, 'product_qty'] -= 1
                        machines_df.at[machine_idx, 'day_remains'] -= 1

                machines_to_schedule = machines_to_schedule[machines_to_schedule['original_index'] != machine_idx]

            # Обновляем qty для продукта
            products_to_schedule.iloc[0, products_to_schedule.columns.get_loc('qty')] -= min(
                qty_needed, sum(1 for day in range(count_days) if schedule[machine_idx][day] == product_idx)
            )
            # Удаляем продукт, если qty <= 0
            products_to_schedule = products_to_schedule[products_to_schedule['qty'] > 0]
            # Пересортировка
            machines_to_schedule = machines_to_schedule.sort_values(
                by=['type', 'product_qty', 'day_remains'], ascending=[False, True, True]
            )
            products_to_schedule = products_to_schedule.sort_values(
                by=['machine_type', 'qty'], ascending=[False, False]
            )

    # Отдельное распределение для продуктов типа 1
    schedule_remaining_days(machine_type_filter=1)
    # Распределение для всех продуктов
    schedule_remaining_days()

    # --- Подведение итогов ---

    # proportions_input - массив qty продуктов с индексом больше нуля
    proportions_input = products_df[products_df['idx'] > 0]['qty'].values
    total_work_days = sum(
        1 for machine in schedule for day in machine if day not in [-2, 0]
    )

    # Коэффициент для штрафов
    kf_downtime_penalty = round(0.1 * sum(proportions_input) / len([d for d in schedule for day in d if day != -2]))
    if kf_downtime_penalty < 10:
        kf_downtime_penalty = 10

    # Подсчет отклонений пропорций
    proportion_objective_terms = []
    for product_idx in products_df[products_df['idx'] > 0]['idx']:
        planned_qty = sum(
            1 for machine in schedule for day in machine if day == product_idx
        )
        required_qty = products_df[products_df['idx'] == product_idx]['qty'].iloc[0]
        proportion = planned_qty / total_work_days if total_work_days > 0 else 0
        expected_proportion = required_qty / sum(proportions_input) if sum(proportions_input) > 0 else 0
        proportion_objective_terms.append(abs(round(proportion - expected_proportion)))

    # Подсчет переходов
    count_product_zero = sum(1 for machine in schedule for day in machine if day == 0)

    # Итоговый показатель
    objective_value = sum(proportion_objective_terms) + count_product_zero * kf_downtime_penalty
    deviation_proportion = sum(proportion_objective_terms)

    return schedule, objective_value, deviation_proportion, count_product_zero

if __name__ == "__main__":
    with open("test_in.json", encoding="utf8") as f:
        test_in = f.read()

    data = json.loads(test_in)
    machines = data["machines"]
    schedule, objective_value, deviation_proportion, count_product_zero = (
        create_schedule_init(data["machines"], data["products"], data["cleans"], data["count_days"], data["max_daily_prod_zero"])
    )

    # Вывод результатов
    print(f'Итоговый показатель предварительного плана: {objective_value}'
          f', в т.ч. переходов {count_product_zero}, отклонения пропорций {deviation_proportion}')
    print("\nРасписание (schedule[machine_idx][day_idx]):")
    for machine_idx in range(len(machines)):
        print(f"Машина {machine_idx}: {schedule[machine_idx]}")