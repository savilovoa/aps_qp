import pandas as pd
import json
import uuid

def create_schedule_init(machines, products, cleans, count_days, max_daily_prod_zero):
    machines_df = pd.DataFrame(machines)
    products_df = pd.DataFrame(products)
    num_machines = len(machines_df)
    schedule = [[None for _ in range(count_days)] for _ in range(num_machines)]

    # Предварительно заполняем дни очистки (cleans)
    for clean in cleans:
        machine_idx = clean['machine_idx']
        day_idx = clean['day_idx']
        if 0 <= machine_idx < num_machines and 0 <= day_idx < count_days:
            schedule[machine_idx][day_idx] = -2

    # Функция для проверки возможности установки перехода
    def can_place_zero(day, zeros_per_day, max_daily_prod_zero):
        return zeros_per_day.get(day, 0) < max_daily_prod_zero

    zeros_per_day = {day: 0 for day in range(count_days)}
    machines_df.reset_index(inplace=True)
    machines_df.rename(columns={'index': 'original_index'}, inplace=True)
    product_quantities = products_df['qty']
    machines_df['product_qty'] = machines_df['product_idx'].map(product_quantities)

    # --- Часть 1: Планирование начальных продуктов ---
    machines_df['day_remains'] = count_days - machines_df['idx'].map(
        lambda x: sum(1 for d in schedule[x] if d == -2)
    )

    for idx, machine in machines_df.iterrows():
        product_idx = machine['product_idx']
        qty = machine['product_qty']
        machine_idx = machine['idx']
        if qty >= count_days / 2:
            days_to_plan = min(int(qty), machines_df.at[idx, 'day_remains'])
            current_day = 0
            planned_days = 0
            while planned_days < days_to_plan and current_day < count_days:
                if schedule[machine_idx][current_day] is None:
                    schedule[machine_idx][current_day] = product_idx
                    planned_days += 1
                    machines_df.at[idx, 'day_remains'] -= 1
                    products_df.at[product_idx, 'qty'] -= 1
                current_day += 1

            # Планируем переход (2 дня product_idx = 0)
            if current_day + 1 < count_days and planned_days > 0:
                transition_days = 0
                while transition_days < 2 and current_day < count_days:
                    if schedule[machine_idx][current_day] is None and can_place_zero(current_day, zeros_per_day, max_daily_prod_zero):
                        schedule[machine_idx][current_day] = 0
                        zeros_per_day[current_day] = zeros_per_day.get(current_day, 0) + 1
                        machines_df.at[idx, 'day_remains'] -= 1
                        transition_days += 1
                    elif schedule[machine_idx][current_day] is None:
                        schedule[machine_idx][current_day] = product_idx
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[idx, 'day_remains'] -= 1
                    current_day += 1
                if current_day >= count_days - 2 and transition_days < 2:
                    while current_day < count_days and schedule[machine_idx][current_day] is None:
                        schedule[machine_idx][current_day] = product_idx
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[idx, 'day_remains'] -= 1
                        current_day += 1

    # --- Часть 2: Заполнение оставшихся дней ---
    def sort_machines_and_products(machines_df, products_df):
        machines_df_sorted = machines_df.sort_values(
            by=['type', 'product_qty', 'day_remains'], ascending=[False, True, True]
        ).copy()
        products_df_sorted = products_df.sort_values(
            by=['machine_type', 'qty'], ascending=[False, False]
        ).copy()
        return machines_df_sorted, products_df_sorted

    while machines_df['day_remains'].sum() > 0:
        machines_df_sorted, products_df_sorted = sort_machines_and_products(machines_df, products_df)
        
        # Отдельная ветка для продуктов типа 1
        for _, product in products_df_sorted[products_df_sorted['machine_type'] == 1].iterrows():
            product_idx = product['idx']
            qty = product['qty']
            if qty <= 0:
                continue
            # Проверяем машины с совпадающим начальным продуктом
            matching_machines = machines_df_sorted[
                (machines_df_sorted['product_idx'] == product_idx) & 
                (machines_df_sorted['day_remains'] > 0) & 
                (machines_df_sorted['type'] == 1)
            ]
            if not matching_machines.empty:
                machine = matching_machines.iloc[0]
                machine_idx = machine['idx']
                current_day = 0
                while current_day < count_days and schedule[machine_idx][current_day] is not None:
                    current_day += 1
                while current_day < count_days and products_df.at[product_idx, 'qty'] > 0 and machines_df.at[machine_idx, 'day_remains'] > 0:
                    if schedule[machine_idx][current_day] is None:
                        schedule[machine_idx][current_day] = product_idx
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[machine_idx, 'day_remains'] -= 1
                    current_day += 1
                continue

            # Если нет совпадений, берем первую подходящую машину
            for _, machine in machines_df_sorted[machines_df_sorted['type'] == 1].iterrows():
                machine_idx = machine['idx']
                if machines_df.at[machine_idx, 'day_remains'] == 0:
                    continue
                current_day = 0
                while current_day < count_days and schedule[machine_idx][current_day] is not None:
                    current_day += 1
                if current_day < count_days:
                    if machine['product_idx'] == product_idx:
                        while current_day < count_days and products_df.at[product_idx, 'qty'] > 0:
                            if schedule[machine_idx][current_day] is None:
                                schedule[machine_idx][current_day] = product_idx
                                products_df.at[product_idx, 'qty'] -= 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                            current_day += 1
                    else:
                        # Планируем начальный продукт машины
                        initial_product = machine['product_idx']
                        if products_df.at[initial_product, 'qty'] > 0:
                            while current_day < count_days and schedule[machine_idx][current_day] is None and products_df.at[initial_product, 'qty'] > 0:
                                schedule[machine_idx][current_day] = initial_product
                                products_df.at[initial_product, 'qty'] -= 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                                current_day += 1
                        # Планируем переход
                        transition_days = 0
                        while transition_days < 2 and current_day < count_days:
                            if schedule[machine_idx][current_day] is None and can_place_zero(current_day, zeros_per_day, max_daily_prod_zero):
                                schedule[machine_idx][current_day] = 0
                                zeros_per_day[current_day] += 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                                transition_days += 1
                            elif schedule[machine_idx][current_day] is None:
                                schedule[machine_idx][current_day] = initial_product
                                products_df.at[initial_product, 'qty'] -= 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                            current_day += 1
                        # Планируем новый продукт
                        while current_day < count_days and products_df.at[product_idx, 'qty'] > 0:
                            if schedule[machine_idx][current_day] is None:
                                schedule[machine_idx][current_day] = product_idx
                                products_df.at[product_idx, 'qty'] -= 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                            current_day += 1
                    break

        # Продукты типа 0
        for _, product in products_df_sorted[products_df_sorted['machine_type'] == 0].iterrows():
            product_idx = product['idx']
            qty = product['qty']
            if qty <= 0:
                continue
            matching_machines = machines_df_sorted[
                (machines_df_sorted['product_idx'] == product_idx) & 
                (machines_df_sorted['day_remains'] > 0)
            ]
            if not matching_machines.empty:
                machine = matching_machines.iloc[0]
                machine_idx = machine['idx']
                current_day = 0
                while current_day < count_days and schedule[machine_idx][current_day] is not None:
                    current_day += 1
                while current_day < count_days and products_df.at[product_idx, 'qty'] > 0:
                    if schedule[machine_idx][current_day] is None:
                        schedule[machine_idx][current_day] = product_idx
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[machine_idx, 'day_remains'] -= 1
                    current_day += 1
                continue

            for _, machine in machines_df_sorted.iterrows():
                machine_idx = machine['idx']
                if machines_df.at[machine_idx, 'day_remains'] == 0:
                    continue
                current_day = 0
                while current_day < count_days and schedule[machine_idx][current_day] is not None:
                    current_day += 1
                if current_day < count_days:
                    if machine['product_idx'] == product_idx:
                        while current_day < count_days and products_df.at[product_idx, 'qty'] > 0:
                            if schedule[machine_idx][current_day] is None:
                                schedule[machine_idx][current_day] = product_idx
                                products_df.at[product_idx, 'qty'] -= 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                            current_day += 1
                    else:
                        initial_product = machine['product_idx']
                        if products_df.at[initial_product, 'qty'] > 0:
                            while current_day < count_days and schedule[machine_idx][current_day] is None and products_df.at[initial_product, 'qty'] > 0:
                                schedule[machine_idx][current_day] = initial_product
                                products_df.at[initial_product, 'qty'] -= 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                                current_day += 1
                        transition_days = 0
                        while transition_days < 2 and current_day < count_days:
                            if schedule[machine_idx][current_day] is None and can_place_zero(current_day, zeros_per_day, max_daily_prod_zero):
                                schedule[machine_idx][current_day] = 0
                                zeros_per_day[current_day] += 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                                transition_days += 1
                            elif schedule[machine_idx][current_day] is None:
                                schedule[machine_idx][current_day] = initial_product
                                products_df.at[initial_product, 'qty'] -= 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                            current_day += 1
                        while current_day < count_days and products_df.at[product_idx, 'qty'] > 0:
                            if schedule[machine_idx][current_day] is None:
                                schedule[machine_idx][current_day] = product_idx
                                products_df.at[product_idx, 'qty'] -= 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                            current_day += 1
                    break

    # --- Часть 3: Заполнение оставшихся дней ---
    machines_df_sorted = machines_df.sort_values(
        by=['type', 'day_remains'], ascending=[False, False]
    ).copy()
    products_df_sorted = products_df.sort_values(
        by=['qty'], ascending=[False]
    ).copy()

    for _, machine in machines_df_sorted.iterrows():
        machine_idx = machine['idx']
        if machines_df.at[machine_idx, 'day_remains'] == 0:
            continue
        current_day = 0
        while current_day < count_days:
            if schedule[machine_idx][current_day] is None:
                initial_product = machine['product_idx']
                if products_df.at[initial_product, 'qty'] > 0:
                    schedule[machine_idx][current_day] = initial_product
                    products_df.at[initial_product, 'qty'] -= 1
                    machines_df.at[machine_idx, 'day_remains'] -= 1
                    current_day += 1
                else:
                    # Планируем переход
                    transition_days = 0
                    while transition_days < 2 and current_day < count_days:
                        if schedule[machine_idx][current_day] is None and can_place_zero(current_day, zeros_per_day, max_daily_prod_zero):
                            schedule[machine_idx][current_day] = 0
                            zeros_per_day[current_day] += 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                            transition_days += 1
                        elif schedule[machine_idx][current_day] is None:
                            schedule[machine_idx][current_day] = initial_product
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                        current_day += 1
                    # Планируем продукт из отсортированного списка
                    for _, product in products_df_sorted.iterrows():
                        product_idx = product['idx']
                        if products_df.at[product_idx, 'qty'] > 0 and (product['machine_type'] == 0 or machine['type'] == 1):
                            while current_day < count_days and products_df.at[product_idx, 'qty'] > 0:
                                if schedule[machine_idx][current_day] is None:
                                    schedule[machine_idx][current_day] = product_idx
                                    products_df.at[product_idx, 'qty'] -= 1
                                    machines_df.at[machine_idx, 'day_remains'] -= 1
                                current_day += 1
                            break
            else:
                current_day += 1

    # --- Подведение итогов ---
    work_days = sum(machines_df['day_remains'].map(lambda x: count_days - x))
    proportions_input = products_df[products_df['idx'] > 0]['qty'].values
    kf_downtime_penalty = round(0.1 * sum(proportions_input) / len([d for m in schedule for d in m if d != -2]))
    if kf_downtime_penalty < 10:
        kf_downtime_penalty = 10

    # Подсчет отклонений
    planned_counts = {i: 0 for i in range(len(products_df))}
    for machine_schedule in schedule:
        for day in machine_schedule:
            if day is not None and day > 0:
                planned_counts[day] += 1

    proportion_objective_terms = []
    for idx, row in products_df.iterrows():
        if idx == 0:
            continue
        required = row['qty']
        planned = planned_counts.get(idx, 0)
        proportion_objective_terms.append(abs(required - planned))

    count_product_zero = sum(zeros_per_day.values)
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

    print(f'Итоговый показатель предварительного плана: {objective_value}'
          f', в т.ч. переходов {count_product_zero}, отклонения пропорций {deviation_proportion}')
    print("\nРасписание (schedule[machine_idx][day_idx]):")
    for machine_idx in range(len(machines)):
        print(f"Машина {machine_idx}: {schedule[machine_idx]}")