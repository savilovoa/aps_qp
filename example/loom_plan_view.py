import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Предположим, у нас есть такие данные
machines = ['Станок 1', 'Станок 2', 'Станок 3']  # Массив оборудования
products = ['Продукт A', 'Продукт B', 'Продукт C']  # Массив продуктов
days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт']  # Массив дней

# Пример расписания
schedule = [
    {"machine_id": 0, "day_id": 0, "product_id": 1},
    {"machine_id": 0, "day_id": 1, "product_id": 1},
    {"machine_id": 0, "day_id": 2, "product_id": 2},
    {"machine_id": 1, "day_id": 0, "product_id": 0},
    {"machine_id": 1, "day_id": 1, "product_id": 2},
    {"machine_id": 2, "day_id": 2, "product_id": 1}
]

# Создаем пустую таблицу
table = np.full((len(machines), len(days)), '', dtype=object)
colors = np.full((len(machines), len(days)), 'white', dtype=object)

# Заполняем таблицу данными
for entry in schedule:
    machine_id = entry['machine_id']
    day_id = entry['day_id']
    product_id = entry['product_id']

    # Проверяем предыдущий день
    if day_id > 0 and table[machine_id][day_id - 1] == products[product_id]:
        table[machine_id][day_id] = ''  # Не выводим название, если тот же продукт
    else:
        table[machine_id][day_id] = products[product_id]

    # Назначаем цвет фона
    colors[machine_id][day_id] = f'rgba({255 - product_id * 85}, {255 - product_id * 85}, {255 - product_id * 85}, 0.5)'

# Создаем DataFrame для удобства работы
df = pd.DataFrame(table, index=machines, columns=days)

# Создаем таблицу в Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=['Оборудование'] + list(days),
                fill_color='lightgrey',
                align='center'),
    cells=dict(values=[[machine] + list(df.loc[machine]) for machine in machines],
               fill_color=[['white'] + list(colors[i]) for i in range(len(machines))],
               align='center'))
])

# Настраиваем внешний вид
fig.update_layout(
    width=800,
    height=400,
    title_text="Производственное планирование",
    title_x=0.5
)

fig.show()
