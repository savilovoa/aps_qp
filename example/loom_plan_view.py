import plotly.graph_objects as go
from datetime import datetime, timedelta
import colorsys


def view_schedule(machines: list, products: list, days: list, schedules: list, title_text: str = ""):
    dt_begin_str = "01.07.2025"

    # --- 1. Подготовка сетки расписания ---
    # Создаем 2D-массив, где grid[machine_id][day_id] будет хранить product_id
    # Инициализируем None для пустых ячеек
    schedule_grid = [[None for _ in days] for _ in machines]

    for item in schedules:
        schedule_grid[item['machine_idx']][item['day_idx']] = item['product_idx']

    # --- 2. Генерация уникальных цветов для продуктов ---
    product_colors = {}
    num_products = len(products)
    # Используем HLS для генерации perceptually distinct цветов
    for i, product_name in enumerate(products):
        hue = i / num_products
        # Немного рандомизируем насыщенность и яркость для большего разнообразия
        saturation = 0.7  # random.uniform(0.5, 0.9)
        lightness = 0.6  # random.uniform(0.5, 0.8)
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]
        product_colors[i] = f'rgb({r},{g},{b})'

    # Дополнительный цвет для пустых ячеек
    empty_cell_color = 'rgb(240, 240, 240)'  # Светло-серый

    # Подготовка данных для таблицы
    dt_begin = datetime.strptime(dt_begin_str, "%d.%m.%Y")
    shifts = ['У', 'В', 'Н']
    x_labels = []
    x_dates = []
    x_positions = []
    x_idx = 0
    for day in days:
        date_str = (dt_begin + timedelta(days=day // 3)).strftime("%d.%m")
        for shift in shifts:
            x_labels.append(f'{date_str}<br>{shift}')
            x_positions.append(x_idx)
            x_idx += 1

    # --- 3. Создание фигур и аннотаций для Plotly ---
    shapes = []
    annotations = []

    # Размеры ячейки для центрирования
    cell_width = 1
    cell_height = 1

    for machine_idx, machine_row in enumerate(schedule_grid):
        for day_idx, product_idx in enumerate(machine_row):
            # Координаты центра ячейки
            # X-координата соответствует day_idx
            # Y-координата должна быть инвертирована, чтобы Станок 1 был сверху
            x_center = day_idx
            y_center = len(machines) - 1 - machine_idx  # Инвертируем Y для отображения сверху вниз

            fill_color = empty_cell_color
            text_to_display = ""

            if product_idx is not None:
                fill_color = product_colors[product_idx]
                # Проверяем, был ли тот же продукт на этой машине в предыдущий день
                if day_idx > 0 and schedule_grid[machine_idx][day_idx - 1] == product_idx:
                    text_to_display = ""  # Скрываем текст, если продукт продолжается
                else:
                    text_to_display = products[product_idx]  # Показываем название продукта

                # Добавляем прямоугольник (фон ячейки)
            shapes.append(
                dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=x_center - cell_width / 2, y0=y_center - cell_height / 2,
                    x1=x_center + cell_width / 2, y1=y_center + cell_height / 2,
                    fillcolor=fill_color,
                    line=dict(color="black", width=1),  # Граница ячейки
                    layer="below"  # Рисуем прямоугольник под текстом
                )
            )

            # Добавляем аннотацию (текст продукта)
            if text_to_display:
                annotations.append(
                    dict(
                        x= x_center + 0.2 - cell_width / 2, y=y_center,
                        text=text_to_display,
                        showarrow=False,
                        font=dict(color="black", size=12),
                        xanchor='left', yanchor='middle'
                    )
                )


    # --- 4. Настройка макета Plotly ---
    fig = go.Figure()

    # Добавляем пустой scatter trace, чтобы оси отображались корректно
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', hoverinfo='none'))

    fig.update_layout(
        title=f"Расписание производства: {title_text}",
        xaxis=dict(
            title='Дни недели',
            tickvals=[i for i in range(len(days))],
            ticktext=x_labels,
            side='top',  # Дни сверху
            range=[-0.5, len(days) - 0.5],  # Центрируем ячейки вокруг тиков
            showgrid=False,  # Скрываем сетку
            zeroline=False,
            fixedrange=True  # Запрещаем зум по X
        ),
        yaxis=dict(
            title='Оборудование',
            tickvals=[len(machines) - 1 - i for i in range(len(machines))],  # Инвертируем tickvals для машин
            ticktext=machines,
            range=[-0.5, len(machines) - 0.5],  # Центрируем ячейки вокруг тиков
            showgrid=False,  # Скрываем сетку
            zeroline=False,
            autorange='reversed',  # Также можно использовать для инверсии Y-оси
            fixedrange=True  # Запрещаем зум по Y
        ),
        shapes=shapes,  # Добавляем все прямоугольники
        annotations=annotations,  # Добавляем все текстовые аннотации
        plot_bgcolor='white',  # Цвет фона графика
        hovermode=False,  # Отключаем всплывающие подсказки по умолчанию
        height=400 + len(machines) * 30,  # Динамическая высота
        width=600 + len(days) * 50,  # Динамическая ширина
        margin=dict(l=100, r=50, t=100, b=50)  # Отступы
    )
    f_name = "example/res.html"
    with open(f_name, "w", encoding="utf8") as f:
        f.write(fig.to_html(full_html=True))
    #fig.show()

if __name__ == "__main__":

    # 1. Исходные данные
    machines = ['ТС Тойота №1', 'ТС Тойота №2', 'ТС Тойота №3']
    products = ['', 'ст87017t3 ', 'ст87416t1 ', 'ст2022УИСt4 ']
    days = list(range(21))  # Теперь это индексы смен
    schedules = [{'machine_idx': 0, 'day_idx': 0, 'product_idx': 1}, {'machine_idx': 0, 'day_idx': 1, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 2, 'product_idx': 1}, {'machine_idx': 0, 'day_idx': 3, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 4, 'product_idx': 1}, {'machine_idx': 0, 'day_idx': 5, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 6, 'product_idx': 1}, {'machine_idx': 0, 'day_idx': 7, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 8, 'product_idx': 1}, {'machine_idx': 0, 'day_idx': 9, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 10, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 11, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 12, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 13, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 14, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 15, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 16, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 17, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 18, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 19, 'product_idx': 1},
                 {'machine_idx': 0, 'day_idx': 20, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 0, 'product_idx': 1}, {'machine_idx': 1, 'day_idx': 1, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 2, 'product_idx': 1}, {'machine_idx': 1, 'day_idx': 3, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 4, 'product_idx': 1}, {'machine_idx': 1, 'day_idx': 5, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 6, 'product_idx': 1}, {'machine_idx': 1, 'day_idx': 7, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 8, 'product_idx': 1}, {'machine_idx': 1, 'day_idx': 9, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 10, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 11, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 12, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 13, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 14, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 15, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 16, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 17, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 18, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 19, 'product_idx': 1},
                 {'machine_idx': 1, 'day_idx': 20, 'product_idx': 1},
                 {'machine_idx': 2, 'day_idx': 0, 'product_idx': 3}, {'machine_idx': 2, 'day_idx': 1, 'product_idx': 3},
                 {'machine_idx': 2, 'day_idx': 2, 'product_idx': 3}, {'machine_idx': 2, 'day_idx': 3, 'product_idx': 3},
                 {'machine_idx': 2, 'day_idx': 4, 'product_idx': 0}, {'machine_idx': 2, 'day_idx': 5, 'product_idx': 0},
                 {'machine_idx': 2, 'day_idx': 6, 'product_idx': 2}, {'machine_idx': 2, 'day_idx': 7, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 8, 'product_idx': 2}, {'machine_idx': 2, 'day_idx': 9, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 10, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 11, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 12, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 13, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 14, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 15, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 16, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 17, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 18, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 19, 'product_idx': 2},
                 {'machine_idx': 2, 'day_idx': 20, 'product_idx': 2}]
    view_schedule(machines, products, days, schedules)