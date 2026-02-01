import plotly.graph_objects as go
from datetime import timedelta, date
import colorsys
import traceback as tr
from typing import Iterable, Any

# TODO: переменную days нужно оптимизировать - list не нужен, достаточно количества дней
def schedule_to_html(machines: list, products: list, days: list, schedules: list, dt_begin: date, title_text: str = ""):

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
    shifts = ['У', 'В', 'Н']
    x_labels = []
    x_dates = []
    x_positions = []
    x_idx = 0
    for day in range(len(days) // 3):
        date_str = (dt_begin + timedelta(days=day)).strftime("%d.%m")
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

            if product_idx is not None and product_idx > 0:
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
    return fig.to_html(full_html=True)


def _safe_get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def aggregated_schedule_to_html(
    machines: list[dict] | list[Any],
    schedule: list[dict] | list[Any],
    products: list[dict],
    long_schedule: Iterable[Any],
    dt_begin: date,
    title_text: str = "",
) -> str:
    """HTML-отчёт для агрегированного плана (LONG_SIMPLE/LONG_TWOLEVEL).

    Вертикаль: дни (реальные календарные даты от dt_begin).
    Горизонталь: продукты, сгруппированные по цехам (div) и отсортированные по name.
    В ячейке – количество машин, занятых продуктом в этот день.
    Цветом выделяем только переходы (изменение количества машин по сравнению с предыдущим днём).
    """

    # Карта: product_idx -> (name, div)
    product_meta: dict[int, tuple[str, int | None]] = {}
    for p in products:
        try:
            idx = int(p.get("idx"))
        except Exception:
            continue
        name = p.get("name", f"p{idx}")
        div = p.get("div", None)
        product_meta[idx] = (name, div)

    # Собираем матрицу: counts[(day_idx, product_idx)] = machine_count
    counts: dict[tuple[int, int], int] = {}
    max_day = 0
    for rec in long_schedule:
        d = _safe_get(rec, "day_idx", 0) or 0
        p = _safe_get(rec, "product_idx", 0) or 0
        c = _safe_get(rec, "machine_count", 0) or 0
        try:
            d = int(d)
            p = int(p)
            c = int(c)
        except Exception:
            continue
        if p <= 0:
            continue
        counts[(d, p)] = counts.get((d, p), 0) + c
        if d > max_day:
            max_day = d

    days = list(range(max_day + 1)) if counts else []

    # Группируем продукты по div и сортируем по имени внутри группы.
    groups: dict[int | None, list[int]] = {}
    for idx, (name, div) in product_meta.items():
        g = div if div in (1, 2) else None
        groups.setdefault(g, []).append(idx)
    for g in groups:
        groups[g].sort(key=lambda i: product_meta[i][0])

    # Предварительно считаем ежедневные итоги:
    # - transitions_per_day[d]: число переходов по всем машинам (как в анализаторе),
    #   где переход в день 0 считается, если на машине был стартовый продукт >0 и
    #   продукт в день 0 отличается от стартового; для d>0 — если p(d-1)>0, p(d)>0
    #   и они различаются.
    # - machines_per_div_day[(div, d)]: суммарное количество машин по цеху/группе в день d.
    transitions_per_day: dict[int, int] = {d: 0 for d in days}
    machines_per_div_day: dict[tuple[int | None, int], int] = {}

    # Строим карту (m,d) -> product_idx из детального расписания.
    md: dict[tuple[int, int], int] = {}
    for rec in schedule:
        m = _safe_get(rec, "machine_idx", 0) or 0
        d = _safe_get(rec, "day_idx", 0) or 0
        p = _safe_get(rec, "product_idx", 0) or 0
        try:
            m = int(m); d = int(d); p = int(p)
        except Exception:
            continue
        md[(m, d)] = p

    # Стартовые продукты по машинам
    init_prod: dict[int, int] = {}
    for m_idx, mrec in enumerate(machines):
        p0 = _safe_get(mrec, "product_idx", None)
        if p0 is None and isinstance(mrec, (list, tuple)) and len(mrec) > 1:
            p0 = mrec[1]
        try:
            init_prod[m_idx] = int(p0 or 0)
        except Exception:
            init_prod[m_idx] = 0

    num_machines = len(machines)

    for d in days:
        day_transitions = 0
        # Переходы по машинам
        for m in range(num_machines):
            p_cur = md.get((m, d), 0) or 0
            if d == 0:
                p_init = init_prod.get(m, 0)
                if p_init > 0 and p_cur > 0 and p_cur != p_init:
                    day_transitions += 1
            else:
                p_prev = md.get((m, d - 1), 0) or 0
                if p_prev > 0 and p_cur > 0 and p_cur != p_prev:
                    day_transitions += 1
        transitions_per_day[d] = day_transitions

        # Суммарное количество машин по цехам/группам в этот день
        for idx, (name, div) in product_meta.items():
            g = div if div in (1, 2) else None
            c = counts.get((d, idx), 0)
            machines_per_div_day[(g, d)] = machines_per_div_day.get((g, d), 0) + c

    # CSS-стили для таблицы и подсветки переходов.
    styles = """
    <style>
    body { font-family: Arial, sans-serif; }
    table.plan { border-collapse: collapse; margin: 16px 0; }
    table.plan th, table.plan td { border: 1px solid #ccc; padding: 4px 6px; text-align: center; font-size: 12px; }
    table.plan th { background-color: #f0f0f0; }
    td.zero { background-color: #ffffff; color: #bbb; }
    td.nonzero { background-color: #ffffff; }
    td.transition { background-color: #ffe8a3; }
    .div-header { margin-top: 24px; font-weight: bold; }
    </style>
    """

    html_parts: list[str] = []
    html_parts.append("<html><head>")
    html_parts.append(styles)
    html_parts.append("</head><body>")
    html_parts.append(f"<h2>Агрегированное расписание: {title_text}</h2>")

    if not days or not groups:
        html_parts.append("<p>Нет данных для отображения.</p>")
        html_parts.append("</body></html>")
        return "".join(html_parts)

    # Единая таблица: день, переходы (все цеха), затем итоги и продукты по цехам.
    html_parts.append("<h3>Сводка по дням и цехам</h3>")
    html_parts.append("<table class='plan'>")

    # Определяем порядок цехов
    div_keys = sorted(groups.keys(), key=lambda x: (999 if x is None else x))

    # Заголовок: Дата | Переходов | Итого цех1 | продукты цех1... | Итого цех2 | продукты цех2...
    html_parts.append("<tr><th>Дата</th><th>Переходов</th>")
    for div in div_keys:
        if div not in (1, 2):
            continue
        html_parts.append(f"<th>Итого цех {div}</th>")
        for p_idx in groups.get(div, []):
            name, _ = product_meta[p_idx]
            html_parts.append(f"<th>{name}</th>")
    html_parts.append("</tr>")

    # Строки по дням
    for d in days:
        date_str = (dt_begin + timedelta(days=d)).strftime("%d.%m.%Y")
        html_parts.append(f"<tr><td>{date_str}</td>")
        html_parts.append(f"<td>{transitions_per_day.get(d, 0)}</td>")
        for div in div_keys:
            if div not in (1, 2):
                continue
            total_m = machines_per_div_day.get((div, d), 0)
            html_parts.append(f"<td>{total_m}</td>")
            for p_idx in groups.get(div, []):
                c = counts.get((d, p_idx), 0)
                prev_c = counts.get((d - 1, p_idx), 0) if d > 0 else 0
                is_transition = (d == 0 and c > 0) or (d > 0 and c != prev_c)
                if c == 0:
                    cls = "zero"
                    text = ""
                else:
                    cls = "transition" if is_transition else "nonzero"
                    text = str(c)
                html_parts.append(f"<td class='{cls}'>{text}</td>")
        html_parts.append("</tr>")

    html_parts.append("</table>")

    html_parts.append("</body></html>")
    return "".join(html_parts)

