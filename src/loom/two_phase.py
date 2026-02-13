import logging
import pandas as pd
from ortools.sat.python import cp_model
from src.config import settings
from .schedule_loom import create_model_simple, solver_result, ProductsDFToArray, MachinesDFToArray, CleansDFToArray, create_simple_greedy_hint

logger = logging.getLogger(settings.PROJECT_NAME)

def solve_phase1_allocation(data_full: dict, machines_full_df: pd.DataFrame, products_full_df: pd.DataFrame, clean_df: pd.DataFrame, count_days: int) -> dict[int, set[int]]:
    """
    Фаза 1: Решаем задачу распределения отдельно для каждого цеха (div).
    Цель: Определить множество допустимых продуктов для каждой машины (allowed_products[m]).
    
    Возвращает: {machine_internal_idx: {product_internal_idx, ...}}
    """
    # 1. Определяем список цехов
    divs = machines_full_df["div"].unique().tolist()
    # Обрабатываем NaN и 0 как div=1 по умолчанию, если они есть
    divs = [int(d) if pd.notna(d) and d > 0 else 1 for d in divs]
    divs = sorted(list(set(divs)))
    
    logger.info(f"Two-Phase Phase 1: Solving for divisions {divs}")
    
    allowed_products_map: dict[str, set[str]] = {} # machine_id -> set of product_ids
    
    # Словари для маппинга ID
    machine_id_to_internal_full = {row["id"]: i for i, row in machines_full_df.iterrows()}
    product_id_to_internal_full = {row["id"]: i for i, row in products_full_df.iterrows()}
    
    for div in divs:
        logger.info(f"--- Phase 1: Solving DIV {div} ---")
        
        # 2. Фильтруем данные для текущего цеха
        # Машины: только из текущего div
        machines_div_mask = machines_full_df["div"] == div
        machines_div = machines_full_df[machines_div_mask].copy().reset_index(drop=True)
        
        if machines_div.empty:
            continue
            
        # Продукты: те, которые привязаны к этому div, ИЛИ flex (div=0/NaN)
        # В Phase 1 мы можем упростить: берем продукты, у которых div == current_div OR div == 0
        products_div_mask = (products_full_df["div"] == div) | (products_full_df["div"].fillna(0) == 0)
        products_div = products_full_df[products_div_mask].copy().reset_index(drop=True)
        
        # Чистки: только для машин из текущего div
        # Нам нужно знать ID машин, чтобы отфильтровать cleans
        machine_ids_div = set(machines_div["id"])
        
        # В data["cleans"] индексы машин глобальные (из исходного JSON).
        # Нам нужно отобразить их. Но проще работать с CleanDF
        # CleanDF уже имеет machine_idx, который ссылается на ГЛОБАЛЬНЫЙ список.
        # Нам нужно пересчитать индексы для локального списка.
        
        # Строим маппинг global_m_idx -> local_m_idx
        # Для этого нам нужен исходный глобальный индекс. 
        # В machines_full_df индекс строки = global_idx.
        # Добавим колонку global_idx в machines_div
        machines_div_indices = machines_full_df.index[machines_div_mask].tolist()
        global_to_local_m = {g_idx: l_idx for l_idx, g_idx in enumerate(machines_div_indices)}
        
        # --- RENUMBERING START ---
        # 1. Перенумерация машин (idx)
        # Сохраняем старый idx для отладки/связи (если нужно), но для модели делаем 0..N-1
        machines_div = machines_div.copy() # explicit copy
        # Карту старых idx (из JSON) в новые локальные
        old_mach_idx_to_new = {}
        for local_i, (_, row) in enumerate(machines_div.iterrows()):
            old_idx = row["idx"]
            old_mach_idx_to_new[old_idx] = local_i
        
        # Присваиваем новые последовательные индексы
        machines_div["idx"] = range(len(machines_div))

        # 2. Перенумерация продуктов (idx) и обновление ссылок в машинах
        products_div = products_div.copy()
        
        # Карта: old_prod_idx (из JSON) -> new_prod_idx (0..K-1)
        # Но у нас проблема: machines_full_df ссылается на product_idx из products_full_df.
        # Нам нужно знать, какой idx был у продукта в products_full_df.
        # products_full_df index = глобальный индекс.
        
        # Получаем индексы строк из products_full_df, которые попали в выборку
        products_div_indices = products_full_df.index[products_div_mask].tolist()
        global_prod_idx_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(products_div_indices)}
        
        # Но в JSON поле "idx" у продуктов может отличаться от индекса строки (хотя обычно совпадает).
        # Давайте строить карту на основе поля "idx" исходного DF.
        old_prod_idx_to_new = {}
        for local_i, (_, row) in enumerate(products_div.iterrows()):
            old_idx = row["idx"]
            old_prod_idx_to_new[old_idx] = local_i
            
        # Присваиваем новые индексы продуктам
        products_div["idx"] = range(len(products_div))
        
        # Важно: гарантируем lday >= 1 для всех продуктов.
        # Продукты с lday=0 вызывают Access Violation в OR-Tools.
        # Для продуктов с qty=0 (технические стартовые коды) lday=0 допустим
        # в исходных данных, но для модели нужен валидный lday.
        products_div.loc[products_div["lday"] <= 0, "lday"] = 10  # default как в schedule_loom
        
        # 3. Обновляем product_idx в машинах
        # machines_div["product_idx"] ссылается на старые индексы.
        # Нужно перевести их в новые. Если продукта нет в локальном списке (например, flex из другого цеха,
        # который мы не взяли - хотя мы берем всех flex), мапим в 0?
        # В Phase 1 мы берем: products_div = (div == current) OR (div == 0).
        # Машина может стартовать с продукта, который имеет div != current и div != 0?
        # Теоретически нет (машина div=1 не должна делать продукт div=2).
        # Но если такое есть, мапим в 0.
        
        def map_prod_idx_for_machine(old_p_idx):
            if old_p_idx in old_prod_idx_to_new:
                return old_prod_idx_to_new[old_p_idx]
            # Если не нашли по idx, пробуем по id (надежнее)
            # Но у нас нет id под рукой в map.
            # Давайте надеяться на целостность данных. Если продукта нет в выборке - ставим 0.
            return 0 

        machines_div["product_idx"] = machines_div["product_idx"].apply(map_prod_idx_for_machine).astype(int)

        # 4. Обновляем cleans (machine_idx)
        # clean_df["machine_idx"] - это старый idx машины.
        cleans_div_list = []
        for _, row in clean_df.iterrows():
            old_m = int(row["machine_idx"])
            if old_m in old_mach_idx_to_new:
                cleans_div_list.append((old_mach_idx_to_new[old_m], int(row["day_idx"])))
        # --- RENUMBERING END ---
        
        # Преобразуем в структуры для create_model_simple
        # ВАЖНО: product_idx уже обновлен выше.
        # Старый блок с pid_to_local_idx больше не нужен, так как мы сделали renumbering явно.

        machines_arr = MachinesDFToArray(machines_div)
        products_arr = ProductsDFToArray(products_div)
        
        # Параметры
        max_daily_prod_zero = int(data_full.get("max_daily_prod_zero", 3))
        # Для отдельного цеха можно пропорционально уменьшить лимит переходов? 
        # Или оставить глобальный, так как в Phase 1 мы просто ищем допустимость.
        # Оставим жесткий лимит, чтобы отсечь нереализуемые варианты.
        # Но если цехов 2, а лимит 3, то каждому достанется ~1.5. 
        # Безопаснее дать полный лимит каждому, чтобы не пережать.
        # В Phase 2 глобальный лимит все равно все поправит.
        
        # Строим модель
        # Важно: product_divs/machine_divs нужны для create_model_simple
        prod_divs_local = products_div["div"].fillna(0).astype(int).tolist()
        mach_divs_local = machines_div["div"].fillna(1).astype(int).tolist()
        
        # Hints
        greedy_hint = None
        if settings.USE_GREEDY_HINT:
             try:
                greedy_hint = create_simple_greedy_hint(
                    machines_arr, products_arr, count_days, prod_divs_local, mach_divs_local
                )
             except Exception:
                 pass
        
        # --- JSON DUMP START ---
        # Сохраняем подзадачу в JSON для отладки
        try:
            import json
            import os
            # Восстанавливаем структуру JSON из DataFrame
            # machines_div и products_div - это уже DF
            # Нам нужно превратить их в список словарей
            # Replace NaN with None to ensure valid JSON (standard JSON does not support NaN)
            machines_dump = machines_div.where(pd.notnull(machines_div), None).to_dict(orient="records")
            products_dump = products_div.where(pd.notnull(products_div), None).to_dict(orient="records")
            
            # cleans_div_list - это список кортежей (m_local, d).
            # Нам нужно превратить в список словарей {"machine_idx": ..., "day_idx": ...}
            cleans_dump = []
            for (m_idx, d_idx) in cleans_div_list:
                cleans_dump.append({"machine_idx": m_idx, "day_idx": d_idx})
                
            dump_data = {
                "machines": machines_dump,
                "products": products_dump,
                "remains": [], # dummy
                "cleans": cleans_dump,
                "max_daily_prod_zero": max_daily_prod_zero,
                "count_days": count_days,
                "dt_begin": data_full.get("dt_begin", "2026-01-01T00:00:00"),
                "apply_qty_minus": data_full.get("apply_qty_minus"),
                "apply_index_up": data_full.get("apply_index_up")
            }
            
            dump_filename = f"debug_phase1_div_{div}.json"
            dump_path = os.path.join(settings.BASE_DIR, "log", dump_filename)
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(dump_data, f, indent=4, ensure_ascii=False, default=str)
            logger.info(f"Phase 1 DIV {div}: Dumped debug JSON to {dump_path}")
        except Exception as e:
            logger.error(f"Phase 1 JSON dump failed: {e}")
        # --- JSON DUMP END ---

        # --- OUT-OF-PROCESS EXECUTION START ---
        # Запускаем внешний процесс python run.py для решения подзадачи
        import subprocess
        import os
        import json
        
        result_filename = f"result_phase1_div_{div}.json"
        result_path = os.path.join(settings.BASE_DIR, "log", result_filename)
        
        # Удаляем старый результат если есть
        if os.path.exists(result_path):
            try:
                os.remove(result_path)
            except Exception:
                pass
                
        # Формируем окружение для подпроцесса
        env = os.environ.copy()
        env["TEST_INPUT_FILE"] = dump_path
        env["HORIZON_MODE"] = "LONG_SIMPLE"
        env["CALC_TEST_DATA"] = "true"
        env["SAVE_RESULT_JSON_PATH"] = result_path
        # Важно: отключаем Two-Phase внутри подпроцесса, чтобы не уйти в рекурсию!
        
        # Запускаем
        logger.info(f"Phase 1 DIV {div}: Launching subprocess run.py...")
        try:
            # Используем тот же python интерпретатор
            import sys
            cmd = [sys.executable, "run.py"]
            # Запускаем и ждем (таймаут контролируется внутри run.py через LOOM_MAX_TIME)
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if proc.returncode != 0:
                logger.error(f"Phase 1 DIV {div} subprocess failed with code {proc.returncode}.\nStderr: {proc.stderr}")
                status_ok = False
            else:
                logger.info(f"Phase 1 DIV {div} subprocess finished successfully.")
                status_ok = True
        except Exception as e:
            logger.error(f"Phase 1 DIV {div} subprocess execution error: {e}")
            status_ok = False
            
        # Читаем результат
        if status_ok and os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    schedule_data = json.load(f)
                
                # Обрабатываем результат: какие продукты были на каких машинах
                # schedule_data - список словарей: [{"machine_idx": ..., "day_idx": ..., "product_idx": ...}, ...]
                # Индексы машин и продуктов здесь ЛОКАЛЬНЫЕ (0..N-1 для текущего div).
                # Нам нужно отобразить их обратно в product_id / machine_id.
                
                # Для этого нам нужны обратные карты, которые мы строили при дампе.
                # Но мы их не сохранили. Восстановим логику:
                # machines_div и products_div - это те DF, которые ушли в JSON.
                # У них индексы строк 0..N совпадают с machine_idx/product_idx в результате.
                
                # Итерируемся по расписанию
                for item in schedule_data:
                    m_local = item.get("machine_idx")
                    p_local = item.get("product_idx")
                    
                    if m_local is None or p_local is None:
                        continue
                    
                    # Пропускаем служебные значения (-2, 0 и т.д. если они есть)
                    if p_local <= 0:
                        continue
                        
                    # Находим ID машины
                    if 0 <= m_local < len(machines_div):
                        m_id = machines_div.iloc[m_local]["id"]
                        if m_id not in allowed_products_map:
                            allowed_products_map[m_id] = set()
                        
                        # Находим ID продукта
                        if 0 <= p_local < len(products_div):
                            p_id = products_div.iloc[p_local]["id"]
                            allowed_products_map[m_id].add(p_id)
                
                # Также добавляем начальные продукты (они обязательны)
                for m_local in range(len(machines_div)):
                    m_id = machines_div.iloc[m_local]["id"]
                    init_p_idx_local = int(machines_div.iloc[m_local]["product_idx"]) # Это уже локальный индекс
                    if init_p_idx_local > 0:
                         if m_id not in allowed_products_map:
                             allowed_products_map[m_id] = set()
                         if 0 <= init_p_idx_local < len(products_div):
                             p_id = products_div.iloc[init_p_idx_local]["id"]
                             allowed_products_map[m_id].add(p_id)

            except Exception as e:
                logger.error(f"Phase 1 DIV {div}: Failed to read result JSON: {e}")
        else:
            logger.warning(f"Phase 1 DIV {div}: Result file not found or process failed. Fallback: allow all valid products.")
            # Fallback (то же самое что и было)
            for m_local in range(len(machines_arr)):
                m_id = machines_div.iloc[m_local]["id"]
                m_type = machines_arr[m_local][3]
                if m_id not in allowed_products_map:
                    allowed_products_map[m_id] = set()
                for p_local in range(len(products_arr)):
                    p_type = products_arr[p_local][3]
                    p_div = prod_divs_local[p_local]
                    # Совместимость по типу и цеху
                    if (p_type == 0 or p_type == m_type) and (p_div == 0 or p_div == div):
                         p_id = products_div.iloc[p_local]["id"]
                         allowed_products_map[m_id].add(p_id)
        # --- OUT-OF-PROCESS EXECUTION END ---
        
        # Старый код in-process решения закомментирован/удален
        """
        (model, jobs, product_counts, _, _, _, _, _, _, _, _, _, _) = create_model_simple(
            ...
        )
        ...
        """

    # 3. Конвертируем allowed_products_map (по ID) в allowed_products_internal (по индексам глобальной модели)
    allowed_products_internal: dict[int, set[int]] = {}
    
    for m_id, p_id_set in allowed_products_map.items():
        if m_id not in machine_id_to_internal_full:
            continue
        m_idx = machine_id_to_internal_full[m_id]
        allowed_products_internal[m_idx] = set()
        
        for p_id in p_id_set:
            if p_id in product_id_to_internal_full:
                p_idx = product_id_to_internal_full[p_id]
                allowed_products_internal[m_idx].add(p_idx)
                
    return allowed_products_internal
