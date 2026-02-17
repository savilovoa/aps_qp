import logging
import os
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from ortools.sat.python import cp_model
from src.config import settings
from .schedule_loom import create_model_simple, solver_result, ProductsDFToArray, MachinesDFToArray, CleansDFToArray, create_simple_greedy_hint

logger = logging.getLogger(settings.PROJECT_NAME)


def _prepare_div_data(
    div: int,
    data_full: dict,
    machines_full_df: pd.DataFrame,
    products_full_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    count_days: int,
) -> dict | None:
    """
    Подготавливает данные для одного div: фильтрует, перенумеровывает, сохраняет JSON.
    Возвращает словарь с информацией для запуска и обработки результата, или None если div пустой.
    """
    # Фильтруем машины для текущего div
    machines_div_mask = machines_full_df["div"] == div
    machines_div = machines_full_df[machines_div_mask].copy().reset_index(drop=True)
    
    if machines_div.empty:
        return None
        
    # Продукты: привязанные к этому div ИЛИ flex (div=0/NaN)
    products_div_mask = (products_full_df["div"] == div) | (products_full_df["div"].fillna(0) == 0)
    products_div = products_full_df[products_div_mask].copy().reset_index(drop=True)
    
    # --- RENUMBERING ---
    # Карта старых idx машин в новые локальные
    old_mach_idx_to_new = {}
    for local_i, (_, row) in enumerate(machines_div.iterrows()):
        old_idx = row["idx"]
        old_mach_idx_to_new[old_idx] = local_i
    
    machines_div["idx"] = range(len(machines_div))

    # Карта старых idx продуктов в новые локальные
    old_prod_idx_to_new = {}
    for local_i, (_, row) in enumerate(products_div.iterrows()):
        old_idx = row["idx"]
        old_prod_idx_to_new[old_idx] = local_i
        
    products_div["idx"] = range(len(products_div))
    
    # lday >= 1 для всех продуктов
    products_div.loc[products_div["lday"] <= 0, "lday"] = 10

    # Обновляем product_idx в машинах
    def map_prod_idx(old_p_idx):
        return old_prod_idx_to_new.get(old_p_idx, 0)

    machines_div["product_idx"] = machines_div["product_idx"].apply(map_prod_idx).astype(int)

    # Обновляем cleans
    cleans_div_list = []
    for _, row in clean_df.iterrows():
        old_m = int(row["machine_idx"])
        if old_m in old_mach_idx_to_new:
            cleans_div_list.append((old_mach_idx_to_new[old_m], int(row["day_idx"])))

    # --- JSON DUMP ---
    max_daily_prod_zero = int(data_full.get("max_daily_prod_zero", 3))
    
    machines_dump = machines_div.where(pd.notnull(machines_div), None).to_dict(orient="records")
    products_dump = products_div.where(pd.notnull(products_div), None).to_dict(orient="records")
    cleans_dump = [{"machine_idx": m_idx, "day_idx": d_idx} for (m_idx, d_idx) in cleans_div_list]
    
    dump_data = {
        "machines": machines_dump,
        "products": products_dump,
        "remains": [],
        "cleans": cleans_dump,
        "max_daily_prod_zero": max_daily_prod_zero,
        "count_days": count_days,
        "dt_begin": data_full.get("dt_begin", "2026-01-01T00:00:00"),
        "apply_qty_minus": data_full.get("apply_qty_minus"),
        "apply_index_up": data_full.get("apply_index_up")
    }
    
    dump_filename = f"debug_phase1_div_{div}.json"
    dump_path = os.path.join(settings.BASE_DIR, "log", dump_filename)
    
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=4, ensure_ascii=False, default=str)
        logger.info(f"Phase 1 DIV {div}: Dumped debug JSON to {dump_path}")
    except Exception as e:
        logger.error(f"Phase 1 DIV {div} JSON dump failed: {e}")
        return None

    result_filename = f"result_phase1_div_{div}.json"
    result_path = os.path.join(settings.BASE_DIR, "log", result_filename)
    
    # Удаляем старый результат
    if os.path.exists(result_path):
        try:
            os.remove(result_path)
        except Exception:
            pass

    return {
        "div": div,
        "dump_path": dump_path,
        "result_path": result_path,
        "machines_div": machines_div,
        "products_div": products_div,
    }


def _run_subprocess(div_info: dict) -> dict:
    """
    Запускает subprocess для одного div и возвращает результат.
    """
    div = div_info["div"]
    dump_path = div_info["dump_path"]
    result_path = div_info["result_path"]
    
    env = os.environ.copy()
    env["TEST_INPUT_FILE"] = dump_path
    env["HORIZON_MODE"] = "LONG_SIMPLE"
    env["CALC_TEST_DATA"] = "true"
    env["SAVE_RESULT_JSON_PATH"] = result_path
    
    logger.info(f"Phase 1 DIV {div}: Launching subprocess run.py...")
    
    try:
        cmd = [sys.executable, "run.py"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if proc.returncode != 0:
            logger.error(f"Phase 1 DIV {div} subprocess failed with code {proc.returncode}.\nStderr: {proc.stderr}")
            return {"div": div, "success": False, "div_info": div_info}
        else:
            logger.info(f"Phase 1 DIV {div} subprocess finished successfully.")
            return {"div": div, "success": True, "div_info": div_info}
    except Exception as e:
        logger.error(f"Phase 1 DIV {div} subprocess execution error: {e}")
        return {"div": div, "success": False, "div_info": div_info}


def _process_result(
    result: dict,
    allowed_products_map: dict[str, set[str]],
    products_full_df: pd.DataFrame,
) -> None:
    """
    Обрабатывает результат subprocess и обновляет allowed_products_map.
    """
    div = result["div"]
    div_info = result["div_info"]
    result_path = div_info["result_path"]
    machines_div = div_info["machines_div"]
    products_div = div_info["products_div"]
    
    if result["success"] and os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                schedule_data = json.load(f)
            
            for item in schedule_data:
                m_local = item.get("machine_idx")
                p_local = item.get("product_idx")
                
                if m_local is None or p_local is None:
                    continue
                
                if p_local <= 0:
                    continue
                    
                if 0 <= m_local < len(machines_div):
                    m_id = machines_div.iloc[m_local]["id"]
                    if m_id not in allowed_products_map:
                        allowed_products_map[m_id] = set()
                    
                    if 0 <= p_local < len(products_div):
                        p_id = products_div.iloc[p_local]["id"]
                        allowed_products_map[m_id].add(p_id)
            
            # Добавляем начальные продукты
            for m_local in range(len(machines_div)):
                m_id = machines_div.iloc[m_local]["id"]
                init_p_idx_local = int(machines_div.iloc[m_local]["product_idx"])
                if init_p_idx_local > 0:
                    if m_id not in allowed_products_map:
                        allowed_products_map[m_id] = set()
                    if 0 <= init_p_idx_local < len(products_div):
                        p_id = products_div.iloc[init_p_idx_local]["id"]
                        allowed_products_map[m_id].add(p_id)

        except Exception as e:
            logger.error(f"Phase 1 DIV {div}: Failed to read result JSON: {e}")
            _fallback_allow_all(div, div_info, allowed_products_map, products_full_df)
    else:
        logger.warning(f"Phase 1 DIV {div}: Result file not found or process failed. Fallback: allow all valid products.")
        _fallback_allow_all(div, div_info, allowed_products_map, products_full_df)


def _fallback_allow_all(
    div: int,
    div_info: dict,
    allowed_products_map: dict[str, set[str]],
    products_full_df: pd.DataFrame,
) -> None:
    """
    Fallback: разрешаем все совместимые продукты для машин div.
    """
    machines_div = div_info["machines_div"]
    products_div = div_info["products_div"]
    
    machines_arr = MachinesDFToArray(machines_div)
    products_arr = ProductsDFToArray(products_div)
    prod_divs_local = products_div["div"].fillna(0).astype(int).tolist()
    
    for m_local in range(len(machines_arr)):
        m_id = machines_div.iloc[m_local]["id"]
        m_type = machines_arr[m_local][3]
        if m_id not in allowed_products_map:
            allowed_products_map[m_id] = set()
        for p_local in range(len(products_arr)):
            p_type = products_arr[p_local][3]
            p_div = prod_divs_local[p_local]
            if (p_type == 0 or p_type == m_type) and (p_div == 0 or p_div == div):
                p_id = products_div.iloc[p_local]["id"]
                allowed_products_map[m_id].add(p_id)


def solve_phase1_allocation(
    data_full: dict,
    machines_full_df: pd.DataFrame,
    products_full_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    count_days: int,
) -> dict[int, set[int]]:
    """
    Фаза 1: Решаем задачу распределения ПАРАЛЛЕЛЬНО для каждого цеха (div).
    Цель: Определить множество допустимых продуктов для каждой машины (allowed_products[m]).
    
    Возвращает: {machine_internal_idx: {product_internal_idx, ...}}
    """
    # Определяем список цехов
    divs = machines_full_df["div"].unique().tolist()
    divs = [int(d) if pd.notna(d) and d > 0 else 1 for d in divs]
    divs = sorted(list(set(divs)))
    
    logger.info(f"Two-Phase Phase 1: Solving for divisions {divs} IN PARALLEL")
    
    allowed_products_map: dict[str, set[str]] = {}
    
    # Словари для маппинга ID
    machine_id_to_internal_full = {row["id"]: i for i, row in machines_full_df.iterrows()}
    product_id_to_internal_full = {row["id"]: i for i, row in products_full_df.iterrows()}
    
    # 1. Подготавливаем данные для всех div
    div_infos = []
    for div in divs:
        logger.info(f"--- Phase 1: Preparing DIV {div} ---")
        div_info = _prepare_div_data(div, data_full, machines_full_df, products_full_df, clean_df, count_days)
        if div_info is not None:
            div_infos.append(div_info)
    
    if not div_infos:
        logger.warning("No divisions to process in Phase 1")
        return {}
    
    # 2. Запускаем subprocess'ы ПАРАЛЛЕЛЬНО
    logger.info(f"Phase 1: Launching {len(div_infos)} subprocesses in parallel...")
    
    results = []
    with ThreadPoolExecutor(max_workers=len(div_infos)) as executor:
        futures = {executor.submit(_run_subprocess, di): di for di in div_infos}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                div_info = futures[future]
                logger.error(f"Phase 1 DIV {div_info['div']} future failed: {e}")
                results.append({"div": div_info["div"], "success": False, "div_info": div_info})
    
    logger.info(f"Phase 1: All {len(results)} subprocesses completed")
    
    # 3. Обрабатываем результаты
    for result in results:
        _process_result(result, allowed_products_map, products_full_df)
    
    # 4. Конвертируем allowed_products_map (по ID) в allowed_products_internal (по индексам)
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
