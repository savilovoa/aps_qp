"""Diagnostic: find structural infeasibility causes in FULL model input."""
import json, sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "example/test_in.json"
with open(input_file, encoding="utf8") as f:
    data = json.load(f)

prod_lday = {}
prod_name = {}
for p in data["products"]:
    prod_lday[p["idx"]] = p["lday"]
    prod_name[p["idx"]] = p["name"].strip()

count_days = data["count_days"]
max_daily_prod_zero = data["max_daily_prod_zero"]

print(f"count_days={count_days}, max_daily_prod_zero={max_daily_prod_zero}")
print(f"machines={len(data['machines'])}, products with qty>0: "
      f"{sum(1 for p in data['products'] if p['qty'] > 0)}")
print()

# --- Check 1: remain_day > lday (negative start_val) ---
print("=== Check 1: remain_day vs lday (days_in_batch domain) ===")
issues_found = False
for m in data["machines"]:
    p_idx = m["product_idx"]
    lday = prod_lday.get(p_idx, 0)
    remain = m["remain_day"]
    if remain <= 0:
        continue
    start_val = lday - remain + 1
    if start_val < 0:
        print(f"  *** INFEASIBLE: machine {m['idx']} ({m['name'].strip()}): "
              f"product {p_idx} ({prod_name.get(p_idx,'?')}), "
              f"lday={lday}, remain_day={remain} -> start_val={start_val}")
        issues_found = True
    elif start_val == 0:
        print(f"  WARNING: machine {m['idx']} ({m['name'].strip()}): "
              f"start_val=0 (edge case), lday={lday}, remain_day={remain}")
        issues_found = True

if not issues_found:
    print("  OK")

# --- Check 2: clean on forced days ---
print()
print("=== Check 2: cleans during forced remain_day period ===")
cleans_set = set()
for c in data["cleans"]:
    cleans_set.add((c["machine_idx"], c["day_idx"]))

issues2 = False
for m in data["machines"]:
    remain = m["remain_day"]
    if remain <= 0:
        continue
    forced_days_used = 0
    for d in range(count_days):
        if (m["idx"], d) not in cleans_set:
            forced_days_used += 1
            if forced_days_used >= remain:
                break

# --- Check 3: transition capacity ---
print()
print("=== Check 3: transition capacity (max_daily_prod_zero) ===")
# Count machines that NEED at least one transition (start on qty=0 product)
need_transition = []
for m in data["machines"]:
    p_idx = m["product_idx"]
    p_qty = 0
    for p in data["products"]:
        if p["idx"] == p_idx:
            p_qty = p["qty"]
            break
    # machines starting on qty=0 products ideally need a transition
    if p_qty == 0 and m["remain_day"] < count_days:
        # earliest transition day = remain_day (after batch ends)
        remain = m["remain_day"]
        p_lday = prod_lday.get(p_idx, 0)
        if p_lday > 0 and remain < p_lday:
            earliest = remain  # batch hasn't ended yet; need to wait for lday
            # but start_val = lday - remain + 1, batch ends when days_in_batch == lday
            # that's after remain more working days
            earliest = remain
        else:
            earliest = 0  # batch already done or lday issue
        need_transition.append((m["idx"], m["name"].strip(), p_idx, earliest))

print(f"  Machines needing transition (start on qty=0 product): {len(need_transition)}")
for midx, mname, pidx, earliest in need_transition:
    print(f"    machine {midx} ({mname}): product {pidx}, earliest transition day ~{earliest}")

# How many transition slots per day?
total_transition_days_needed = len(need_transition) * 2  # 2 days per transition
available_slots = count_days * max_daily_prod_zero
print(f"  Total transition-days needed: {total_transition_days_needed}")
print(f"  Available slots (count_days * max_daily_prod_zero): {available_slots}")

# --- Check 4: capacity by machine type ---
print()
print("=== Check 4: capacity by machine type ===")
type1_machines = sum(1 for m in data["machines"] if m["type"] == 1)
type0_machines = sum(1 for m in data["machines"] if m["type"] == 0)
type1_products_qty = sum(p["qty"] for p in data["products"] if p["machine_type"] == 1 and p["qty"] > 0)
any_products_qty = sum(p["qty"] for p in data["products"] if p["qty"] > 0)
type1_capacity = type1_machines * count_days
total_capacity = len(data["machines"]) * count_days - len(data["cleans"])
print(f"  type=1 machines: {type1_machines}, capacity: {type1_capacity}")
print(f"  type=0 machines: {type0_machines}")
print(f"  type=1-only products total qty: {type1_products_qty}")
print(f"  all products total qty: {any_products_qty}")
print(f"  total capacity (minus cleans): {total_capacity}")

print()
print("=== Summary ===")
if issues_found:
    print("STRUCTURAL INFEASIBILITY DETECTED in Check 1.")
    print("Fix: cap remain_day to lday, or set remain_day = min(remain_day, lday)")
