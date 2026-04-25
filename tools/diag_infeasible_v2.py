"""Extended diagnostic: find ALL infeasibility causes in FULL model input."""
import json, sys, math

input_file = sys.argv[1] if len(sys.argv) > 1 else "example/test_in.json"
with open(input_file, encoding="utf8") as f:
    data = json.load(f)

count_days = data["count_days"]
max_daily_prod_zero = data["max_daily_prod_zero"]

# --- Compute effective ldays (same logic as create_model) ---
remains = data.get("remains", [])
remains_batches = []
if remains:
    first = remains[0]
    if isinstance(first, list) and (not first or isinstance(first[0], (int, float))):
        remains_batches = remains
    elif isinstance(first, list) and first and isinstance(first[0], list):
        remains_batches = remains[0]

prod_map = {}
for p in data["products"]:
    base_lday = p["lday"]
    qty = p["qty"]
    if qty > 0 and base_lday <= 0:
        base_lday = 10
    lday_eff = base_lday
    src_root = p.get("src_root", -1)
    if isinstance(src_root, int) and 0 <= src_root < len(remains_batches):
        batches = remains_batches[src_root]
        if isinstance(batches, list) and len(batches) > 0:
            long_batches = [int(b) for b in batches if isinstance(b, (int, float)) and int(b) >= 6]
            if long_batches:
                avg = round(sum(long_batches) / len(long_batches))
                if avg > 0:
                    lday_eff = avg
    prod_map[p["idx"]] = {
        "name": p["name"].strip(),
        "qty": qty,
        "lday_base": p["lday"],
        "lday_eff": lday_eff,
        "machine_type": p.get("machine_type", 0),
        "qty_minus": int(p.get("qty_minus", 0) or 0),
        "div": p.get("div", 0),
        "src_root": src_root,
    }

cleans_set = set()
for c in data["cleans"]:
    cleans_set.add((c["machine_idx"], c["day_idx"]))

print(f"count_days={count_days}, max_daily_prod_zero={max_daily_prod_zero}")
print(f"machines={len(data['machines'])}, products total={len(data['products'])}")
print()

# === Check 1: remain_day vs effective lday ===
print("=== Check 1: remain_day vs EFFECTIVE lday ===")
issues1 = False
for m in data["machines"]:
    p_idx = m["product_idx"]
    pi = prod_map.get(p_idx)
    if not pi:
        continue
    remain = m["remain_day"]
    if remain <= 0:
        continue
    lday = pi["lday_eff"]
    start_val = lday - remain + 1
    if start_val < 1:
        print(f"  WARN machine {m['idx']} ({m['name'].strip()}): "
              f"product {p_idx} ({pi['name']}), "
              f"lday_eff={lday} (base={pi['lday_base']}), remain_day={remain} "
              f"-> start_val={start_val} (capped to 1)")
        issues1 = True
if not issues1:
    print("  OK")

# === Check 2: QTY_MINUS forced overproduction ===
print()
print("=== Check 2: forced production vs qty (QTY_MINUS conflicts) ===")
print("  (Relevant if APPLY_QTY_MINUS=True)")
# For each product, count total forced days from remain_day
forced_by_product = {}
for m in data["machines"]:
    p_idx = m["product_idx"]
    remain = m["remain_day"]
    if remain > 0 and p_idx > 0:
        # How many working days will this machine be forced on this product?
        work_days_forced = 0
        for d in range(count_days):
            if (m["idx"], d) not in cleans_set:
                work_days_forced += 1
                if work_days_forced >= remain:
                    break
        forced_by_product.setdefault(p_idx, []).append(
            (m["idx"], m["name"].strip(), work_days_forced)
        )

issues2 = False
for p_idx, machines_list in sorted(forced_by_product.items()):
    pi = prod_map.get(p_idx)
    if not pi:
        continue
    total_forced = sum(wd for _, _, wd in machines_list)
    qty = pi["qty"]
    qm = pi["qty_minus"]
    if qty > 0 and qm == 0 and total_forced > qty:
        print(f"  *** CONFLICT: product {p_idx} ({pi['name']}): "
              f"qty={qty}, qty_minus=0 (STRICT), but forced_days={total_forced}")
        for midx, mname, wd in machines_list:
            print(f"      machine {midx} ({mname}): forced {wd} working days")
        print(f"      -> model requires product_counts == {qty}, but forced >= {total_forced}")
        print(f"      -> auto-relax should handle this, but check logs")
        issues2 = True
    elif qty > 0 and total_forced > qty:
        print(f"  INFO: product {p_idx} ({pi['name']}): "
              f"qty={qty}, forced={total_forced} (qty_minus={qm}, flexible)")

if not issues2:
    print("  OK (no strict qty_minus conflicts)")

# === Check 3: lday > count_days (batch can never complete) ===
print()
print("=== Check 3: lday_eff > count_days (batch never completes) ===")
issues3 = False
for m in data["machines"]:
    p_idx = m["product_idx"]
    pi = prod_map.get(p_idx)
    if not pi:
        continue
    remain = m["remain_day"]
    lday = pi["lday_eff"]
    if remain <= 0:
        continue
    # Working days on this machine
    work_days_total = sum(1 for d in range(count_days) if (m["idx"], d) not in cleans_set)
    # After capping start_val to 1, the batch starts from 1 and needs lday days to complete
    # But if remain_day < lday, the real start_val = lday - remain + 1 and batch completes after remain days
    if remain < lday:
        days_to_complete_batch = remain
    else:
        # start_val capped to 1, need lday working days to complete
        days_to_complete_batch = lday
    if days_to_complete_batch > work_days_total:
        print(f"  WARN machine {m['idx']} ({m['name'].strip()}): "
              f"product {p_idx}, lday_eff={lday}, remain={remain}, "
              f"work_days={work_days_total}, days_to_complete={days_to_complete_batch} "
              f"-> batch never completes in horizon")
        issues3 = True
if not issues3:
    print("  OK")

# === Check 4: ZERO_PER_MACHINE_LIMIT conflicts ===
print()
print("=== Check 4: machines that MUST transition but have limited zero slots ===")
weeks = max(1, count_days // 21)
max_zero_per_machine = 2 * weeks
# Machines that start on product with remain_day=0: immediate batch complete,
# may need transition on day 0 itself
issues4 = False
for m in data["machines"]:
    p_idx = m["product_idx"]
    pi = prod_map.get(p_idx)
    if not pi:
        continue
    remain = m["remain_day"]
    lday = pi["lday_eff"]
    work_days = sum(1 for d in range(count_days) if (m["idx"], d) not in cleans_set)
    # How many batch completions happen? Each needs potential transition
    if remain <= 0:
        continue
    # After initial batch, how many full batches fit?
    if remain < lday:
        days_after_initial = work_days - remain
    else:
        days_after_initial = work_days - lday  # capped start_val
    if days_after_initial <= 0:
        continue  # machine locked for entire horizon
    full_batches_after = days_after_initial // lday if lday > 0 else 0
    # Each batch boundary is a POTENTIAL transition point, not mandatory

# === Check 5: product lday_eff comparison with base ===
print()
print("=== Check 5: products where remains changed effective lday ===")
for p_idx, pi in sorted(prod_map.items()):
    if pi["lday_eff"] != pi["lday_base"] and pi["lday_base"] > 0:
        print(f"  product {p_idx} ({pi['name']}): base lday={pi['lday_base']} -> eff lday={pi['lday_eff']} "
              f"(src_root={pi['src_root']})")

# === Check 6: Summary of machine initial assignments ===
print()
print("=== Check 6: machine initial state summary ===")
for m in data["machines"]:
    p_idx = m["product_idx"]
    pi = prod_map.get(p_idx, {})
    remain = m["remain_day"]
    lday = pi.get("lday_eff", 0)
    work_days = sum(1 for d in range(count_days) if (m["idx"], d) not in cleans_set)
    locked_all = remain >= work_days
    status = "LOCKED" if locked_all else f"free after ~{remain} days"
    print(f"  m={m['idx']:2d} type={m['type']} div={m['div']} "
          f"product={p_idx:2d} ({pi.get('name','?'):25s}) "
          f"remain={remain:2d} lday_eff={lday:2d} work_days={work_days:2d} -> {status}")
