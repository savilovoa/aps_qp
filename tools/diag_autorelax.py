"""Simulate auto-relax logic from schedule_loom_calc and check remaining strict products."""
import json, sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "example/test_in.json"
with open(input_file, encoding="utf8") as f:
    data = json.load(f)

count_days = data["count_days"]
products = {p["idx"]: dict(p) for p in data["products"]}
machines = data["machines"]

# Simulate auto-relax (same logic as schedule_loom_calc lines 1589-1671)
relaxed = []
for p in data["products"]:
    p_idx = p["idx"]
    if p_idx == 0:
        continue
    qty = int(p.get("qty", 0) or 0)
    if qty <= 0:
        continue
    qty_minus_flag = int(p.get("qty_minus", 0) or 0)
    if qty_minus_flag != 0:
        continue  # already flexible

    lday_val = int(p.get("lday", 0) or 0)
    if lday_val <= 0:
        lday_val = 10

    # Machines starting with this product
    machines_with_p = [m for m in machines if m["product_idx"] == p_idx]
    remain_days_list = [int(m.get("remain_day", 0) or 0) for m in machines_with_p if int(m.get("remain_day", 0) or 0) > 0]
    total_forced = sum(remain_days_list)

    need_relax = False
    reason = ""

    # Check A: total remain > qty
    if total_forced > qty:
        need_relax = True
        reason = f"total remain_day={total_forced} > qty={qty}"

    # Check B: batch granularity
    elif 0 < total_forced < qty:
        gap = qty - total_forced
        min_continue = min(
            (count_days - rd for rd in remain_days_list),
            default=float("inf"),
        )
        min_new_machine = min(lday_val, max(0, count_days - 2))
        min_additional = min(min_continue, min_new_machine)
        if min_additional > gap:
            need_relax = True
            reason = (f"batch granularity: forced={total_forced}, qty={qty}, "
                      f"gap={gap}, min_additional={min_additional}, lday={lday_val}")

    if need_relax:
        relaxed.append((p_idx, p["name"].strip(), qty, reason))
        products[p_idx]["qty_minus"] = 1
        products[p_idx]["qty_minus_min"] = qty

print("=== Auto-relax results ===")
if relaxed:
    for p_idx, name, qty, reason in relaxed:
        print(f"  RELAXED: product {p_idx} ({name}): qty={qty} -> qty_minus=1, qty_minus_min={qty}")
        print(f"           reason: {reason}")
else:
    print("  No products relaxed")

# Now check: which strict products remain?
print()
print("=== Remaining STRICT products (qty_minus=0, qty>0) ===")
strict = []
for p in data["products"]:
    p_idx = p["idx"]
    if p_idx == 0:
        continue
    pi = products[p_idx]
    qty = int(pi.get("qty", 0) or 0)
    if qty <= 0:
        continue
    qm = int(pi.get("qty_minus", 0) or 0)
    if qm == 0:
        strict.append((p_idx, pi["name"].strip(), qty, pi.get("lday", 0), pi.get("machine_type", 0)))

total_strict_qty = 0
for p_idx, name, qty, lday, mtype in strict:
    total_strict_qty += qty
    # Check: does any machine start on this product?
    start_machines = [m for m in machines if m["product_idx"] == p_idx]
    forced = sum(int(m.get("remain_day", 0) or 0) for m in start_machines)
    print(f"  product {p_idx:2d} ({name:30s}): qty={qty:3d}, lday={lday:2d}, "
          f"machine_type={mtype}, starts_on={len(start_machines)} machines, forced={forced}")

print(f"\nTotal strict qty required: {total_strict_qty}")

# Capacity analysis
cleans_set = set()
for c in data["cleans"]:
    cleans_set.add((c["machine_idx"], c["day_idx"]))

total_capacity = sum(
    1 for m in machines for d in range(count_days) if (m["idx"], d) not in cleans_set
)

# Forced days by all products (strict + relaxed + qty=0 starts)
forced_days_total = 0
for m in machines:
    remain = int(m.get("remain_day", 0) or 0)
    if remain > 0:
        work_days = sum(1 for d in range(count_days) if (m["idx"], d) not in cleans_set)
        forced_days_total += min(remain, work_days)

# Transitions needed: 2 days each for machines that need to switch
need_switch = []
for m in machines:
    p_idx = m["product_idx"]
    remain = int(m.get("remain_day", 0) or 0)
    work_days = sum(1 for d in range(count_days) if (m["idx"], d) not in cleans_set)
    if remain < work_days:  # machine has free time after initial batch
        need_switch.append(m["idx"])

transition_days = len(need_switch) * 2  # 2 days per transition

available_for_strict = total_capacity - forced_days_total - transition_days
print(f"\nCapacity: total={total_capacity}, forced={forced_days_total}, "
      f"transitions={transition_days} ({len(need_switch)} machines)")
print(f"Available for new product batches: ~{available_for_strict}")
print(f"Strict qty needed: {total_strict_qty}")

if available_for_strict < total_strict_qty:
    print("*** CAPACITY DEFICIT for strict products! ***")
