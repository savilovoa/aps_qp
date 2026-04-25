"""Check if strict qty is achievable given one-transition-per-machine limit."""
import json
from itertools import combinations

with open("example/test_in.json", encoding="utf8") as f:
    data = json.load(f)

cd = data["count_days"]
cleans = {(c["machine_idx"], c["day_idx"]) for c in data["cleans"]}
prod_map = {p["idx"]: p for p in data["products"]}

# Simulate auto-relax
relaxed_idx = set()
for p in data["products"]:
    if p["idx"] == 0 or p["qty"] <= 0: continue
    qm = int(p.get("qty_minus", 0) or 0)
    if qm != 0: continue
    ms = [m for m in data["machines"] if m["product_idx"] == p["idx"]]
    tf = sum(max(0, m["remain_day"]) for m in ms)
    if tf > p["qty"]: relaxed_idx.add(p["idx"])
    elif 0 < tf < p["qty"]:
        gap = p["qty"] - tf
        lday = p["lday"] if p["lday"] > 0 else 10
        if min(lday, max(0, cd - 2)) > gap: relaxed_idx.add(p["idx"])

# Per div: find strict products and compute free days per machine
for div in [1, 2]:
    machines_in_div = [m for m in data["machines"] if m["div"] == div]
    strict_in_div = []
    for p in data["products"]:
        if p["idx"] == 0 or p["qty"] <= 0: continue
        if int(p.get("qty_minus", 0) or 0) != 0: continue
        if p["idx"] in relaxed_idx: continue
        if p.get("div", 0) != div: continue
        strict_in_div.append((p["idx"], p["name"].strip(), p["qty"]))

    if not strict_in_div:
        continue

    # Free days per machine (after forced + one 2-day transition)
    free_days = []
    for m in machines_in_div:
        wd = sum(1 for d in range(cd) if (m["idx"], d) not in cleans)
        remain = max(0, m["remain_day"])
        forced = min(remain, wd)
        # 2 days for transition (ZERO_PER_MACHINE_LIMIT allows max 1 transition)
        free = wd - forced - 2
        if free < 0: free = 0
        free_days.append((m["idx"], m["name"].strip(), free))

    free_vals = [f[2] for f in free_days]
    total_free = sum(free_vals)
    total_strict = sum(q for _, _, q in strict_in_div)

    print(f"=== div={div} ===")
    print(f"  Strict products: {[(idx, name, qty) for idx, name, qty in strict_in_div]}")
    print(f"  Machine free days (after forced + 1 transition): {[(f[0], f[2]) for f in free_days]}")
    print(f"  Total free: {total_free}, Total strict needed: {total_strict}")

    if total_free < total_strict:
        print(f"  *** CAPACITY DEFICIT: {total_free} < {total_strict} ***")
        continue

    # Check: can free_vals be partitioned into groups meeting each product's qty?
    # For == constraint: need exact match
    # For >= constraint (retry): need at least the value
    qtys = [q for _, _, q in strict_in_div]
    print(f"\n  Partition check: can {free_vals} be split into groups summing to {qtys}?")

    # Brute force: try all possible assignments of machines to products
    n_machines = len(free_vals)
    n_products = len(qtys)
    found_exact = False
    found_geq = False

    # Try all assignments (each machine -> one product)
    from itertools import product as iproduct
    if n_machines <= 12:  # only feasible for small N
        for assignment in iproduct(range(n_products), repeat=n_machines):
            sums = [0] * n_products
            for mi, pi in enumerate(assignment):
                sums[pi] += free_vals[mi]
            if all(sums[i] == qtys[i] for i in range(n_products)):
                found_exact = True
                break
            if not found_geq and all(sums[i] >= qtys[i] for i in range(n_products)):
                found_geq = True

        print(f"  Exact partition (==): {'FOUND' if found_exact else '*** NOT POSSIBLE ***'}")
        print(f"  GEQ partition (>=):   {'FOUND' if found_geq else '*** NOT POSSIBLE ***'}")
        if not found_geq:
            print(f"  -> Even INFEASIBLE retry cannot help!")
            print(f"  -> Root cause: APPLY_ZERO_PER_MACHINE_LIMIT restricts each machine")
            print(f"     to 1 transition (max 2 PRODUCT_ZERO days), locking it on 1 product.")
            print(f"  -> Fix: set APPLY_ZERO_PER_MACHINE_LIMIT=false in .env")
