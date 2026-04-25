import json
with open("example/test_in.json", encoding="utf8") as f:
    data = json.load(f)
cleans = {(c["machine_idx"], c["day_idx"]) for c in data["cleans"]}
cd = data["count_days"]

# capacity and forced per div
cap, forced_d = {}, {}
for m in data["machines"]:
    d = m["div"]
    wd = sum(1 for day in range(cd) if (m["idx"], day) not in cleans)
    cap[d] = cap.get(d, 0) + wd
    remain = max(0, m["remain_day"])
    forced_d[d] = forced_d.get(d, 0) + min(remain, wd)

print("Capacity per div:", cap)
print("Forced per div:", forced_d)

# simulate auto-relax
relaxed_idx = set()
for p in data["products"]:
    if p["idx"] == 0 or p["qty"] <= 0:
        continue
    qm = int(p.get("qty_minus", 0) or 0)
    if qm != 0:
        continue
    ms = [m for m in data["machines"] if m["product_idx"] == p["idx"]]
    tf = sum(max(0, m["remain_day"]) for m in ms)
    if tf > p["qty"]:
        relaxed_idx.add(p["idx"])
    elif 0 < tf < p["qty"]:
        gap = p["qty"] - tf
        lday = p["lday"] if p["lday"] > 0 else 10
        min_add = min(lday, max(0, cd - 2))
        if min_add > gap:
            relaxed_idx.add(p["idx"])

# strict products per div
strict_per_div = {}
print("\nStrict products after auto-relax:")
for p in data["products"]:
    if p["idx"] == 0 or p["qty"] <= 0:
        continue
    qm = int(p.get("qty_minus", 0) or 0)
    if qm != 0:
        continue
    if p["idx"] in relaxed_idx:
        continue
    d = p.get("div", 0)
    strict_per_div[d] = strict_per_div.get(d, 0) + p["qty"]
    mt = p.get("machine_type", 0)
    print(f"  product {p['idx']:2d} ({p['name'].strip():30s}) qty={p['qty']:3d} div={d} machine_type={mt}")

print("\nStrict qty per div:", strict_per_div)

# available per div
print("\nPer-div feasibility:")
for d in sorted(cap.keys()):
    machines_in_div = [m for m in data["machines"] if m["div"] == d]
    trans_est = sum(
        2 for m in machines_in_div
        if max(0, m["remain_day"]) < sum(
            1 for day in range(cd) if (m["idx"], day) not in cleans
        )
    )
    avail = cap[d] - forced_d[d] - trans_est
    strict = strict_per_div.get(d, 0)
    deficit = "*** DEFICIT ***" if strict > avail else "OK"
    print(
        f"  div={d}: capacity={cap[d]}, forced={forced_d[d]}, "
        f"transitions~{trans_est}, available~{avail}, strict_needed={strict} {deficit}"
    )
