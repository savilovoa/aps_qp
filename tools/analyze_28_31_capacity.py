import json
from pathlib import Path

path = Path("example/test_in.json")
with path.open(encoding="utf-8") as f:
    data = json.load(f)

products = data["products"]
machines = data["machines"]
orig_count_days = data["count_days"]
count_days_simple = (orig_count_days + 3 - 1) // 3

print(f"orig_count_days={orig_count_days}, simple_days={count_days_simple}")

def_prod_div = 0
def_m_div = 1

for p in products:
    idx = p["idx"]
    if idx == 0:
        continue
    name = p["name"]
    qty_shifts = int(p.get("qty", 0) or 0)
    plan_days = (qty_shifts + 3 - 1) // 3 if qty_shifts > 0 else 0
    prod_mt = int(p.get("machine_type", 0) or 0)
    prod_div = int(p.get("div", def_prod_div) or 0)
    qmm_shifts = int(p.get("qty_minus_min", 0) or 0)
    min_days = (qmm_shifts + 3 - 1) // 3 if qmm_shifts > 0 else 0

    compat = 0
    for m in machines:
        m_type = int(m.get("type", 0) or 0)
        m_div = int(m.get("div", def_m_div) or 0)
        type_incompatible = (prod_mt > 0 and m_type != prod_mt)
        div_incompatible = (prod_div in (1, 2) and m_div != prod_div)
        if not (type_incompatible or div_incompatible):
            compat += 1
    cap_days = compat * count_days_simple

    if idx in (28, 29, 31, 32):
        print(
            f"idx={idx} name={name} qty={qty_shifts} plan_days={plan_days} "
            f"min_days={min_days} compat_machines={compat} cap_days={cap_days}"
        )
