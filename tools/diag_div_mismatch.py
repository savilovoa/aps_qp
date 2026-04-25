"""Check if any machine starts with a product from a different div."""
import json
with open("example/test_in.json", encoding="utf8") as f:
    data = json.load(f)

prod_div = {p["idx"]: p.get("div", 0) for p in data["products"]}
prod_name = {p["idx"]: p["name"].strip() for p in data["products"]}

print("=== Initial product div vs machine div ===")
issues = 0
for m in data["machines"]:
    p_idx = m["product_idx"]
    p_d = prod_div.get(p_idx, 0)
    m_d = m["div"]
    mismatch = ""
    if p_d in (1, 2) and m_d != p_d:
        mismatch = " *** DIV MISMATCH - INFEASIBLE on day 0! ***"
        issues += 1
    if mismatch or m["remain_day"] < 0:
        print(f"  machine {m['idx']:2d} (div={m_d}) -> product {p_idx} "
              f"({prod_name.get(p_idx,'?')}, div={p_d}) remain={m['remain_day']}{mismatch}")

if issues == 0:
    print("  No div mismatches found")
else:
    print(f"\n*** {issues} MISMATCH(ES) FOUND - these cause INFEASIBLE ***")
