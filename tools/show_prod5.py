import json
from pathlib import Path

p = Path("example/test_in.json")
with p.open(encoding="utf-8") as f:
    data = json.load(f)

for prod in data["products"]:
    if prod.get("idx") == 5:
        print(prod)
