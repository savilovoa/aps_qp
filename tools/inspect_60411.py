import json
from pathlib import Path

from src.config import settings


def main() -> None:
    base = Path(settings.BASE_DIR)
    raw_path = settings.TEST_INPUT_FILE or (base / "example" / "test_in.json")
    input_path = Path(raw_path)
    with input_path.open("r", encoding="utf8") as f:
        data = json.load(f)

    print("INPUT FILE:", input_path)

    print("\nTARGET PRODUCTS (name contains '60411'):")
    for p in data["products"]:
        if "60411" in p.get("name", ""):
            print(p)

    print("\nMACHINES (idx, name, type, div if present):")
    for i, m in enumerate(data["machines"]):
        print(i, m.get("name"), "type=", m.get("type"), "div=", m.get("div"))


if __name__ == "__main__":  # pragma: no cover
    main()
