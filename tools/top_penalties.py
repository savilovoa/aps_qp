from pathlib import Path

from src.config import settings
from tools.compare_long_vs_simple import load_input, collect_stats


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    print(f"APPLY_QTY_MINUS={settings.APPLY_QTY_MINUS}")

    def run_scenario(label: str, horizon_mode: str, flags: dict) -> None:
        settings.HORIZON_MODE = horizon_mode
        settings.APPLY_PROP_OBJECTIVE = True
        settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = flags["APPLY_OVERPENALTY_INSTEAD_OF_PROP"]
        settings.SIMPLE_USE_PROP_MULT = flags["SIMPLE_USE_PROP_MULT"]

        stats, prop_penalty, per_prod_penalty = collect_stats(
            horizon_mode, data, machines, products, cleans, remains
        )

        items: list[tuple[int, int, str, int, int]] = []
        for idx, prod in enumerate(data["products"]):
            if idx == 0:
                continue
            s = stats.get(idx)
            if not s:
                continue
            plan = int(s["plan"])
            fact = int(s["fact"])
            pen = int(per_prod_penalty.get(idx, 0))
            items.append((pen, idx, prod["name"], plan, fact))

        items.sort(reverse=True, key=lambda x: x[0])

        print(f"\n=== {label}: total_penalty={prop_penalty} ===")
        print("pen\tidx\tname\tplan\tfact\tdelta")
        for pen, idx, name, plan, fact in items[:20]:
            print(f"{pen}\t{idx}\t{name}\t{plan}\t{fact}\t{fact - plan}")

    scenarios = [
        ("LONG_OVER", "LONG", {"APPLY_OVERPENALTY_INSTEAD_OF_PROP": True, "SIMPLE_USE_PROP_MULT": False}),
        ("LS_OVER", "LONG_SIMPLE", {"APPLY_OVERPENALTY_INSTEAD_OF_PROP": False, "SIMPLE_USE_PROP_MULT": False}),
        ("LS_PROP", "LONG_SIMPLE", {"APPLY_OVERPENALTY_INSTEAD_OF_PROP": False, "SIMPLE_USE_PROP_MULT": True}),
    ]

    for label, horizon_mode, flags in scenarios:
        run_scenario(label, horizon_mode, flags)


if __name__ == "__main__":  # pragma: no cover
    main()
