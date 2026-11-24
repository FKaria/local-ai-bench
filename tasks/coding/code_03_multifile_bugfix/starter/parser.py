"""CSV price parser (contains intentional bug with quoted commas)."""


def parse_prices(csv_text: str) -> list[float]:
    lines = [line for line in csv_text.splitlines() if line.strip()]
    if not lines:
        return []
    prices: list[float] = []
    # Bug: naive split(',') breaks on quoted item names containing commas!
    for line in lines[1:]:
        parts = line.split(",")
        prices.append(float(parts[1].strip()))
    return prices
