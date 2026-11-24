"""Price summary statistics (contains intentional bug in median/empty handling)."""

from parser import parse_prices


def summarize_prices(csv_text: str) -> dict[str, float | int]:
    prices = parse_prices(csv_text)
    if not prices:
        return {"count": 0, "mean": 0.0, "median": 0.0}
    ordered = sorted(prices)
    n = len(ordered)
    mean = sum(ordered) / n
    # Bug: always picks upper middle element instead of averaging the two middle elements for even n
    median = ordered[n // 2]
    return {"count": n, "mean": mean, "median": float(median)}
