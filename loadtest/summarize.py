"""Turn locust's *_stats.csv into a Markdown table (for the CI summary and docs/PERFORMANCE.md)."""

from __future__ import annotations

import csv
import sys


def main(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("| endpoint | requests | failures | median ms | p95 ms | p99 ms | req/s |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        name = r["Name"] if r["Name"] != "Aggregated" else "**all**"
        print(
            f"| {name} | {r['Request Count']} | {r['Failure Count']} | {r['50%']} | {r['95%']} | {r['99%']} | "
            f"{float(r['Requests/s']):.2f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
