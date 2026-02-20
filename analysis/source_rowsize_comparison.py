#!/usr/bin/env python3
"""
source_rowsize_comparison.py

Compare row counts between DIME CSV and DIME Parquet for one or more years.

Usage:
    python source_rowsize_comparison.py 2020
    python source_rowsize_comparison.py 2018 2020 2022
    python source_rowsize_comparison.py 2018 2020 --fail-on-mismatch
"""

from __future__ import annotations

import argparse
import csv
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import duckdb

from utils.paths import DIME_DATA_DIR, DIME_PARQUET_FILE


def count_csv_data_rows(csv_path: Path) -> int:
    """Count data rows in a CSV file (excluding header)."""
    count = 0
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for _ in reader:
            count += 1
    return count


def count_parquet_rows(parquet_path: Path) -> int:
    """Count rows in a parquet file using DuckDB."""
    con = duckdb.connect()
    try:
        return int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(parquet_path)]).fetchone()[0])
    finally:
        con.close()


def compare_year(year: int) -> dict:
    csv_path = Path(DIME_DATA_DIR(year))
    parquet_path = Path(DIME_PARQUET_FILE(year))

    result = {
        "year": year,
        "csv_path": csv_path,
        "parquet_path": parquet_path,
        "csv_rows": None,
        "parquet_rows": None,
        "diff": None,
        "status": "ok",
        "error": None,
    }

    if not csv_path.exists():
        result["status"] = "error"
        result["error"] = f"CSV missing: {csv_path}"
        return result

    if not parquet_path.exists():
        result["status"] = "error"
        result["error"] = f"Parquet missing: {parquet_path}"
        return result

    try:
        csv_rows = count_csv_data_rows(csv_path)
        parquet_rows = count_parquet_rows(parquet_path)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    diff = parquet_rows - csv_rows

    result["csv_rows"] = csv_rows
    result["parquet_rows"] = parquet_rows
    result["diff"] = diff
    result["status"] = "match" if diff == 0 else "mismatch"
    return result


def print_results(results: list[dict]) -> None:
    print("\nRow Count Comparison (CSV vs Parquet)")
    print("=" * 74)
    print(f"{'YEAR':>6}  {'CSV_ROWS':>14}  {'PARQUET_ROWS':>14}  {'DIFF(P-C)':>12}  STATUS")
    print("-" * 74)

    for r in results:
        if r["status"] == "error":
            print(f"{r['year']:>6}  {'-':>14}  {'-':>14}  {'-':>12}  ERROR")
            print(f"{'':>6}  {r['error']}")
            continue

        print(
            f"{r['year']:>6}  "
            f"{r['csv_rows']:>14,}  "
            f"{r['parquet_rows']:>14,}  "
            f"{r['diff']:>+12,}  "
            f"{r['status'].upper()}"
        )

    print("=" * 74)

    n_error = sum(1 for r in results if r["status"] == "error")
    n_mismatch = sum(1 for r in results if r["status"] == "mismatch")
    n_match = sum(1 for r in results if r["status"] == "match")

    print(f"Summary: {n_match} match, {n_mismatch} mismatch, {n_error} error")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DIME CSV row counts to parquet row counts for each provided year."
    )
    parser.add_argument("years", type=int, nargs="+", help="Year(s) to compare, e.g. 2018 2020 2022")
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with code 1 if any mismatch or error is found.",
    )
    args = parser.parse_args()

    results = [compare_year(y) for y in args.years]
    print_results(results)

    has_problem = any(r["status"] in {"mismatch", "error"} for r in results)
    if args.fail_on_mismatch and has_problem:
        sys.exit(1)


if __name__ == "__main__":
    main()
