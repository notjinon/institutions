#!/usr/bin/env python3
"""
nul_recipients.py

List the most common NULL-party recipients in DIME for a given year.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from utils.paths import DIME_PARQUET_FILE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show top NULL-party recipients for a year.")
    p.add_argument("--year", type=int, required=True, help="Election cycle year (e.g., 2020)")
    p.add_argument("--top", type=int, default=30, help="Number of recipients to show")
    p.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    return p.parse_args()


def run(year: int, top_n: int) -> tuple[pd.DataFrame, dict]:
    parquet_file = DIME_PARQUET_FILE(year)
    if not Path(parquet_file).exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_file}")

    con = duckdb.connect()
    try:
        summary = con.execute(
            """
            SELECT
              COUNT(*) AS stateleg_rows,
              SUM(CASE WHEN "recipient.party" IS NULL THEN 1 ELSE 0 END) AS null_rows,
              COUNT(DISTINCT "bonica.rid") AS stateleg_unique_rids,
              COUNT(DISTINCT CASE WHEN "recipient.party" IS NULL THEN "bonica.rid" END) AS null_unique_rids
            FROM read_parquet(?)
            WHERE cycle = ?
              AND seat IN ('state:upper', 'state:lower')
              AND "recipient.type" = 'CAND'
            """,
            [str(parquet_file), year],
        ).fetchone()

        stateleg_rows, null_rows, stateleg_unique_rids, null_unique_rids = [int(x or 0) for x in summary]

        top_df = con.execute(
            """
            SELECT
              "bonica.rid" AS rid,
              ANY_VALUE("recipient.name") AS recipient_name,
              ANY_VALUE("recipient.state") AS state,
              ANY_VALUE(seat) AS seat,
              COUNT(*) AS donation_rows,
              COUNT(DISTINCT "bonica.cid") AS unique_donors,
              ROUND(SUM(COALESCE(amount, 0)), 2) AS total_amount,
              MIN(date) AS first_date,
              MAX(date) AS last_date
            FROM read_parquet(?)
            WHERE cycle = ?
              AND seat IN ('state:upper', 'state:lower')
              AND "recipient.type" = 'CAND'
              AND "recipient.party" IS NULL
            GROUP BY "bonica.rid"
            ORDER BY donation_rows DESC, unique_donors DESC, total_amount DESC
            LIMIT ?
            """,
            [str(parquet_file), year, top_n],
        ).df()
    finally:
        con.close()

    stats = {
        "year": year,
        "stateleg_rows": stateleg_rows,
        "null_rows": null_rows,
        "null_rows_pct": (null_rows / stateleg_rows * 100.0) if stateleg_rows else 0.0,
        "stateleg_unique_rids": stateleg_unique_rids,
        "null_unique_rids": null_unique_rids,
        "null_unique_rids_pct": (null_unique_rids / stateleg_unique_rids * 100.0) if stateleg_unique_rids else 0.0,
    }
    return top_df, stats


def main() -> None:
    args = parse_args()
    top_df, stats = run(args.year, args.top)

    print(f"Year: {stats['year']}")
    print(
        f"State-leg CAND rows: {stats['stateleg_rows']:,} | NULL-party rows: {stats['null_rows']:,} "
        f"({stats['null_rows_pct']:.2f}%)"
    )
    print(
        f"State-leg unique recipients: {stats['stateleg_unique_rids']:,} | NULL-party recipients: {stats['null_unique_rids']:,} "
        f"({stats['null_unique_rids_pct']:.2f}%)"
    )
    print()
    print(top_df.to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        top_df.to_csv(args.out, index=False)
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()

