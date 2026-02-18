#!/usr/bin/env python3
"""
bonica_rid_counts.py

Count unique bonica.rid values by year from DIME parquet files.

Defaults:
- Scope: state-leg candidates only (seat in state:upper/state:lower, recipient.type='CAND')
- Output: prints table and optional CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb
import pandas as pd


def find_year_parquets(base_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in sorted(base_dir.glob("*_parquet/*_candidate_donor.parquet")):
        token = p.stem.split("_", 1)[0]
        if token.isdigit():
            out.append((int(token), p))
    return out


def build_where(scope: str) -> str:
    if scope == "all":
        return ""
    return (
        " WHERE seat IN ('state:upper', 'state:lower')"
        "   AND \"recipient.type\" = 'CAND'"
    )


def run(scope: str) -> tuple[pd.DataFrame, int]:
    year_files = find_year_parquets(Path("DIME data"))
    if not year_files:
        raise FileNotFoundError("No parquet files found in DIME data/*_parquet/")

    where = build_where(scope)
    con = duckdb.connect()
    try:
        rows: list[dict] = []
        union_sql_parts: list[str] = []

        for year, parquet in year_files:
            q = (
                "SELECT COUNT(DISTINCT \"bonica.rid\") "
                "FROM read_parquet(?)"
                + where
            )
            per_year = int(con.execute(q, [str(parquet)]).fetchone()[0] or 0)
            rows.append(
                {
                    "year": year,
                    "unique_bonica_rid": per_year,
                    "parquet": str(parquet),
                    "scope": scope,
                }
            )
            union_sql_parts.append(
                f"SELECT DISTINCT \"bonica.rid\" AS rid FROM read_parquet('{str(parquet)}'){where}"
            )

        union_sql = "SELECT COUNT(*) FROM (" + " UNION ".join(union_sql_parts) + ")"
        global_unique = int(con.execute(union_sql).fetchone()[0] or 0)
    finally:
        con.close()

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return df, global_unique


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Count unique bonica.rid by year.")
    p.add_argument(
        "--scope",
        choices=["stateleg", "all"],
        default="stateleg",
        help="stateleg=state upper/lower CAND only; all=entire parquet",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df, global_unique = run(args.scope)

    print(df[["year", "unique_bonica_rid"]].to_string(index=False))
    print(f"\nGlobal unique bonica.rid across all listed years ({args.scope}): {global_unique:,}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote CSV: {args.out}")


if __name__ == "__main__":
    main()

