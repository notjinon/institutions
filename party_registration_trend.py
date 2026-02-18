#!/usr/bin/env python3
"""
party_registration_trend.py

Track stated party coding in DIME state-leg CAND records by year.

Buckets:
- DEM: recipient.party == '100'
- REP: recipient.party == '200'
- 3PT: recipient.party is non-null and not 100/200
- NUL: recipient.party is NULL

Default behavior:
- Scans all available DIME parquet files under "DIME data/*_parquet/*.parquet"
- Uses row-based counts (donation rows)
- Writes CSV to outputs/party_registration_trend_rows.csv

Optional:
- --basis candidates  -> uses distinct bonica.rid counts instead of rows
- --out <path>        -> choose output file
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb
import pandas as pd


def find_year_parquets(base_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for p in sorted(base_dir.glob("*_parquet/*_candidate_donor.parquet")):
        stem = p.stem
        year_token = stem.split("_", 1)[0]
        if year_token.isdigit():
            pairs.append((int(year_token), p))
    return pairs


def build_query(basis: str) -> str:
    if basis == "rows":
        value_expr = "COUNT(*)"
    else:
        value_expr = "COUNT(DISTINCT \"bonica.rid\")"

    return f"""
    WITH b AS (
      SELECT *
      FROM read_parquet(?)
      WHERE cycle = ?
        AND seat IN ('state:upper', 'state:lower')
        AND "recipient.type" = 'CAND'
    ),
    agg AS (
      SELECT
        {value_expr} AS total,
        {value_expr} FILTER (WHERE CAST("recipient.party" AS VARCHAR) = '100') AS dem,
        {value_expr} FILTER (WHERE CAST("recipient.party" AS VARCHAR) = '200') AS rep,
        {value_expr} FILTER (
          WHERE "recipient.party" IS NOT NULL
            AND CAST("recipient.party" AS VARCHAR) NOT IN ('100', '200')
        ) AS third_party,
        {value_expr} FILTER (WHERE "recipient.party" IS NULL) AS nul,
        COUNT(DISTINCT "recipient.state") AS state_count,
        COUNT(*) AS stateleg_rows
      FROM b
    )
    SELECT * FROM agg
    """


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def run(basis: str, out_path: Path) -> pd.DataFrame:
    year_files = find_year_parquets(Path("DIME data"))
    if not year_files:
        raise FileNotFoundError("No parquet files found in DIME data/*_parquet/")

    con = duckdb.connect()
    try:
        q = build_query(basis)
        rows: list[dict] = []

        for year, parquet_path in year_files:
            total, dem, rep, third_party, nul, state_count, stateleg_rows = con.execute(
                q, [str(parquet_path), year]
            ).fetchone()

            total = int(total or 0)
            dem = int(dem or 0)
            rep = int(rep or 0)
            third_party = int(third_party or 0)
            nul = int(nul or 0)
            state_count = int(state_count or 0)
            stateleg_rows = int(stateleg_rows or 0)

            rows.append(
                {
                    "year": year,
                    "basis": basis,
                    "total": total,
                    "DEM": dem,
                    "REP": rep,
                    "3PT": third_party,
                    "NUL": nul,
                    "DEM_pct": round(pct(dem, total), 2),
                    "REP_pct": round(pct(rep, total), 2),
                    "3PT_pct": round(pct(third_party, total), 2),
                    "NUL_pct": round(pct(nul, total), 2),
                    "state_count": state_count,
                    "stateleg_rows": stateleg_rows,
                    "parquet": str(parquet_path),
                }
            )
    finally:
        con.close()

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track DIME party coding by year (DEM/REP/3PT/NUL).")
    p.add_argument(
        "--basis",
        choices=["rows", "candidates"],
        default="rows",
        help="rows = donation rows; candidates = distinct bonica.rid",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: outputs/party_registration_trend_<basis>.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or Path("outputs") / f"party_registration_trend_{args.basis}.csv"
    df = run(args.basis, out)

    print(f"Wrote {len(df)} years -> {out}")
    print(df[["year", "total", "DEM", "REP", "3PT", "NUL", "DEM_pct", "REP_pct", "3PT_pct", "NUL_pct", "state_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
