#!/usr/bin/env python3
"""
nul_combo_hotspots.py

Show where NULL-party DIME records are concentrated for state-leg candidates.

Interpretation:
- A (state, year) with any NULL-party records implies both party runs (DEM + REP)
  can be affected in cspy-match3, because NULL-party candidates are loaded into
  both runs before downstream matching.
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


def run(basis: str, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = find_year_parquets(Path("DIME data"))
    if not files:
        raise FileNotFoundError("No parquet files found in DIME data/*_parquet/")

    if basis == "rows":
        total_expr = "COUNT(*)"
        nul_expr = "SUM(CASE WHEN \"recipient.party\" IS NULL THEN 1 ELSE 0 END)"
    else:
        total_expr = "COUNT(DISTINCT \"bonica.rid\")"
        nul_expr = (
            "COUNT(DISTINCT CASE WHEN \"recipient.party\" IS NULL "
            "THEN \"bonica.rid\" ELSE NULL END)"
        )

    con = duckdb.connect()
    try:
        detail_parts: list[pd.DataFrame] = []
        for year, parquet in files:
            q = f"""
            SELECT
              {year} AS year,
              "recipient.state" AS state,
              {total_expr} AS total_count,
              {nul_expr} AS nul_count
            FROM read_parquet(?)
            WHERE cycle = ?
              AND seat IN ('state:upper', 'state:lower')
              AND "recipient.type" = 'CAND'
              AND "recipient.state" IS NOT NULL
            GROUP BY "recipient.state"
            """
            df = con.execute(q, [str(parquet), year]).df()
            detail_parts.append(df)
    finally:
        con.close()

    detail = pd.concat(detail_parts, ignore_index=True)
    detail["total_count"] = detail["total_count"].fillna(0).astype(int)
    detail["nul_count"] = detail["nul_count"].fillna(0).astype(int)
    detail["nul_pct"] = (detail["nul_count"] / detail["total_count"]).replace([pd.NA], 0).fillna(0) * 100
    detail["nul_pct"] = detail["nul_pct"].round(2)
    detail["DEM_combo_affected"] = (detail["nul_count"] > 0).astype(int)
    detail["REP_combo_affected"] = (detail["nul_count"] > 0).astype(int)
    detail = detail.sort_values(["nul_count", "nul_pct"], ascending=[False, False]).reset_index(drop=True)

    summary = (
        detail.groupby("state", as_index=False)
        .agg(
            nul_total=("nul_count", "sum"),
            total_total=("total_count", "sum"),
            years_with_nul=("nul_count", lambda s: int((s > 0).sum())),
            dem_combo_count=("DEM_combo_affected", "sum"),
            rep_combo_count=("REP_combo_affected", "sum"),
        )
    )
    summary["nul_pct_overall"] = (summary["nul_total"] / summary["total_total"]).fillna(0) * 100
    summary["nul_pct_overall"] = summary["nul_pct_overall"].round(2)
    summary = summary.sort_values(["nul_total", "years_with_nul"], ascending=[False, False]).reset_index(drop=True)

    return detail.head(top_n), summary.head(top_n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank states/party-combos by NULL-party concentration.")
    p.add_argument(
        "--basis",
        choices=["rows", "candidates"],
        default="candidates",
        help="rows = donation rows; candidates = distinct bonica.rid",
    )
    p.add_argument("--top", type=int, default=20, help="Top N rows to print/save")
    p.add_argument("--out-detail", type=Path, default=None, help="Optional detail CSV output")
    p.add_argument("--out-summary", type=Path, default=None, help="Optional summary CSV output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detail, summary = run(args.basis, args.top)

    print("Top state-year NUL hotspots:")
    print(detail.to_string(index=False))
    print("\nTop states by NUL burden and affected party combos:")
    print(summary.to_string(index=False))

    if args.out_detail:
        args.out_detail.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.out_detail, index=False)
        print(f"\nWrote detail: {args.out_detail}")
    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_summary, index=False)
        print(f"Wrote summary: {args.out_summary}")


if __name__ == "__main__":
    main()

