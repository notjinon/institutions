#!/usr/bin/env python3
"""
source_integrity_checks.py

Three integrity checks between DIME CSV and DIME Parquet for each year:
1) Critical-column null count comparison
2) Distinct candidate count comparison by (cycle, recipient.state, seat)
3) Random row sample exact-match check (selected columns)

Usage:
  python source_integrity_checks.py 2020
  python source_integrity_checks.py 2000 2020 --sample-size 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from paths import DIME_DATA_DIR, DIME_PARQUET_FILE

# Keep this aligned with convert_year.py type overrides.
FORCED_TYPES = {
    "contributor.zipcode": "VARCHAR",
    "recipient.party": "VARCHAR",
}

CRITICAL_COLUMNS = [
    "bonica.rid",
    "bonica.cid",
    "cycle",
    "recipient.state",
    "recipient.party",
    "recipient.type",
    "seat",
    "contributor.zipcode",
]

SAMPLE_COMPARE_COLUMNS = [
    "bonica.rid",
    "bonica.cid",
    "cycle",
    "recipient.name",
    "recipient.state",
    "recipient.party",
    "recipient.type",
    "seat",
    "contributor.zipcode",
]


def _quote_ident(col: str) -> str:
    return f'"{col.replace('"', '""')}"'


def _csv_read_sql(csv_path: Path) -> str:
    type_pairs = ", ".join([f"'{k}': '{v}'" for k, v in FORCED_TYPES.items()])
    return (
        "read_csv_auto("
        f"'{str(csv_path.resolve()).replace('\\', '/')}', "
        "delim=',', header=true, sample_size=200000, ignore_errors=true, "
        f"types={{{type_pairs}}}"
        ")"
    )


def _register_views(con: duckdb.DuckDBPyConnection, csv_path: Path, parquet_path: Path) -> None:
    con.execute(f"CREATE OR REPLACE VIEW csv_src AS SELECT * FROM {_csv_read_sql(csv_path)}")
    pq = str(parquet_path.resolve()).replace('\\', '/')
    con.execute(f"CREATE OR REPLACE VIEW parquet_src AS SELECT * FROM read_parquet('{pq}')")


def check_null_counts(con: duckdb.DuckDBPyConnection) -> list[dict]:
    out = []
    for col in CRITICAL_COLUMNS:
        qc = _quote_ident(col)
        csv_nulls = con.execute(
            f"SELECT COUNT(*) FROM csv_src WHERE {qc} IS NULL OR TRIM(CAST({qc} AS VARCHAR)) = ''"
        ).fetchone()[0]
        pq_nulls = con.execute(
            f"SELECT COUNT(*) FROM parquet_src WHERE {qc} IS NULL OR TRIM(CAST({qc} AS VARCHAR)) = ''"
        ).fetchone()[0]
        out.append(
            {
                "column": col,
                "csv_nulls": int(csv_nulls),
                "parquet_nulls": int(pq_nulls),
                "diff": int(pq_nulls) - int(csv_nulls),
            }
        )
    return out


def check_grouped_candidate_counts(con: duckdb.DuckDBPyConnection, limit: int = 30):
    sql = f"""
    WITH csv_g AS (
      SELECT
        cycle,
        "recipient.state" AS state,
        seat,
        COUNT(DISTINCT "bonica.rid") AS cands
      FROM csv_src
      WHERE seat IN ('state:upper', 'state:lower')
        AND "recipient.type" = 'CAND'
      GROUP BY cycle, "recipient.state", seat
    ),
    pq_g AS (
      SELECT
        cycle,
        "recipient.state" AS state,
        seat,
        COUNT(DISTINCT "bonica.rid") AS cands
      FROM parquet_src
      WHERE seat IN ('state:upper', 'state:lower')
        AND "recipient.type" = 'CAND'
      GROUP BY cycle, "recipient.state", seat
    )
    SELECT
      COALESCE(c.cycle, p.cycle) AS cycle,
      COALESCE(c.state, p.state) AS state,
      COALESCE(c.seat, p.seat)   AS seat,
      c.cands AS csv_cands,
      p.cands AS parquet_cands,
      COALESCE(p.cands, 0) - COALESCE(c.cands, 0) AS diff
    FROM csv_g c
    FULL OUTER JOIN pq_g p
      ON c.cycle = p.cycle
     AND c.state = p.state
     AND c.seat = p.seat
    WHERE COALESCE(c.cands, -1) <> COALESCE(p.cands, -1)
    ORDER BY ABS(COALESCE(p.cands, 0) - COALESCE(c.cands, 0)) DESC, cycle, state, seat
    LIMIT {int(limit)}
    """
    return con.execute(sql).fetchall()


def check_random_sample_match(con: duckdb.DuckDBPyConnection, sample_size: int = 100) -> tuple[int, int, int]:
    cols_q = ", ".join([_quote_ident(c) for c in SAMPLE_COMPARE_COLUMNS])
    join_cond = " AND ".join([
        f"c.{_quote_ident(c)} IS NOT DISTINCT FROM s.{_quote_ident(c)}" for c in SAMPLE_COMPARE_COLUMNS
    ])

    sql = f"""
    WITH sample AS (
      SELECT {cols_q}
      FROM parquet_src
      USING SAMPLE RESERVOIR({int(sample_size)} ROWS)
    ),
    matched AS (
      SELECT
        s.*,
        COUNT(c."bonica.rid") AS match_count
      FROM sample s
      LEFT JOIN csv_src c
        ON {join_cond}
      GROUP BY ALL
    )
    SELECT
      COUNT(*) AS sampled,
      SUM(CASE WHEN match_count > 0 THEN 1 ELSE 0 END) AS matched,
      SUM(CASE WHEN match_count = 0 THEN 1 ELSE 0 END) AS unmatched
    FROM matched
    """
    sampled, matched, unmatched = con.execute(sql).fetchone()
    return int(sampled), int(matched), int(unmatched)


def run_year(year: int, sample_size: int, mismatch_limit: int) -> None:
    csv_path = Path(DIME_DATA_DIR(year))
    parquet_path = Path(DIME_PARQUET_FILE(year))

    print("\n" + "=" * 88)
    print(f"YEAR {year}")
    print("=" * 88)

    if not csv_path.exists():
        print(f"ERROR: missing CSV: {csv_path}")
        return
    if not parquet_path.exists():
        print(f"ERROR: missing Parquet: {parquet_path}")
        return

    con = duckdb.connect()
    try:
        _register_views(con, csv_path, parquet_path)

        print("\n[1] Critical-column NULL count comparison")
        print(f"{'COLUMN':<24} {'CSV_NULLS':>14} {'PARQUET_NULLS':>16} {'DIFF(P-C)':>12}")
        print("-" * 70)
        null_rows = check_null_counts(con)
        for r in null_rows:
            print(f"{r['column']:<24} {r['csv_nulls']:>14,} {r['parquet_nulls']:>16,} {r['diff']:>+12,}")

        print("\n[2] Distinct candidates by (cycle, state, seat) mismatches")
        print(f"{'CYCLE':>6} {'STATE':>6} {'SEAT':<12} {'CSV':>8} {'PARQUET':>10} {'DIFF':>8}")
        print("-" * 60)
        mismatches = check_grouped_candidate_counts(con, limit=mismatch_limit)
        if not mismatches:
            print("No mismatches found.")
        else:
            for cycle, state, seat, csv_c, pq_c, diff in mismatches:
                csv_show = 0 if csv_c is None else int(csv_c)
                pq_show = 0 if pq_c is None else int(pq_c)
                print(f"{int(cycle):>6} {str(state):>6} {str(seat):<12} {csv_show:>8,} {pq_show:>10,} {int(diff):>+8,}")

        print("\n[3] Random row sample exact-match check")
        sampled, matched, unmatched = check_random_sample_match(con, sample_size=sample_size)
        match_pct = (matched / sampled * 100.0) if sampled else 0.0
        print(f"Sampled rows:  {sampled:,}")
        print(f"Matched rows:  {matched:,}")
        print(f"Unmatched:     {unmatched:,}")
        print(f"Match rate:    {match_pct:.1f}%")

    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3 CSV vs Parquet integrity checks for DIME data by year."
    )
    parser.add_argument("years", nargs="+", type=int, help="Year(s), e.g. 2000 2020")
    parser.add_argument("--sample-size", type=int, default=100, help="Rows to sample for check #3")
    parser.add_argument("--mismatch-limit", type=int, default=30, help="Max mismatch rows to print in check #2")
    args = parser.parse_args()

    for y in args.years:
        run_year(y, sample_size=args.sample_size, mismatch_limit=args.mismatch_limit)


if __name__ == "__main__":
    main()
