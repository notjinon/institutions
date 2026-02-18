#!/usr/bin/env python3
"""
source_compare.py

For a given set of (state, year) pairs, pulls candidate statistics from both
data sources independently and prints a side-by-side comparison:

  SOURCE A — DIME parquet  (recipient-level donation rows)
  SOURCE B — NIMSP upperlower CSVs  (election-level candidate rows)

This helps diagnose why cspy-match3 produces zero (or few) candidates for
combinations like WA-REP-2020: you can immediately see whether the problem
is on the DIME side, the NIMSP side, or both.

Usage:
    python source_compare.py
    # prompted for states and years, e.g.  WA, CA   and  2020, 2022
"""

import re
import sys
import duckdb
import pandas as pd
from pathlib import Path
from paths import (
    DIME_PARQUET_FILE,
    UPPER_HOUSE_FILE,
    LOWER_HOUSE_FILE,
)

W    = 70
SEP  = "=" * W
SEP2 = "-" * W
PARTIES = ["DEM", "REP"]
PARTY_CODE = {"DEM": "100", "REP": "200"}


# ─────────────────────────────────────────────────────────────────────────────
# DIME parquet source
# ─────────────────────────────────────────────────────────────────────────────

def dime_stats(state: str, year: int) -> dict:
    """
    Query the DIME parquet for state/year and return per-party candidate stats.
    No party filter is applied here — we report all party codes so missing
    data is immediately visible.
    """
    parquet_file = DIME_PARQUET_FILE(year)
    if not Path(parquet_file).exists():
        return {"error": f"Parquet not found: {parquet_file}"}

    con = duckdb.connect()
    f   = str(parquet_file)

    try:
        # ── Total rows in parquet ────────────────────────────────────────────
        total_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [f]
        ).fetchone()[0]

        # ── State-leg CAND rows, broken down by party code ────────────────
        party_df = con.execute(
            "SELECT \"recipient.party\" AS party_code,"
            "        COUNT(DISTINCT \"bonica.rid\") AS unique_candidates,"
            "        COUNT(DISTINCT \"bonica.cid\") AS unique_donors,"
            "        COUNT(*) AS donation_rows,"
            "        seat"
            " FROM read_parquet(?)"
            " WHERE \"recipient.state\" = ?"
            "   AND cycle = ?"
            "   AND seat IN ('state:upper', 'state:lower')"
            "   AND \"recipient.type\" = 'CAND'"
            " GROUP BY \"recipient.party\", seat"
            " ORDER BY seat, party_code",
            [f, state, year]
        ).df()

        # ── Candidate list (unique bonica.rid) ────────────────────────────
        cand_df = con.execute(
            "SELECT \"bonica.rid\","
            "        ANY_VALUE(\"recipient.name\")  AS name,"
            "        ANY_VALUE(\"recipient.party\") AS party_code,"
            "        ANY_VALUE(seat)                AS seat,"
            "        COUNT(*)                       AS donation_rows"
            " FROM read_parquet(?)"
            " WHERE \"recipient.state\" = ?"
            "   AND cycle = ?"
            "   AND seat IN ('state:upper', 'state:lower')"
            "   AND \"recipient.type\" = 'CAND'"
            " GROUP BY \"bonica.rid\""
            " ORDER BY seat, party_code, name",
            [f, state, year]
        ).df()

        # ── Explicit 100/200 vs NULL breakdown ───────────────────────────
        null_count = int(cand_df['party_code'].isna().sum()) if len(cand_df) else 0
        dem_count  = int((cand_df['party_code'] == '100').sum()) if len(cand_df) else 0
        rep_count  = int((cand_df['party_code'] == '200').sum()) if len(cand_df) else 0
        other_count = len(cand_df) - null_count - dem_count - rep_count

        return {
            "total_parquet_rows": total_rows,
            "party_seat_table":   party_df,
            "candidate_list":     cand_df,
            "null_party_cands":   null_count,
            "dem_party_cands":    dem_count,
            "rep_party_cands":    rep_count,
            "other_party_cands":  other_count,
            "total_unique_cands": len(cand_df),
        }
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# NIMSP upperlower source
# ─────────────────────────────────────────────────────────────────────────────

def _load_upperlower(year: int) -> pd.DataFrame:
    """Load and combine upper + lower CSV files for a year."""
    party_map = {"DEMOCRATIC": "DEM", "REPUBLICAN": "REP"}
    frames = []
    for path_fn, house in [(UPPER_HOUSE_FILE, "U"), (LOWER_HOUSE_FILE, "L")]:
        p = Path(str(path_fn(year)))
        if not p.exists():
            alt = p.with_suffix(".csv.csv")
            if alt.exists():
                p = alt
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        df["_house"]   = house
        df["_state"]   = df["Election_Jurisdiction"].astype(str).str.upper().str.strip()
        df["_party"]   = df["General_Party"].astype(str).str.upper().str.strip().map(party_map)
        df["_name"]    = df["Candidate"].astype(str).str.strip()
        df["_district"]= df["Office_Sought"].astype(str).str.strip()
        df["_status"]  = df["Election_Status"].astype(str).str.strip()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def nimsp_stats(state: str, year: int) -> dict:
    """
    Load NIMSP upperlower CSVs for state/year and return per-party stats.
    """
    all_df = _load_upperlower(year)
    if all_df.empty:
        return {"error": f"No upperlower CSV files found for year {year}"}

    st_df = all_df[all_df["_state"] == state.upper()].copy()
    if st_df.empty:
        return {
            "error": None,
            "total_rows": 0,
            "state_rows": 0,
            "party_house_table": pd.DataFrame(),
            "candidate_list":    pd.DataFrame(),
        }

    by_party_house = (
        st_df.groupby(["_party", "_house"])
        .agg(
            candidates = ("_name", "count"),
            dem_wins   = ("_status", lambda x: (x.str.upper().str.startswith("WON")).sum()),
        )
        .reset_index()
        .rename(columns={"_party": "party", "_house": "house"})
    )

    cand_list = st_df[st_df["_party"].isin(["DEM", "REP"])][
        ["_party", "_house", "_name", "_district", "_status"]
    ].rename(columns={
        "_party": "party", "_house": "house",
        "_name": "name", "_district": "district", "_status": "status",
    }).sort_values(["party", "house", "district"])

    return {
        "error":            None,
        "total_csv_rows":   len(all_df),
        "state_rows":       len(st_df),
        "party_house_table": by_party_house,
        "candidate_list":    cand_list,
        "dem_count":        int((st_df["_party"] == "DEM").sum()),
        "rep_count":        int((st_df["_party"] == "REP").sum()),
        "unknown_count":    int(st_df["_party"].isna().sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_dime_section(state: str, year: int, d: dict):
    if "error" in d:
        print(f"  [DIME] ERROR: {d['error']}")
        return

    print(f"  Total rows in parquet       : {d['total_parquet_rows']:>12,}")
    print(f"  Unique state-leg candidates : {d['total_unique_cands']:>12,}")
    print(f"    — party code 100 (DEM)    : {d['dem_party_cands']:>12,}")
    print(f"    — party code 200 (REP)    : {d['rep_party_cands']:>12,}")
    print(f"    — other codes             : {d['other_party_cands']:>12,}")
    print(f"    — NULL / missing party    : {d['null_party_cands']:>12,}  ← main failure mode in 2018+")

    tbl = d["party_seat_table"]
    if not tbl.empty:
        print(f"\n  Breakdown by (party code, seat):")
        print(f"    {'PARTY':>6}  {'SEAT':<14}  {'CANDS':>6}  {'DONORS':>8}  {'ROWS':>8}")
        print(f"    {'-'*50}")
        for _, row in tbl.iterrows():
            pc = str(row['party_code']) if pd.notna(row['party_code']) else "NULL"
            label = {'100':'DEM','200':'REP'}.get(pc, pc)
            print(f"    {label:>6}  {row['seat']:<14}  {int(row['unique_candidates']):>6}  "
                  f"{int(row['unique_donors']):>8}  {int(row['donation_rows']):>8}")

    cands = d["candidate_list"]
    if not cands.empty:
        print(f"\n  Candidate list  ({len(cands)} unique bonica.rid):")
        print(f"    {'NAME':<30}  {'PARTY':>6}  {'SEAT':<14}  {'ROWS':>5}")
        print(f"    {'-'*60}")
        for _, row in cands.iterrows():
            pc    = str(row['party_code']) if pd.notna(row['party_code']) else "NULL"
            label = {'100':'DEM','200':'REP'}.get(pc, pc)
            print(f"    {str(row['name']):<30}  {label:>6}  {row['seat']:<14}  {int(row['donation_rows']):>5}")
    else:
        print(f"\n  [DIME] No state-leg CAND records found for {state}-{year}.")


def print_nimsp_section(state: str, year: int, d: dict):
    if d.get("error"):
        print(f"  [NIMSP] ERROR: {d['error']}")
        return
    if d["state_rows"] == 0:
        print(f"  [NIMSP] No rows found for state={state} in {year} upperlower files.")
        return

    print(f"  Rows for {state} in upperlower CSVs : {d['state_rows']:>6,}")
    print(f"    — DEM candidates  : {d['dem_count']:>6,}")
    print(f"    — REP candidates  : {d['rep_count']:>6,}")
    print(f"    — unknown party   : {d['unknown_count']:>6,}")

    tbl = d["party_house_table"]
    if not tbl.empty:
        print(f"\n  Breakdown by (party, house):")
        print(f"    {'PARTY':>6}  {'HOUSE':>6}  {'CANDS':>6}  {'WINS':>5}")
        print(f"    {'-'*30}")
        for _, row in tbl.iterrows():
            party = str(row['party']) if pd.notna(row['party']) else "OTHER"
            print(f"    {party:>6}  {row['house']:>6}  {int(row['candidates']):>6}  {int(row['dem_wins']):>5}")

    cands = d["candidate_list"]
    if not cands.empty:
        print(f"\n  Candidate list  ({len(cands)} DEM+REP candidates):")
        print(f"    {'NAME':<32}  {'P':>3}  {'H':>2}  {'DISTRICT':<24}  STATUS")
        print(f"    {'-'*80}")
        for _, row in cands.iterrows():
            print(f"    {str(row['name']):<32}  {str(row['party']):>3}  {row['house']:>2}"
                  f"  {str(row['district']):<24}  {row['status']}")


def print_comparison(state: str, year: int):
    print(f"\n{SEP}")
    print(f"  {state}  —  {year}")
    print(SEP)

    dime  = dime_stats(state, year)
    nimsp = nimsp_stats(state, year)

    # ── DIME ──────────────────────────────────────────────────────────────
    print(f"\n  ┌─ SOURCE A: DIME parquet ({year}_candidate_donor.parquet)")
    print(f"  │")
    for line in _capture(print_dime_section, state, year, dime):
        print(f"  │  {line}")
    print(f"  └{'─'*(W-3)}")

    # ── NIMSP ─────────────────────────────────────────────────────────────
    print(f"\n  ┌─ SOURCE B: NIMSP upperlower CSVs ({year}_upper / {year}_lower)")
    print(f"  │")
    for line in _capture(print_nimsp_section, state, year, nimsp):
        print(f"  │  {line}")
    print(f"  └{'─'*(W-3)}")

    # ── Quick mismatch flags ───────────────────────────────────────────────
    print(f"\n  MISMATCH FLAGS:")
    dime_total  = dime.get("total_unique_cands", 0)  if "error" not in dime  else "ERR"
    nimsp_total = (nimsp.get("dem_count", 0) + nimsp.get("rep_count", 0)) if not nimsp.get("error") else "ERR"

    if dime_total == "ERR" or nimsp_total == "ERR":
        print(f"    Cannot compare — one or both sources errored.")
    else:
        diff = dime_total - nimsp_total
        flag = "⚠  LARGE GAP" if abs(diff) > 10 else ("✓  roughly aligned" if abs(diff) <= 3 else "△  minor gap")
        print(f"    DIME unique cands : {dime_total:>5}")
        print(f"    NIMSP DEM+REP     : {nimsp_total:>5}")
        print(f"    Difference        : {diff:>+5}  {flag}")
        if isinstance(dime_total, int) and dime.get("null_party_cands", 0) > 0:
            pct_null = dime["null_party_cands"] / dime_total * 100 if dime_total else 0
            print(f"    NULL-party cands  : {dime['null_party_cands']:>5}  ({pct_null:.0f}% of DIME cands)"
                  f"  ← will be picked up by cspy-match3")


def _capture(fn, *args):
    """Capture stdout lines from a print function into a list."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue().splitlines()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_states = input("States (comma-separated, e.g. WA, CA, TX): ").strip()
    raw_years  = input("Years  (comma-separated, e.g. 2018, 2020, 2022): ").strip()

    states = [s.strip().upper() for s in raw_states.split(",") if s.strip()]
    years  = [int(y.strip()) for y in raw_years.split(",") if y.strip()]

    if not states or not years:
        print("No states or years provided. Exiting.")
        sys.exit(1)

    print(f"\n{SEP}")
    print(f"  SOURCE COMPARISON — states: {states}  years: {years}")
    print(SEP)

    for year in years:
        for state in states:
            print_comparison(state, year)

    print(f"\n{SEP}\n")
