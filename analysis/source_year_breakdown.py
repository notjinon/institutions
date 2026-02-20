#!/usr/bin/env python3
"""
source_year_breakdown.py

Interactive subset explorer for NIMSP or DIME candidate sources.

Prompts:
1) Source: N or D
2) Year
3) House: U or L
4) Party: D, R, A (all), or N (not D/R)
5) State (2-letter code)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import duckdb
import pandas as pd

from utils.paths import DIME_PARQUET_FILE, LOWER_HOUSE_FILE, UPPER_HOUSE_FILE


def ask_choice(prompt: str, valid: set[str]) -> str:
    while True:
        value = input(prompt).strip().upper()
        if value in valid:
            return value
        print(f"Invalid input. Choose one of: {', '.join(sorted(valid))}")


def ask_year(prompt: str) -> int:
    while True:
        value = input(prompt).strip()
        if value.isdigit():
            return int(value)
        print("Invalid year. Enter digits only (example: 2020).")


def _pick_nimsp_file(year: int, house: str) -> Path:
    path = Path(str(UPPER_HOUSE_FILE(year) if house == "U" else LOWER_HOUSE_FILE(year)))
    if not path.exists():
        alt = path.with_suffix(".csv.csv")
        if alt.exists():
            path = alt
    return path


def list_dime_candidates(year: int, house: str, party: str, state: str) -> pd.DataFrame:
    parquet_file = DIME_PARQUET_FILE(year)
    if not Path(parquet_file).exists():
        raise FileNotFoundError(f"DIME parquet not found: {parquet_file}")

    seat = "state:upper" if house == "U" else "state:lower"
    con = duckdb.connect()
    try:
        where_party = ""
        params = [str(parquet_file), year, state, seat]
        if party == "D":
            where_party = " AND CAST(\"recipient.party\" AS VARCHAR) = ?"
            params.append("100")
        elif party == "R":
            where_party = " AND CAST(\"recipient.party\" AS VARCHAR) = ?"
            params.append("200")
        elif party == "N":
            where_party = (
                " AND (CAST(\"recipient.party\" AS VARCHAR) NOT IN ('100', '200')"
                "      OR \"recipient.party\" IS NULL)"
            )

        query = (
            "SELECT \"bonica.rid\" AS rid,"
            "       ANY_VALUE(\"recipient.name\")  AS name,"
            "       ANY_VALUE(\"recipient.party\") AS party_code,"
            "       ANY_VALUE(seat)                AS seat,"
            "       COUNT(*)                       AS donation_rows,"
            "       COUNT(DISTINCT \"bonica.cid\") AS unique_donors"
            " FROM read_parquet(?)"
            " WHERE cycle = ?"
            "   AND \"recipient.state\" = ?"
            "   AND seat = ?"
            "   AND \"recipient.type\" = 'CAND'"
            f"{where_party}"
            " GROUP BY \"bonica.rid\""
            " ORDER BY name"
        )
        df = con.execute(query, params).df()
    finally:
        con.close()
    return df


def _normalize_nimsp_party(value: str) -> str:
    upper = value.strip().upper()
    if upper.startswith("DEM"):
        return "D"
    if upper.startswith("REP"):
        return "R"
    return ""


def list_nimsp_candidates(year: int, house: str, party: str, state: str) -> pd.DataFrame:
    csv_path = _pick_nimsp_file(year, house)
    if not csv_path.exists():
        raise FileNotFoundError(f"NIMSP file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    state_series = df["Election_Jurisdiction"].astype(str).str.upper().str.strip()
    party_series = df["General_Party"].astype(str).map(_normalize_nimsp_party)

    subset = df[state_series == state].copy()
    if party in {"D", "R"}:
        subset = subset[party_series[state_series == state] == party]
    elif party == "N":
        subset = subset[party_series[state_series == state] == ""]

    if subset.empty:
        return pd.DataFrame(columns=["name", "party", "raw_party", "house", "district", "status"])

    subset_party = subset["General_Party"].astype(str).map(_normalize_nimsp_party).replace("", "N")

    result = subset.assign(
        name=subset["Candidate"].astype(str).str.strip(),
        district=subset["Office_Sought"].astype(str).str.strip(),
        status=subset["Election_Status"].astype(str).str.strip(),
        party=subset_party,
        raw_party=subset["General_Party"].astype(str).str.strip(),
        house=house,
    )[["name", "party", "raw_party", "house", "district", "status"]].sort_values(["district", "name"])

    return result.reset_index(drop=True)


def main() -> None:
    print("SOURCE YEAR BREAKDOWN")
    source = ask_choice("Choose source (N/D): ", {"N", "D"})
    year = ask_year("Year: ")
    house = ask_choice("House (U/L): ", {"U", "L"})
    party_raw = input("Party (D/R/A/N or NULL): ").strip().upper()
    if party_raw in {"NULL", "OTHER", "NOT"}:
        party = "N"
    else:
        party = party_raw
    if party not in {"D", "R", "A", "N"}:
        print("Invalid party. Use D, R, A, N, or NULL.")
        return
    state = input("State (2-letter): ").strip().upper()

    if len(state) != 2:
        print("State should be a 2-letter code.")
        return

    try:
        if source == "D":
            out = list_dime_candidates(year, house, party, state)
        else:
            out = list_nimsp_candidates(year, house, party, state)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return

    print()
    print(f"Source={source} Year={year} House={house} Party={party} State={state}")
    print(f"Candidates found: {len(out)}")
    print("-" * 80)
    if out.empty:
        print("No candidates found for this subset.")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
