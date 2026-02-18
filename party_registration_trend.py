#!/usr/bin/env python3
"""
party_registration_trend.py

Track stated party coding by year for either:
- DIME parquet state-leg candidates (`--source D`)
- NIMSP upper/lower CSV candidates (`--source N`)

Buckets:
- DEM
- REP
- 3PT
- NUL

Optional:
- --source D|N
- --basis rows|candidates
- --out <path>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import duckdb
import pandas as pd
from paths import LOWER_HOUSE_FILE, UPPER_HOUSE_FILE


def find_dime_year_parquets(base_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for p in sorted(base_dir.glob("*_parquet/*_candidate_donor.parquet")):
        stem = p.stem
        year_token = stem.split("_", 1)[0]
        if year_token.isdigit():
            pairs.append((int(year_token), p))
    return pairs


def find_nimsp_years(base_dir: Path) -> list[int]:
    years: set[int] = set()
    pat = re.compile(r"^(\d{4})_(upper|lower)\.csv(?:\.csv)?$", re.IGNORECASE)
    for p in base_dir.glob("*"):
        m = pat.match(p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def pick_upperlower_file(year: int, house: str) -> Path | None:
    path = Path(str(UPPER_HOUSE_FILE(year) if house == "U" else LOWER_HOUSE_FILE(year)))
    if path.exists():
        return path
    alt = path.with_suffix(".csv.csv")
    return alt if alt.exists() else None


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "cp1252", "latin-1")
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    return pd.read_csv(path, low_memory=False)


def build_dime_query(basis: str) -> str:
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


def normalize_nimsp_party(value: object) -> str:
    if pd.isna(value):
        return "NUL"
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return "NUL"
    if text.startswith("DEM"):
        return "DEM"
    if text.startswith("REP"):
        return "REP"
    return "3PT"


def run_dime(basis: str) -> pd.DataFrame:
    year_files = find_dime_year_parquets(Path("DIME data"))
    if not year_files:
        raise FileNotFoundError("No parquet files found in DIME data/*_parquet/")

    con = duckdb.connect()
    try:
        q = build_dime_query(basis)
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
                    "source": "D",
                    "source_path": str(parquet_path),
                }
            )
    finally:
        con.close()

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def run_nimsp(basis: str) -> pd.DataFrame:
    years = find_nimsp_years(Path("upperlower"))
    if not years:
        raise FileNotFoundError("No files found in upperlower matching *_upper/_lower.csv(.csv)")

    rows: list[dict] = []
    for year in years:
        frames: list[pd.DataFrame] = []
        for house in ("U", "L"):
            p = pick_upperlower_file(year, house)
            if p is None:
                continue
            df = read_csv_with_fallback(p)
            df.columns = df.columns.str.strip()
            if "Election_Jurisdiction" not in df.columns or "General_Party" not in df.columns:
                continue

            df["_state"] = df["Election_Jurisdiction"].astype(str).str.upper().str.strip()
            df["_party_bucket"] = df["General_Party"].map(normalize_nimsp_party)
            df["_house"] = house
            df["_candidate"] = df["Candidate"].astype(str).str.upper().str.strip() if "Candidate" in df.columns else ""
            df["_district"] = df["Office_Sought"].astype(str).str.upper().str.strip() if "Office_Sought" in df.columns else ""
            frames.append(df[["_state", "_party_bucket", "_house", "_candidate", "_district"]].copy())

        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)
        if basis == "rows":
            total = int(len(merged))
            dem = int((merged["_party_bucket"] == "DEM").sum())
            rep = int((merged["_party_bucket"] == "REP").sum())
            third_party = int((merged["_party_bucket"] == "3PT").sum())
            nul = int((merged["_party_bucket"] == "NUL").sum())
        else:
            merged["_key"] = (
                merged["_state"].fillna("")
                + "|"
                + merged["_house"].fillna("")
                + "|"
                + merged["_district"].fillna("")
                + "|"
                + merged["_candidate"].fillna("")
            )
            total = int(merged["_key"].nunique())
            dem = int(merged.loc[merged["_party_bucket"] == "DEM", "_key"].nunique())
            rep = int(merged.loc[merged["_party_bucket"] == "REP", "_key"].nunique())
            third_party = int(merged.loc[merged["_party_bucket"] == "3PT", "_key"].nunique())
            nul = int(merged.loc[merged["_party_bucket"] == "NUL", "_key"].nunique())

        state_count = int(merged["_state"].replace("", pd.NA).dropna().nunique())
        stateleg_rows = int(len(merged))
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
                "source": "N",
                "source_path": "upperlower",
            }
        )

    if not rows:
        raise RuntimeError("No usable NIMSP rows found across available upper/lower files.")
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def run(source: str, basis: str, out_path: Path) -> pd.DataFrame:
    if source == "D":
        df = run_dime(basis)
    else:
        df = run_nimsp(basis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track party coding by year (DEM/REP/3PT/NUL).")
    p.add_argument(
        "--source",
        choices=["D", "N"],
        default="D",
        help="D = DIME parquet, N = NIMSP upper/lower CSVs",
    )
    p.add_argument(
        "--basis",
        choices=["rows", "candidates"],
        default="rows",
        help="rows = raw rows; candidates = distinct candidate IDs (source-dependent)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: outputs/party_registration_trend_<source>_<basis>.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_name = "dime" if args.source == "D" else "nimsp"
    out = args.out or Path("outputs") / f"party_registration_trend_{source_name}_{args.basis}.csv"
    df = run(args.source, args.basis, out)

    print(f"Wrote {len(df)} years -> {out}")
    print(
        df[
            [
                "year",
                "source",
                "total",
                "DEM",
                "REP",
                "3PT",
                "NUL",
                "DEM_pct",
                "REP_pct",
                "3PT_pct",
                "NUL_pct",
                "state_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
