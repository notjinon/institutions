#!/usr/bin/env python3
"""
cspy-match3.py  —  INVERTED / NIMSP-FIRST REWRITE
===================================================

WHY THE OLD APPROACH FAILED
----------------------------
All previous versions (v1–v2 and earlier v3 fixes) relied on DIME's internal
`seat` column to identify state-leg candidates:

    WHERE seat IN ('state:upper', 'state:lower')

Starting around 2018, DIME largely stopped populating this column. By 2020,
the vast majority of state-leg transactions have seat = NULL. The old pipeline
therefore silently discarded millions of valid records — this is why high-
profile candidates like James Talarico appeared in the raw parquet but were
absent from the final output.

THE NEW "INVERTED" STRATEGY
----------------------------
NIMSP is treated as the authoritative source of truth for state-leg candidates.
The matching logic now operates in three phases:

  Phase 1 — Ingest Without Filters:
    Load ALL CAND-type DIME records for a given state/year with NO seat and
    NO party-code filter. Every transaction is preserved regardless of how
    DIME labelled (or failed to label) the recipient.

  Phase 2 — NIMSP-First Matching:
    Build a master candidate universe from the upperlower CSV files (the NIMSP
    Candidate Master List). For each NIMSP candidate, search the unfiltered
    DIME pool by composite key of Name + State + Year using the multi-tier
    fuzzy matching engine. This is the inverse of the old direction.

  Phase 3 — Retroactive Imputation:
    Any DIME bonica.rid that matches a known NIMSP candidate is retroactively
    labelled as a state donation. Party, house, district, and election status
    are all imputed from the NIMSP record. NULL seat and NULL party columns in
    DIME are no longer a disqualifier — they are irrelevant.

KEY STRUCTURAL CHANGES vs OLD VERSION
---------------------------------------
  • extract_unique_combos_parquet() REMOVED — it queried DIME's broken seat
    column to discover which states to process. Replaced by reading the list
    of states directly from the NIMSP upperlower CSV universe.

  • load_candidate_data_parquet() REPLACED by load_all_dime_cand_records() —
    the new function omits the `seat` and `recipient.party` filters entirely.

  • analyze_candidate_party_overlap_efficient() REPLACED by
    run_nimsp_first_analysis() — loop now iterates over NIMSP candidates and
    searches for them in DIME, not the other direction.

  • cross_party_conflict logic REMOVED — because NIMSP definitively assigns
    party, there is no ambiguity about which party a candidate belongs to.

  • log_parquet_summary() UPDATED — no longer reports state-leg rows by seat
    column; reports CAND-type totals and unique state coverage instead.

Usage:
    python cspy-match3.py
    # Enter years when prompted: 2018, 2020, 2022
"""

import pandas as pd
import re
import sys
import importlib.util
import duckdb
import time
from difflib import SequenceMatcher
from pathlib import Path

# Load utils.paths directly by file path (avoids import system issues)
workspace_root = Path(__file__).parent.parent
paths_module_path = workspace_root / "utils" / "paths.py"
spec = importlib.util.spec_from_file_location("utils.paths", paths_module_path)
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)

DIME_PARQUET_FILE = paths.DIME_PARQUET_FILE
NIMSP_PARTY_DATA = paths.NIMSP_PARTY_DATA
NIMSP_PARTY_DATA_PARQUET = paths.NIMSP_PARTY_DATA_PARQUET
PRIMARY_DATES_FILE = paths.PRIMARY_DATES_FILE
UPPER_HOUSE_FILE = paths.UPPER_HOUSE_FILE
LOWER_HOUSE_FILE = paths.LOWER_HOUSE_FILE
OUTPUT_FILE = paths.OUTPUT_FILE
ensure_output_dir_exists = paths.ensure_output_dir_exists


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_primary_dates(file_path):
    """Load primary dates CSV → dict {(state_abbrev, year): datetime}."""
    state_to_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
        'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY',
    }
    df = pd.read_csv(file_path)
    primary_lookup = {}
    for _, row in df.iterrows():
        raw_state = str(row['state']).strip()
        state = state_to_abbrev.get(raw_state, raw_state.upper())
        year = int(row['year'])
        try:
            primary_lookup[(state, year)] = pd.to_datetime(row['leg_date'])
        except Exception:
            print(f"  Warning: could not parse date for {state}-{year}: {row['leg_date']}")
    print(f"Loaded {len(primary_lookup)} primary dates")
    return primary_lookup


def normalize_name(name):
    """Normalize a candidate name → 'LAST FIRST [MIDDLE ...]' uppercase tokens."""
    if pd.isna(name):
        return ""
    text = str(name).upper()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in {"JR", "SR", "II", "III", "IV"}]
    return " ".join(tokens)


def extract_last_name(name):
    """
    Extract normalized last name, handling both 'LAST, FIRST' and 'FIRST LAST'.
    For 'LAST, FIRST' the comma branch returns the pre-comma part.
    For 'FIRST LAST' (no comma) returns the final token after normalization.
    """
    if pd.isna(name):
        return ""
    text = str(name).upper().strip()
    if "," in text:
        return normalize_name(text.split(",", 1)[0])
    parts = normalize_name(text).split()
    return parts[-1] if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4 + Fix 5: match_candidate_name now takes explicit dime_last_norm and
# requires >=2 tokens + last-name-in-intersection for token_subset.
# ─────────────────────────────────────────────────────────────────────────────

def match_candidate_name(dime_name_norm, dime_last_norm, ref_candidates, fuzzy_threshold=0.80):
    """
    Match a normalized DIME name against a list of NIMSP reference candidates.

    Parameters
    ----------
    dime_name_norm  : str  — output of normalize_name(candidate_name)
    dime_last_norm  : str  — output of extract_last_name(candidate_name)   [Fix 4]
    ref_candidates  : list of dicts with keys
                      'name_norm', 'last_norm', 'district',
                      'candidate_index', 'candidate_state'
    fuzzy_threshold : float — minimum SequenceMatcher ratio for tier-3

    Matching tiers
    --------------
    1. Exact full-name match
    2. Token-subset  (DIME tokens ⊆ ref tokens or vice versa)
       — requires shorter side ≥ 2 tokens                      [Fix 5]
       — requires last name in token intersection               [Fix 5]
       — catches 'BERKOWITZ ETHAN' vs 'BERKOWITZ ETHAN A'
    3. Last-name + fuzzy within same-last-name group
       — catches 'HIGGINS PATTI' vs 'HIGGINS PATRICIA C'

    Returns (matched_ref_dict, match_method) or (None, None).
    """
    dime_tokens = set(dime_name_norm.split())
    if not dime_tokens:
        return None, None

    # Fix 4: use the explicitly-computed last name, not split()[0]
    dime_last = dime_last_norm

    # ── Tier 1: exact ────────────────────────────────────────────────────────
    for ref in ref_candidates:
        if ref["name_norm"] == dime_name_norm:
            return ref, "exact"

    # ── Tier 2: token-subset ─────────────────────────────────────────────────
    # Fix 5: guard against 1-token names and require last name in intersection.
    subset_matches = []
    for ref in ref_candidates:
        ref_tokens = set(ref["name_norm"].split())
        if dime_tokens <= ref_tokens or ref_tokens <= dime_tokens:
            shorter = dime_tokens if len(dime_tokens) <= len(ref_tokens) else ref_tokens
            shared  = dime_tokens & ref_tokens
            # Require at least 2 tokens on shorter side AND last name must overlap
            if len(shorter) >= 2 and dime_last and dime_last in shared:
                subset_matches.append(ref)
    if len(subset_matches) == 1:
        return subset_matches[0], "token_subset"

    # ── Tier 3: last-name + fuzzy ────────────────────────────────────────────
    same_last = [r for r in ref_candidates if r["last_norm"] == dime_last]
    if not same_last:
        return None, None
    if len(same_last) == 1:
        score = SequenceMatcher(None, dime_name_norm, same_last[0]["name_norm"]).ratio()
        if score >= fuzzy_threshold:
            return same_last[0], f"fuzzy_last({score:.2f})"
        return None, None

    best_score, best_ref = 0.0, None
    for ref in same_last:
        score = SequenceMatcher(None, dime_name_norm, ref["name_norm"]).ratio()
        if score > best_score:
            best_score, best_ref = score, ref
    if best_score >= fuzzy_threshold:
        return best_ref, f"fuzzy_best({best_score:.2f})"
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# District / status helpers (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_district(office_sought):
    if pd.isna(office_sought):
        return "UNKNOWN"
    text = str(office_sought).upper()
    if "AT LARGE" in text or "AT-LARGE" in text:
        return "AL"
    cleaned = re.sub(r"^(SENATE|HOUSE|ASSEMBLY)\s+DISTRICT\s+", "", text).strip()
    if cleaned in {"AL", "AT LARGE", "AT-LARGE"}:
        return "AL"
    has_alpha = bool(re.search(r"[A-Z]", cleaned))
    has_digit = bool(re.search(r"\d", cleaned))
    if has_alpha and has_digit:
        return re.sub(r"[^A-Z0-9]", "", cleaned) or "UNKNOWN"
    if has_digit:
        m = re.search(r"(\d+)", cleaned)
        if m:
            return str(int(m.group(1)))
    return re.sub(r"[^A-Z0-9]", "", cleaned) or "UNKNOWN"


def map_election_status(election_status):
    """Map NIMSP Election_Status string → short outcome code."""
    status = str(election_status).strip().upper() if pd.notna(election_status) else ""
    STATUS_MAP = {
        "WON-GENERAL":             "W",
        "LOST-GENERAL":            "P",
        "WITHDREW-GENERAL":        "H",
        "DISQUALIFIED-GENERAL":    "DG",
        "DECEASED-GENERAL":        "XG",
        "DEFAULT WINNER-GENERAL":  "DW",
        "TIED-GENERAL":            "TG",
        "WON-GENERAL RUNOFF":      "WR",
        "LOST-GENERAL RUNOFF":     "LR",
        "LOST-PRIMARY":            "L",
        "LOST-TOP TWO PRIMARY":    "LT",
        "LOST-CONVENTION":         "LC",
        "WON-PRIMARY":             "WP",
        "WITHDREW-PRIMARY":        "HP",
        "DISQUALIFIED-PRIMARY":    "DP",
        "WON-PRIMARY RUNOFF":      "WPR",
        "LOST-PRIMARY RUNOFF":     "LPR",
        "DECEASED-PRIMARY RUNOFF": "XPR",
    }
    code = STATUS_MAP.get(status)
    if code is not None:
        return code
    if status:
        print(f"    WARNING: Unmapped election status: '{status}'")
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# NIMSP reference loader (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def load_district_reference(year, upper_path=None, lower_path=None):
    """
    Load upperlower CSV files for a given year and build
    {(state, party, house): [ref_candidate_dicts]} lookup.
    """
    party_map = {"DEMOCRATIC": "DEM", "REPUBLICAN": "REP"}

    if upper_path is None:
        upper_path = UPPER_HOUSE_FILE(year)
    if lower_path is None:
        lower_path = LOWER_HOUSE_FILE(year)

    frames = []
    for path, house in [(upper_path, "U"), (lower_path, "L")]:
        p = Path(str(path))
        if not p.exists():
            alt = p.with_suffix(".csv.csv")
            if alt.exists():
                print(f"  Note: using {alt.name} (double extension)")
                p = alt
        try:
            df = pd.read_csv(p, low_memory=False)
        except FileNotFoundError:
            print(f"  District reference missing: {p}")
            continue
        df = df.rename(columns=str.strip)
        df["house"]               = house
        df["state"]               = df["Election_Jurisdiction"].astype(str).str.upper().str.strip()
        df["party"]               = df["General_Party"].astype(str).str.upper().str.strip().map(party_map)
        df["candidate_name"]      = df["Candidate"].astype(str).str.strip()
        df["candidate_name_norm"] = df["candidate_name"].map(normalize_name)
        df["district"]            = df["Office_Sought"].map(normalize_district)
        df["total_amount"]        = pd.to_numeric(df["Total_$"], errors="coerce").fillna(0.0)
        df = df[df["party"].isin(["DEM", "REP"])].copy()
        frames.append(df)

    if not frames:
        return {}

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(
        ["state", "party", "house", "district", "total_amount", "candidate_name"],
        ascending=[True, True, True, True, False, True],
    ).copy()
    merged["candidate_index"] = (
        merged.groupby(["state", "party", "house", "district"])["candidate_name"]
        .cumcount()
        .add(1)
        .map(lambda x: str(int(x)).zfill(2))
    )
    merged["candidate_state"]    = merged["Election_Status"].map(map_election_status)
    merged["candidate_last_norm"] = merged["candidate_name"].map(extract_last_name)

    group_lookup = {}
    for _, row in merged.iterrows():
        key = (row["state"], row["party"], row["house"])
        group_lookup.setdefault(key, []).append({
            "name_norm":       row["candidate_name_norm"],
            "last_norm":       row["candidate_last_norm"],
            "district":        row["district"],
            "candidate_index": row["candidate_index"],
            "candidate_state": row["candidate_state"],
        })
    return group_lookup


# ─────────────────────────────────────────────────────────────────────────────
# Pre-run parquet summary (INVERTED version — no seat column required)
# ─────────────────────────────────────────────────────────────────────────────

def log_parquet_summary(parquet_file, year):
    """
    Print a data-coverage summary of the full parquet file before processing.

    The old version filtered by seat IN ('state:upper','state:lower') to count
    state-leg rows — that filter is exactly what broke 2018+ data. The new
    version reports CAND-type rows across all seat values and shows the seat
    NULL rate, making the metadata collapse immediately visible.
    """
    f = str(parquet_file)
    print(f"\n  {'─'*60}")
    print(f"  PARQUET SUMMARY  ({year})  [NIMSP-FIRST MODE]")
    print(f"  {'─'*60}")
    con = duckdb.connect()
    try:
        total = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [f]).fetchone()[0]

        cand_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)"
            " WHERE \"recipient.type\" = 'CAND'",
            [f]
        ).fetchone()[0]

        # Show how many CAND rows have a populated seat vs NULL — this surfaces
        # the extent of the metadata collapse that broke earlier versions.
        seat_breakdown = con.execute(
            "SELECT"
            "  COALESCE(seat, '<NULL>') AS seat_val,"
            "  COUNT(*) AS n"
            " FROM read_parquet(?)"
            " WHERE \"recipient.type\" = 'CAND'"
            " GROUP BY seat_val ORDER BY n DESC LIMIT 10",
            [f]
        ).df()

        unique_states = con.execute(
            "SELECT COUNT(DISTINCT \"recipient.state\") FROM read_parquet(?)"
            " WHERE \"recipient.type\" = 'CAND'"
            "   AND cycle = ? AND \"recipient.state\" IS NOT NULL",
            [f, year]
        ).fetchone()[0]

        party_null_rate = con.execute(
            "SELECT"
            "  ROUND(100.0 * SUM(CASE WHEN \"recipient.party\" IS NULL THEN 1 ELSE 0 END)"
            "        / COUNT(*), 2) AS null_pct"
            " FROM read_parquet(?)"
            " WHERE \"recipient.type\" = 'CAND' AND cycle = ?",
            [f, year]
        ).fetchone()[0]

        print(f"  Total rows in parquet       : {total:>12,}")
        print(f"  CAND-type rows (all seats)  : {cand_rows:>12,}")
        print(f"  Unique states (cycle={year}): {unique_states:>12,}")
        print(f"  recipient.party NULL rate   : {party_null_rate:>11.2f}%")
        print(f"  (High NULL% confirms metadata collapse — NIMSP-first strategy needed)")
        print(f"\n  `seat` value breakdown for CAND rows (top 10):")
        for _, row in seat_breakdown.iterrows():
            bar_n = int(row['n'] / max(cand_rows, 1) * 30)
            print(f"    {str(row['seat_val']):<25}  {int(row['n']):>10,}  {'█' * bar_n}")
    except Exception as e:
        print(f"  WARNING: could not generate parquet summary: {e}")
    finally:
        con.close()
    print(f"  {'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Ingest Without Filters — load ALL DIME CAND records
# ─────────────────────────────────────────────────────────────────────────────

def load_all_dime_cand_records(parquet_file, state, year, primary_date=None):
    """
    Load ALL CAND-type DIME records for a given state/year with NO seat filter
    and NO recipient.party filter.

    This replaces load_candidate_data_parquet(). The old function contained:
        AND seat IN ('state:upper', 'state:lower')
        AND ("recipient.party" = '100' OR "recipient.party" IS NULL)

    Both of those filters are now gone. A federal senator, a state senator,
    and a candidate with a NULL seat column all land in the same result set.
    NIMSP matching (Phase 2) decides which ones are state-leg candidates.
    We retain the primary-date cutoff because that is a temporal validity
    constraint, not a metadata-quality assumption.
    """
    f = str(parquet_file)
    date_clause = ""
    if primary_date is not None:
        primary_str = primary_date.strftime('%Y-%m-%d')
        date_clause = f" AND TRY_CAST(date AS DATE) < DATE '{primary_str}'"

    sql = (
        f"SELECT * FROM read_parquet('{f}')"
        f" WHERE cycle = {year}"
        f"   AND \"recipient.state\" = '{state}'"
        f"   AND \"recipient.type\" = 'CAND'"
        f"{date_clause}"
    )
    con = duckdb.connect()
    try:
        df = con.execute(sql).df()
        seat_null = df['seat'].isna().sum() if 'seat' in df.columns else 0
        party_null = df['recipient.party'].isna().sum() if 'recipient.party' in df.columns else 0
        print(f"  Loaded {len(df):,} CAND records (all seats)  "
              f"[seat NULL: {seat_null:,}, party NULL: {party_null:,}]")
        return df
    finally:
        con.close()


def build_dime_candidate_list(dime_df):
    """
    Build a flat list of unique DIME candidate dicts from the unfiltered pool.

    Each dict has the keys expected by match_candidate_name():
        name_norm  — normalize_name() of recipient.name
        last_norm  — extract_last_name() of recipient.name
    Plus extra fields used after a successful match:
        raw_name   — original recipient.name string
        bonica_rid — bonica.rid value (used to retrieve transactions)

    This list is the search space for Phase 2 NIMSP-first matching.
    """
    candidates = []
    seen_rids = set()
    for rid in dime_df['bonica.rid'].unique():
        if rid in seen_rids:
            continue
        seen_rids.add(rid)
        rows = dime_df[dime_df['bonica.rid'] == rid]
        name = rows['recipient.name'].iloc[0] if not rows.empty else ''
        candidates.append({
            'name_norm': normalize_name(name),
            'last_norm': extract_last_name(name),
            'raw_name':  name,
            'bonica_rid': rid,
        })
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 + 3: NIMSP-First Matching & Retroactive Imputation
# ─────────────────────────────────────────────────────────────────────────────

def run_nimsp_first_analysis(state, year, nimsp_universe, dime_df, dem_donors, rep_donors):
    """
    Core NIMSP-first analysis for a single (state, year).

    Inverted matching logic
    -----------------------
    OLD: iterate over DIME bonica.rids → search NIMSP reference lists
    NEW: iterate over NIMSP candidates   → search DIME candidate pool

    A DIME bonica.rid that is found via NIMSP matching is retroactively
    labelled as a state donation. Its party, house, district, and
    election status are all imputed from the NIMSP record — DIME's NULL
    `seat` and NULL `recipient.party` columns are completely bypassed.

    Parameters
    ----------
    state          : two-letter state abbreviation
    year           : election cycle year
    nimsp_universe : dict from load_district_reference()
                     {(state, party, house): [ref_candidate_dicts]}
    dime_df        : unfiltered DIME CAND records for this state/year
    dem_donors     : NIMSP party donor DF for DEM
    rep_donors     : NIMSP party donor DF for REP

    Returns
    -------
    (results_df, unmatched_nimsp_list)
        results_df          — one row per successfully matched candidate
        unmatched_nimsp_list — NIMSP candidates for which no DIME record found
    """
    print(f"\n  NIMSP-first analysis: {state}-{year}")

    if dime_df.empty:
        print("    No DIME CAND records found — skipping.")
        return pd.DataFrame(), []

    dime_candidate_list = build_dime_candidate_list(dime_df)
    print(f"    DIME candidate pool (unfiltered): {len(dime_candidate_list):,}")

    # ── Pre-build donor name sets ──────────────────────────────────────────
    def _make_donor_set(df):
        if df is None or df.empty:
            return set()
        tmp = df.copy()
        tmp['_c'] = tmp['Name'].str.upper().str.strip()
        return set(tmp['_c'].dropna())

    dem_donor_names = _make_donor_set(dem_donors)
    rep_donor_names = _make_donor_set(rep_donors)
    print(f"    DEM party donor pool: {len(dem_donor_names):,}")
    print(f"    REP party donor pool: {len(rep_donor_names):,}")

    results        = []
    unmatched_nimsp = []
    used_rids       = set()   # guard against double-counting same bonica.rid

    # ── Iterate over NIMSP candidates ─────────────────────────────────────
    for (key_state, party, house), ref_list in nimsp_universe.items():
        if key_state != state:
            continue

        party_donor_names = dem_donor_names if party == 'DEM' else rep_donor_names
        opp_donor_names   = rep_donor_names if party == 'DEM' else dem_donor_names
        total_party_donors = len(party_donor_names)

        for nimsp_ref in ref_list:
            # ── Phase 2: match this NIMSP candidate against the DIME pool ──
            # match_candidate_name() is direction-agnostic; we pass the NIMSP
            # candidate's name as the "query" and the DIME pool as the ref list.
            matched_dime, method = match_candidate_name(
                nimsp_ref['name_norm'],
                nimsp_ref['last_norm'],
                dime_candidate_list,
            )

            if matched_dime is None:
                unmatched_nimsp.append({
                    'nimsp_name': nimsp_ref['name_norm'],
                    'state': state,
                    'party': party,
                    'house': house,
                    'year':  year,
                })
                continue

            rid = matched_dime['bonica_rid']

            # Warn if the same DIME bonica.rid would be double-counted
            if rid in used_rids:
                print(f"    [WARN] bonica.rid {rid} ({matched_dime['raw_name']}) "
                      f"already matched by a previous NIMSP candidate — skipping duplicate.")
                continue
            used_rids.add(rid)

            # ── Phase 3: retroactive imputation ────────────────────────────
            cand_rows = dime_df[dime_df['bonica.rid'] == rid]
            cand_name = matched_dime['raw_name']

            # Party, house, district, election status all come from NIMSP —
            # DIME's NULL seat and NULL party columns are ignored entirely.
            district        = nimsp_ref['district']
            candidate_index = nimsp_ref['candidate_index']
            candidate_state = nimsp_ref.get('candidate_state')
            seat_type       = 'state:upper' if house == 'U' else 'state:lower'
            candidate_id    = f"{state}-{party}-{year}-{house}-{district}-{candidate_index}"

            # ── Donor overlap ──────────────────────────────────────────────
            cand_copy = cand_rows.copy()
            cand_copy['_c'] = cand_copy['contributor.name'].str.upper().str.strip()
            candidate_donors        = set(cand_copy['_c'].dropna())
            total_candidate_donors  = cand_rows['bonica.cid'].nunique()
            party_donors_count      = len(candidate_donors & party_donor_names)
            cross_partisan_count    = len(candidate_donors & opp_donor_names)
            percentage = (party_donors_count / total_party_donors * 100) if total_party_donors > 0 else 0

            results.append({
                'candidate_id':             candidate_id,
                'candidate_name':           cand_name,
                'nimsp_name':               nimsp_ref['name_norm'],
                'party_donors_count':       party_donors_count,
                'total_party_donors':       total_party_donors,
                'total_candidate_donors':   total_candidate_donors,
                'cross_partisan_donations': cross_partisan_count,
                'percentage':               round(percentage, 2),
                'seat_info':                f"{house}-{district}",
                'seat_type':                seat_type,
                'candidate_state':          candidate_state,
                'match_method':             method,
                'in_NIMSP':                 True,
                'state':                    state,
                'party':                    party,
                'year':                     year,
            })

    # ── Diagnostic counts ──────────────────────────────────────────────────
    all_rids = {c['bonica_rid'] for c in dime_candidate_list}
    orphan_count = len(all_rids - used_rids)
    nimsp_total  = sum(
        len(rlist)
        for (s, p, h), rlist in nimsp_universe.items()
        if s == state
    )

    print(f"    NIMSP candidates for {state}          : {nimsp_total:,}")
    print(f"    Matched (NIMSP→DIME)                   : {len(results):,}")
    print(f"    NIMSP unmatched (no DIME record)       : {len(unmatched_nimsp):,}")
    print(f"    DIME orphans (federal/other, discarded): {orphan_count:,}")

    out_df = pd.DataFrame(results) if results else pd.DataFrame(columns=[
        'candidate_id', 'candidate_name', 'nimsp_name',
        'party_donors_count', 'total_party_donors', 'total_candidate_donors',
        'cross_partisan_donations', 'percentage',
        'seat_info', 'seat_type', 'candidate_state',
        'match_method', 'in_NIMSP', 'state', 'party', 'year',
    ])
    return out_df, unmatched_nimsp


# ─────────────────────────────────────────────────────────────────────────────
# NIMSP party data loading (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def load_party_data_parquet(parquet_file, state, party, year, primary_date=None):
    party_name = 'Democratic' if party == 'DEM' else 'Republican'
    f = str(parquet_file)
    date_clause = ""
    if primary_date is not None:
        primary_str = primary_date.strftime('%Y-%m-%d')
        date_clause = f" AND TRY_CAST(CFS_Date AS DATE) < DATE '{primary_str}'"
    sql = (
        f"SELECT * FROM read_parquet('{f}')"
        f" WHERE ElectionYear = {year}"
        f"   AND SAT_State = '{state}'"
        f"   AND FSPC_PartyType = '{party_name}'"
        f"{date_clause}"
    )
    con = duckdb.connect()
    try:
        df = con.execute(sql).df()
        print(f"  Loaded {len(df)} party donor records")
        return df
    finally:
        con.close()


def load_party_data_chunked(file_path, state, party, year, primary_date=None, chunksize=10000):
    party_name = 'Democratic' if party == 'DEM' else 'Republican'
    filtered_chunks = []
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, low_memory=False, chunksize=chunksize)):
        filters = (
            (chunk['ElectionYear'] == year) &
            (chunk['SAT_State'] == state) &
            (chunk['FSPC_PartyType'] == party_name)
        )
        if primary_date is not None:
            chunk['date_parsed'] = pd.to_datetime(chunk['CFS_Date'], errors='coerce')
            filters = filters & (chunk['date_parsed'] < primary_date) & chunk['date_parsed'].notna()
        filtered = chunk[filters]
        if len(filtered) > 0:
            filtered_chunks.append(filtered)
        if chunk_num % 10 == 0:
            print(f"  Processed {(chunk_num + 1) * chunksize:,} rows...")
    if filtered_chunks:
        result = pd.concat(filtered_chunks, ignore_index=True)
        print(f"  Loaded {len(result)} party donor records (CSV)")
        return result
    return pd.DataFrame()


def load_party_data_smart(state, party, year, primary_date=None):
    if NIMSP_PARTY_DATA_PARQUET.exists():
        return load_party_data_parquet(str(NIMSP_PARTY_DATA_PARQUET), state, party, year, primary_date)
    return load_party_data_chunked(str(NIMSP_PARTY_DATA), state, party, year, primary_date)


# ─────────────────────────────────────────────────────────────────────────────
# Year runner — NIMSP-first orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_for_year(parquet_file, primary_dates_path, year):
    """
    Process all states for a given year using the NIMSP-first strategy.

    State discovery
    ---------------
    OLD: queried DIME's `seat` column for unique (state, year) combos.
    NEW: derives the state list from which upperlower CSV files exist for
         `year` — i.e. from the NIMSP Candidate Master List itself. DIME is
         never asked about seat metadata.

    Per-state loop
    --------------
    For each state that NIMSP covers:
      1. Load ALL DIME CAND records for (state, year) — no seat/party filter.
      2. Load DEM and REP party donor pools from NIMSP party_donor.
      3. Run run_nimsp_first_analysis(): NIMSP candidates → DIME pool →
         retroactive imputation of party/house/district.
      4. Accumulate results.
    """
    start_time = time.time()

    log_parquet_summary(parquet_file, year)
    primary_lookup = load_primary_dates(primary_dates_path)

    # Build NIMSP candidate universe from upperlower CSVs
    print(f"\n  Loading NIMSP candidate universe for {year}...")
    nimsp_universe = load_district_reference(year)
    if not nimsp_universe:
        print(f"  No NIMSP upperlower data found for {year} — cannot proceed.")
        return

    # Derive states from NIMSP, not from DIME seat column
    states_in_nimsp = sorted(set(s for s, p, h in nimsp_universe.keys()))
    print(f"  States with NIMSP data: {len(states_in_nimsp)}")

    print(f"\n{'='*60}")
    print(f"  PROCESSING {year}: {len(states_in_nimsp)} states  [NIMSP-FIRST]")
    print(f"{'='*60}")

    year_results    = []
    all_unmatched   = []

    for state in states_in_nimsp:
        primary_date = primary_lookup.get((state, year))
        if primary_date is None:
            print(f"\n  {state}-{year}: SKIPPED — no primary date in lookup")
            continue

        try:
            # Phase 1: load ALL DIME CAND records (no seat/party filter)
            dime_df      = load_all_dime_cand_records(parquet_file, state, year, primary_date)
            # Load both party donor pools up front for this state/year
            dem_donors   = load_party_data_smart(state, 'DEM', year, primary_date)
            rep_donors   = load_party_data_smart(state, 'REP', year, primary_date)

            # Phases 2 + 3: NIMSP-first matching + retroactive imputation
            state_results, unmatched_list = run_nimsp_first_analysis(
                state, year, nimsp_universe, dime_df, dem_donors, rep_donors,
            )

            if not state_results.empty:
                year_results.append(state_results)
            all_unmatched.extend(unmatched_list)

        except Exception as e:
            print(f"  ERROR {state}-{year}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not year_results:
        print(f"\n  Year {year}: no valid results produced.")
        return

    year_df = pd.concat(year_results, ignore_index=True)
    year_df = year_df.sort_values(
        ['state', 'party', 'percentage'],
        ascending=[True, True, False],
    )

    output_path = OUTPUT_FILE(year)
    year_df.to_csv(output_path, index=False)

    total   = len(year_df)
    elapsed = time.time() - start_time

    # Match-method breakdown
    if 'match_method' in year_df.columns:
        method_counts = year_df['match_method'].value_counts()
        method_str = ', '.join(f"{m}:{c}" for m, c in method_counts.items())
    else:
        method_str = 'n/a'

    nimsp_unmatched_total = len(all_unmatched)

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {year}  [NIMSP-FIRST]")
    print(f"{'='*60}")
    print(f"  Candidates in output   : {total:,}")
    print(f"  All in_NIMSP=True      : (all matched via NIMSP-first)")
    print(f"  NIMSP unmatched total  : {nimsp_unmatched_total:,}")
    print(f"  Match-method breakdown : {method_str}")
    print(f"  Runtime                : {int(elapsed//60)}m {elapsed%60:.1f}s")
    print(f"  Output                 : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

TARGET_YEARS = input("Enter target years (comma-separated, e.g. 2020, 2022): ")


def parse_target_years(years_input):
    if isinstance(years_input, list):
        return years_input
    return [int(y.strip()) for y in str(years_input).strip().split(',')]


if __name__ == "__main__":
    target_years = parse_target_years(TARGET_YEARS)
    ensure_output_dir_exists()

    print(f"\n{'='*60}")
    print(f"  cspy-match3  —  PROCESSING YEARS: {target_years}")
    print(f"{'='*60}")

    if NIMSP_PARTY_DATA_PARQUET.exists():
        print(f"  NIMSP: {NIMSP_PARTY_DATA_PARQUET}  (parquet)")
    else:
        print(f"  NIMSP: {NIMSP_PARTY_DATA}  (CSV — run convert_party_data.py for speed)")
    print(f"  Primary dates: {PRIMARY_DATES_FILE}\n")

    for year in target_years:
        parquet_file = DIME_PARQUET_FILE(year)
        print(f"\n{'='*60}")
        print(f"  STARTING YEAR {year}")
        print(f"{'='*60}")
        print(f"  Parquet: {parquet_file}")
        try:
            run_analysis_for_year(parquet_file, str(PRIMARY_DATES_FILE), year)
        except FileNotFoundError as e:
            print(f"  ERROR: missing file — {e}")
            print(f"  TIP: run 'python convert_year.py {year}' to create parquet")
            continue
        except Exception as e:
            import traceback
            print(f"  ERROR year {year}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("  ALL PROCESSING COMPLETE")
    print(f"{'='*60}")
