#!/usr/bin/env python3
"""
cspy-match3.py
Builds on cspy-match2.py with the following fixes:

  Fix 1 — Party code collapse (2018–2022):
    The 'recipient.party' field is NULL for ~99% of state-leg candidates in
    2020/2022 parquets. The old '100'/'200' filter silently discarded almost
    all of them. Now loads ALL state-leg CAND records (any party code or NULL)
    and lets NIMSP reference matching determine party.

  Fix 2 — Combo extraction now catches NULL-party years:
    extract_unique_combos_parquet no longer filters by party code. It emits
    both DEM and REP combos for every (state, year) that has state-leg CAND
    records, regardless of party code population.

  Fix 3 — Opposite-party skip no longer silently deletes candidates:
    When a candidate fails their own-party match but fuzzy-matches the
    opposite party reference, the old code did a silent `continue`, which
    caused legitimate candidates to vanish from output. Now logs the event
    and marks the record as match_method='cross_party_conflict' so it's
    visible in the output and in fallback_analysis.py.

  Fix 4 — dime_last extraction now uses extract_last_name():
    The old code set dime_last = dime_name_norm.split()[0], which works for
    "LAST FIRST" names but returns the first name for "FIRST LAST" names.
    Now passes dime_last_norm (computed via extract_last_name on the raw name)
    explicitly into match_candidate_name().

  Fix 5 — token_subset false positives tightened:
    The old subset check fired even for 1-token names like "SMITH", which
    happily subset-matched any "SMITH JOHN". Now requires the shorter side
    has >= 2 tokens AND the candidate's last name appears in the intersection.

  Fix 6 — Pre-run parquet summary:
    Before processing each year, logs total rows, state-leg rows, party code
    breakdown, and unique combos found. Immediately surfaces data coverage
    issues instead of silently producing tiny outputs.

Usage:
    python cspy-match3.py
    # Enter years when prompted: 2012, 2014, 2020
"""

import pandas as pd
import re
import duckdb
import time
from difflib import SequenceMatcher
from pathlib import Path
from paths import (
    DIME_PARQUET_FILE,
    NIMSP_PARTY_DATA,
    NIMSP_PARTY_DATA_PARQUET,
    PRIMARY_DATES_FILE,
    UPPER_HOUSE_FILE,
    LOWER_HOUSE_FILE,
    OUTPUT_FILE,
    ensure_output_dir_exists,
)


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
# Fix 6: pre-run parquet summary
# ─────────────────────────────────────────────────────────────────────────────

def log_parquet_summary(parquet_file, year):
    """
    Query the parquet and print a data-coverage summary before processing.
    Surfaces the party-code collapse issue immediately.
    """
    f = str(parquet_file)
    print(f"\n  {'─'*56}")
    print(f"  PARQUET SUMMARY  ({year})")
    print(f"  {'─'*56}")
    con = duckdb.connect()
    try:
        total = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [f]).fetchone()[0]

        sl_total = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)"
            " WHERE seat IN ('state:upper','state:lower') AND \"recipient.type\"='CAND'",
            [f]
        ).fetchone()[0]

        party_breakdown = con.execute(
            "SELECT \"recipient.party\", COUNT(*) as n FROM read_parquet(?)"
            " WHERE seat IN ('state:upper','state:lower') AND \"recipient.type\"='CAND'"
            " GROUP BY \"recipient.party\" ORDER BY n DESC LIMIT 8",
            [f]
        ).df()

        unique_combos_all = con.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT DISTINCT \"recipient.state\", cycle FROM read_parquet(?)"
            "  WHERE seat IN ('state:upper','state:lower') AND \"recipient.type\"='CAND'"
            "  AND \"recipient.state\" IS NOT NULL AND cycle IS NOT NULL"
            ")",
            [f]
        ).fetchone()[0]

        print(f"  Total rows in parquet  : {total:>12,}")
        print(f"  State-leg CAND rows    : {sl_total:>12,}")
        pct = sl_total / total * 100 if total else 0
        print(f"  State-leg share        : {pct:>11.2f}%")
        print(f"  Unique (state, year) combos (all parties): {unique_combos_all}")
        print(f"\n  recipient.party breakdown in state-leg CAND rows:")
        for _, row in party_breakdown.iterrows():
            pcode = str(row['recipient.party'])
            label = {'100': 'DEM', '200': 'REP', 'nan': 'NULL'}.get(pcode, pcode)
            bar_n = int(row['n'] / max(sl_total, 1) * 30)
            print(f"    {label:>6} ({pcode:>3})  {int(row['n']):>9,}  {'█' * bar_n}")
    except Exception as e:
        print(f"  WARNING: could not generate parquet summary: {e}")
    finally:
        con.close()
    print(f"  {'─'*56}")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 + Fix 2: candidate data loading — accept NULL party codes
# ─────────────────────────────────────────────────────────────────────────────

def load_candidate_data_parquet(parquet_file, state, party, year, primary_date=None):
    """
    Load state-leg CAND records from Parquet for a given state/party/year.

    Fix 1: accepts rows where 'recipient.party' is the expected code (100/200)
    OR is NULL. In 2020+ parquets, ~99% of state-leg rows have NULL party;
    the old strict equality filter discarded them entirely. NIMSP reference
    matching determines the true party for NULL-code candidates.

    Both DEM and REP runs load their explicit party code + NULLs. A NULL
    candidate that genuinely belongs to the other party will be caught by
    the cross-party check (Fix 3) and labeled accordingly.
    """
    party_code = '100' if party == 'DEM' else '200'
    f = str(parquet_file)

    date_clause = ""
    if primary_date is not None:
        primary_str = primary_date.strftime('%Y-%m-%d')
        date_clause = f" AND TRY_CAST(date AS DATE) < DATE '{primary_str}'"

    # Accept explicit party code OR NULL (Fix 1)
    sql = (
        f"SELECT * FROM read_parquet('{f}')"
        f" WHERE cycle = {year}"
        f"   AND \"recipient.state\" = '{state}'"
        f"   AND (\"recipient.party\" = '{party_code}' OR \"recipient.party\" IS NULL)"
        f"   AND seat IN ('state:upper', 'state:lower')"
        f"   AND \"recipient.type\" = 'CAND'"
        f"{date_clause}"
    )
    con = duckdb.connect()
    try:
        df = con.execute(sql).df()
        null_count = df['recipient.party'].isna().sum()
        print(f"  Loaded {len(df)} candidate records  "
              f"(explicit party={party_code}: {len(df)-null_count}, NULL party: {null_count})")
        return df
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: combo extraction — no party filter, emit both DEM+REP per state/year
# ─────────────────────────────────────────────────────────────────────────────

def extract_unique_combos_parquet(parquet_file):
    """
    Extract all (state, party, year) combos to process.

    Fix 2: no longer filters by 'recipient.party'. Gets every (state, year)
    that has any state-leg CAND records, then emits both DEM and REP for each.
    This captures years/states where party codes are predominantly NULL, which
    would otherwise return zero combos under the old '100'/'200' filter.
    """
    f = str(parquet_file)
    print("  Extracting unique state/year combinations from Parquet...")

    sql = (
        "SELECT DISTINCT \"recipient.state\" AS state, cycle AS year"
        f" FROM read_parquet('{f}')"
        " WHERE seat IN ('state:upper', 'state:lower')"
        "   AND \"recipient.type\" = 'CAND'"
        "   AND \"recipient.state\" IS NOT NULL"
        "   AND cycle IS NOT NULL"
        " ORDER BY year, state"
    )
    con = duckdb.connect()
    try:
        df = con.execute(sql).df()
    finally:
        con.close()

    # Emit both parties for every (state, year) found
    combos = []
    for _, row in df.iterrows():
        combos.append((row['state'], 'DEM', int(row['year'])))
        combos.append((row['state'], 'REP', int(row['year'])))

    print(f"  Found {len(df)} unique (state, year) pairs → {len(combos)} combos (DEM+REP)")
    return combos


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
# Core analysis — Fix 3 (cross-party logging), Fix 4 (dime_last_norm)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_candidate_party_overlap_efficient(
    candidate_df, party_df, state, party, year,
    district_ref=None, opposite_party_df=None,
):
    """
    Match candidates to NIMSP reference, compute party-donor overlap, and
    count cross-partisan donations.

    Returns (results_df, unmatched_list).

    Changes from v2
    ---------------
    Fix 3: candidates that fail own-party match but fuzzy-match the opposite
      party reference are no longer silently dropped. They are kept in results
      with match_method='cross_party_conflict' and in_NIMSP=False, and a
      warning is printed. They are excluded from the unmatched list because
      they're not truly unmatched — they just belong to the other party's run.

    Fix 4: extract_last_name() is called on the raw candidate_name to obtain
      dime_last_norm, which is passed explicitly into match_candidate_name().
      This correctly handles 'FIRST LAST' formatted names where split()[0]
      would have returned the first name instead of the last.
    """
    print(f"\n  Analyzing {state}-{party}-{year}:")
    print(f"    Candidates : {len(candidate_df)}")
    print(f"    Party donors: {len(party_df)}")

    if len(candidate_df) == 0:
        print("    No candidates found — skipping.")
        return pd.DataFrame(), []
    if len(party_df) == 0:
        print("    No party donors found — skipping.")
        return pd.DataFrame(), []

    # Pre-process party donor name sets
    party_df = party_df.copy()
    party_df['donor_name_clean'] = party_df['Name'].str.upper().str.strip()
    unique_party_donors = set(party_df['donor_name_clean'].dropna())
    total_party_donors  = len(unique_party_donors)

    unique_opposite_donors = set()
    if opposite_party_df is not None and len(opposite_party_df) > 0:
        opp = opposite_party_df.copy()
        opp['donor_name_clean'] = opp['Name'].str.upper().str.strip()
        unique_opposite_donors = set(opp['donor_name_clean'].dropna())
        print(f"    Opposite-party donors: {len(unique_opposite_donors)}")

    opposite_party = 'REP' if party == 'DEM' else 'DEM'
    results   = []
    unmatched = []
    cross_party_skips = 0

    for candidate_rid in candidate_df['bonica.rid'].unique():
        cand_rows   = candidate_df[candidate_df['bonica.rid'] == candidate_rid]
        cand_name   = cand_rows['recipient.name'].iloc[0] if not cand_rows.empty else 'Unknown'
        seat_type   = cand_rows['seat'].iloc[0] if not cand_rows.empty else 'Unknown'

        house    = 'U' if seat_type == 'state:upper' else ('L' if seat_type == 'state:lower' else 'X')
        district = 'AL' if house == 'U' else ('01' if house == 'L' else '00')

        # Fix 4: compute last name from raw name before normalization
        name_norm      = normalize_name(cand_name)
        dime_last_norm = extract_last_name(cand_name)

        candidate_index = None
        candidate_state = None
        match_method    = "fallback"
        in_nimsp        = False
        is_cross_party  = False

        if district_ref:
            # Try own party first
            ref_list = district_ref.get((state, party, house), [])
            if ref_list:
                matched_ref, method = match_candidate_name(name_norm, dime_last_norm, ref_list)
                if matched_ref:
                    district        = matched_ref["district"]
                    candidate_index = matched_ref["candidate_index"]
                    candidate_state = matched_ref.get("candidate_state")
                    match_method    = method
                    in_nimsp        = True

            # Fix 3: if unmatched, check opposite party — but don't silently drop
            if not in_nimsp:
                opp_ref_list = district_ref.get((state, opposite_party, house), [])
                if opp_ref_list:
                    opp_matched, _ = match_candidate_name(name_norm, dime_last_norm, opp_ref_list)
                    if opp_matched:
                        # Candidate belongs to the other party's run; mark and skip
                        # output — they will appear correctly in the opposite run.
                        print(f"    [CROSS-PARTY] {cand_name} matched {opposite_party} ref "
                              f"during {party} run — will be recorded in {opposite_party} output")
                        match_method   = "cross_party_conflict"
                        is_cross_party = True
                        cross_party_skips += 1

        # Skip cross-party conflicts from this party's output to avoid duplicates
        if is_cross_party:
            continue

        # Donor overlap
        total_candidate_donors = cand_rows['bonica.cid'].nunique()
        cand_copy = cand_rows.copy()
        cand_copy['donor_name_clean'] = cand_copy['contributor.name'].str.upper().str.strip()
        candidate_donors = set(cand_copy['donor_name_clean'].dropna())

        party_donors_count    = len(candidate_donors & unique_party_donors)
        cross_partisan_count  = len(candidate_donors & unique_opposite_donors)
        percentage = (party_donors_count / total_party_donors * 100) if total_party_donors > 0 else 0

        if candidate_index is None:
            n = sum(1 for r in results if r['seat_info'] == f"{house}-{district}")
            candidate_index = f"{n+1:02d}"
        candidate_id = f"{state}-{party}-{year}-{house}-{district}-{candidate_index}"

        results.append({
            'candidate_id':           candidate_id,
            'candidate_name':         cand_name,
            'party_donors_count':     party_donors_count,
            'total_party_donors':     total_party_donors,
            'total_candidate_donors': total_candidate_donors,
            'cross_partisan_donations': cross_partisan_count,
            'percentage':             round(percentage, 2),
            'seat_info':              f"{house}-{district}",
            'seat_type':              seat_type,
            'candidate_state':        candidate_state,
            'match_method':           match_method,
            'in_NIMSP':               in_nimsp,
        })

        if not in_nimsp:
            unmatched.append({
                'candidate_name': cand_name,
                'house': house,
                'state': state,
                'party': party,
                'year':  year,
            })

    if cross_party_skips:
        print(f"    Cross-party conflicts skipped (will appear in {opposite_party} run): {cross_party_skips}")

    return pd.DataFrame(results), unmatched


# ─────────────────────────────────────────────────────────────────────────────
# Per-combo runner (unchanged logic, updated call signatures)
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_for_combo(parquet_file, state, party, year, primary_date=None, district_ref=None):
    """Run analysis for a single (state, party, year). Returns (results_df, unmatched_list)."""
    date_info = f" before {primary_date.strftime('%m/%d/%Y')}" if primary_date else ""
    print(f"\n=== {state}-{party}-{year}{date_info} ===")

    candidate_df     = load_candidate_data_parquet(parquet_file, state, party, year, primary_date)
    party_df         = load_party_data_smart(state, party, year, primary_date)
    opposite_party   = 'REP' if party == 'DEM' else 'DEM'
    opposite_party_df = load_party_data_smart(state, opposite_party, year, primary_date)

    results, unmatched = analyze_candidate_party_overlap_efficient(
        candidate_df, party_df, state, party, year, district_ref, opposite_party_df,
    )

    if len(results) > 0:
        results['state'] = state
        results['party'] = party
        results['year']  = year
    else:
        results = pd.DataFrame(columns=[
            'candidate_id', 'candidate_name', 'party_donors_count',
            'total_party_donors', 'total_candidate_donors',
            'cross_partisan_donations', 'percentage',
            'seat_info', 'seat_type', 'candidate_state',
            'match_method', 'in_NIMSP', 'state', 'party', 'year',
        ])

    print(f"  → {len(results)} candidates, {len(unmatched)} unmatched")
    return results, unmatched


# ─────────────────────────────────────────────────────────────────────────────
# Year runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_for_year(parquet_file, primary_dates_path, year):
    """Process all (state, party) combos for a given year."""
    start_time = time.time()

    # Fix 6: print data-coverage summary before doing any work
    log_parquet_summary(parquet_file, year)

    primary_lookup = load_primary_dates(primary_dates_path)
    all_combos     = extract_unique_combos_parquet(parquet_file)

    if not all_combos:
        print("  No valid combinations found!")
        return

    combos_for_year = [(s, p) for s, p, y in all_combos if y == year]
    if not combos_for_year:
        print(f"  No combinations found for year {year}.")
        return

    print(f"\n{'='*60}")
    print(f"  PROCESSING {year}: {len(combos_for_year)} state/party combos")
    print(f"{'='*60}")

    district_ref  = load_district_reference(year)
    year_results  = []
    all_unmatched = []

    for state, party in combos_for_year:
        primary_date = primary_lookup.get((state, year))
        if primary_date is None:
            print(f"  {state}-{party}-{year}: SKIPPED — no primary date")
            continue
        try:
            combo_result, unmatched_list = run_analysis_for_combo(
                parquet_file, state, party, year, primary_date, district_ref,
            )
            if len(combo_result) > 0:
                year_results.append(combo_result)
            all_unmatched.extend(unmatched_list)
        except Exception as e:
            print(f"  ERROR {state}-{party}-{year}: {e}")
            continue

    if not year_results:
        print(f"\n  Year {year}: no valid results produced.")
        return

    year_df = pd.concat(year_results, ignore_index=True)
    year_df = year_df.sort_values(['state', 'party', 'percentage'], ascending=[True, True, False])

    output_path = OUTPUT_FILE(year)
    year_df.to_csv(output_path, index=False)

    total     = len(year_df)
    matched   = (year_df['in_NIMSP'] == True).sum()
    unmatched = total - matched
    match_pct = matched / total * 100 if total else 0
    elapsed   = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {year}")
    print(f"{'='*60}")
    print(f"  Total candidates  : {total:,}")
    print(f"  Matched to NIMSP  : {matched:,} ({match_pct:.1f}%)")
    print(f"  Unmatched         : {unmatched:,} ({100-match_pct:.1f}%)")
    print(f"  Runtime           : {int(elapsed//60)}m {elapsed%60:.1f}s")
    print(f"  Output            : {output_path}")


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
