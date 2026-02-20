import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import re
import time
from difflib import SequenceMatcher
from utils.paths import (
    DIME_DATA_DIR,
    NIMSP_PARTY_DATA,
    PRIMARY_DATES_FILE,
    UPPER_HOUSE_FILE,
    LOWER_HOUSE_FILE,
    OUTPUT_FILE,
    ensure_output_dir_exists,
)

def load_primary_dates(file_path):
    """
    Load primary dates from CSV and return lookup dictionary.
    Returns: dict {(state, year): datetime}
    """
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
        'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    df = pd.read_csv(file_path)
    primary_lookup = {}
    
    for _, row in df.iterrows():
        raw_state = str(row['state']).strip()
        state = state_to_abbrev.get(raw_state, raw_state.upper())
        year = int(row['year'])
        date_str = row['leg_date']
        
        try:
            primary_date = pd.to_datetime(date_str)
            primary_lookup[(state, year)] = primary_date
        except:
            print(f"Warning: Could not parse date for {state}-{year}: {date_str}")
    
    print(f"Loaded {len(primary_lookup)} primary dates")
    return primary_lookup

def normalize_name(name):
    """Normalize a candidate name for matching. Returns 'LAST FIRST [MIDDLE ...]'."""
    if pd.isna(name):
        return ""
    text = str(name).upper()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in {"JR", "SR", "II", "III", "IV"}]
    return " ".join(tokens)


def extract_last_name(name):
    """Extract normalized last name from a candidate name string.
    Handles both 'LAST, FIRST' and 'FIRST LAST' formats."""
    if pd.isna(name):
        return ""
    text = str(name).upper().strip()
    if "," in text:
        last = text.split(",", 1)[0]
        return normalize_name(last)
    name_norm = normalize_name(text)
    parts = name_norm.split()
    return parts[-1] if parts else ""


def match_candidate_name(dime_name_norm, ref_candidates, fuzzy_threshold=0.80):
    """
    Match a DIME candidate name against reference candidates using a
    tiered strategy. ref_candidates is a list of dicts with keys:
    'name_norm', 'last_norm', 'district', 'candidate_index', 'candidate_state'.

    Strategy:
      1. Exact full-name match
      2. Token-subset match (DIME tokens ⊆ ref tokens or vice versa)
         Catches: 'BERKOWITZ ETHAN' vs 'BERKOWITZ ETHAN A'
      3. Last-name match + fuzzy scoring within the same-last-name group
         Catches: 'HIGGINS PATTI' vs 'HIGGINS PATRICIA C'

    Returns: (matched_ref_dict, match_method) or (None, None)
    """
    dime_tokens = set(dime_name_norm.split())
    if not dime_tokens:
        return None, None

    dime_last = dime_name_norm.split()[0] if dime_name_norm else ""

    # --- Tier 1: exact ---
    for ref in ref_candidates:
        if ref["name_norm"] == dime_name_norm:
            return ref, "exact"

    # --- Tier 2: token-subset (order-independent) ---
    # If every DIME token appears in the ref name (or vice versa), it's the
    # same person with a missing/extra middle initial.
    subset_matches = []
    for ref in ref_candidates:
        ref_tokens = set(ref["name_norm"].split())
        if dime_tokens <= ref_tokens or ref_tokens <= dime_tokens:
            subset_matches.append(ref)
    if len(subset_matches) == 1:
        return subset_matches[0], "token_subset"

    # --- Tier 3: last-name + fuzzy within same-last-name group ---
    same_last = [r for r in ref_candidates if r["last_norm"] == dime_last]
    if not same_last:
        return None, None
    if len(same_last) == 1:
        # Only one candidate with this last name — high confidence
        score = SequenceMatcher(None, dime_name_norm, same_last[0]["name_norm"]).ratio()
        if score >= fuzzy_threshold:
            return same_last[0], f"fuzzy_last({score:.2f})"
        return None, None

    # Multiple same-last-name candidates — pick best fuzzy using full-name score only
    best_score = 0.0
    best_ref = None
    for ref in same_last:
        full_score = SequenceMatcher(None, dime_name_norm, ref["name_norm"]).ratio()
        if full_score > best_score:
            best_score = full_score
            best_ref = ref
    if best_score >= fuzzy_threshold:
        return best_ref, f"fuzzy_best({best_score:.2f})"
    return None, None

def normalize_district(office_sought):
    if pd.isna(office_sought):
        return "UNKNOWN"
    text = str(office_sought).upper()
    if "AT LARGE" in text or "AT-LARGE" in text:
        return "AL"
    cleaned = re.sub(r"^(SENATE|HOUSE|ASSEMBLY)\s+DISTRICT\s+", "", text).strip()
    if cleaned in {"AL", "AT LARGE", "AT-LARGE"}:
        return "AL"
    has_alpha = re.search(r"[A-Z]", cleaned) is not None
    has_digit = re.search(r"\d", cleaned) is not None
    if has_alpha and has_digit:
        cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
        return cleaned or "UNKNOWN"
    if has_digit:
        match = re.search(r"(\d+)", cleaned)
        return str(int(match.group(1)))
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    return cleaned or "UNKNOWN"

def map_election_status(election_status):
    """
    Map Election_Status to candidate outcome code.

    General-stage outcomes (candidate survived primary):
      W  - Won general election
      P  - Lost general election (implies won primary)
      H  - Withdrew during general
      DG - Disqualified during general
      XG - Deceased during general
      DW - Default winner (unopposed in general)
      TG - Tied in general

    Runoff outcomes:
      WR  - Won general runoff
      LR  - Lost general runoff
      WPR - Won primary runoff
      LPR - Lost primary runoff
      XPR - Deceased during primary runoff

    Primary-stage outcomes:
      L   - Lost primary
      LT  - Lost top-two primary
      LC  - Lost convention
      WP  - Won primary (no general outcome recorded)
      HP  - Withdrew during primary
      DP  - Disqualified during primary
    """
    status = str(election_status).strip().upper() if pd.notna(election_status) else ""

    STATUS_MAP = {
        # General-stage
        "WON-GENERAL":           "W",
        "LOST-GENERAL":          "P",
        "WITHDREW-GENERAL":      "H",
        "DISQUALIFIED-GENERAL":  "DG",
        "DECEASED-GENERAL":      "XG",
        "DEFAULT WINNER-GENERAL": "DW",
        "TIED-GENERAL":          "TG",
        # General runoff
        "WON-GENERAL RUNOFF":    "WR",
        "LOST-GENERAL RUNOFF":   "LR",
        # Primary-stage
        "LOST-PRIMARY":          "L",
        "LOST-TOP TWO PRIMARY":  "LT",
        "LOST-CONVENTION":       "LC",
        "WON-PRIMARY":           "WP",
        "WITHDREW-PRIMARY":      "HP",
        "DISQUALIFIED-PRIMARY":  "DP",
        # Primary runoff
        "WON-PRIMARY RUNOFF":    "WPR",
        "LOST-PRIMARY RUNOFF":   "LPR",
        "DECEASED-PRIMARY RUNOFF": "XPR",
    }

    code = STATUS_MAP.get(status)
    if code is not None:
        return code

    if status:
        print(f"    WARNING: Unmapped election status: '{status}'")
    return "UNKNOWN"

def load_district_reference(year, upper_path=None, lower_path=None):
    party_map = {
        "DEMOCRATIC": "DEM",
        "REPUBLICAN": "REP",
    }

    if upper_path is None:
        upper_path = UPPER_HOUSE_FILE(year)
    if lower_path is None:
        lower_path = LOWER_HOUSE_FILE(year)

    frames = []
    for path, house in [(upper_path, "U"), (lower_path, "L")]:
        from pathlib import Path
        p = Path(str(path))
        # Handle .csv.csv naming issue (some files have double extension)
        if not p.exists():
            alt = p.with_suffix(".csv.csv")
            if alt.exists():
                print(f"  Note: using {alt.name} (double extension)")
                p = alt
        try:
            df = pd.read_csv(p, low_memory=False)
        except FileNotFoundError:
            print(f"District reference missing: {p}")
            continue
        df = df.rename(columns=str.strip)
        df["house"] = house
        df["state"] = df["Election_Jurisdiction"].astype(str).str.upper().str.strip()
        df["party_raw"] = df["General_Party"].astype(str).str.upper().str.strip()
        df["party"] = df["party_raw"].map(party_map)
        df["candidate_name"] = df["Candidate"].astype(str).str.strip()
        df["candidate_name_norm"] = df["candidate_name"].map(normalize_name)
        df["district"] = df["Office_Sought"].map(normalize_district)
        df["total_amount"] = pd.to_numeric(df["Total_$"], errors="coerce").fillna(0.0)
        df = df[df["party"].isin(["DEM", "REP"])].copy()
        frames.append(df)

    if not frames:
        return {}

    merged = pd.concat(frames, ignore_index=True)
    sort_cols = ["state", "party", "house", "district", "total_amount", "candidate_name"]
    merged = merged.sort_values(sort_cols, ascending=[True, True, True, True, False, True]).copy()
    merged["candidate_index"] = (
        merged.groupby(["state", "party", "house", "district"])["candidate_name"]
        .cumcount()
        .add(1)
        .map(lambda x: str(int(x)).zfill(2))
    )

    # Add election status mapping
    merged["candidate_state"] = merged["Election_Status"].map(map_election_status)
    merged["candidate_last_norm"] = merged["candidate_name"].map(extract_last_name)

    # Build lookup keyed by (state, party, house) → list of candidate dicts.
    # This supports the tiered name-matching strategy and avoids the old
    # single-key-overwrite problem.
    group_lookup = {}  # (state, party, house) → [ref_candidate_dicts]
    for _, row in merged.iterrows():
        group_key = (row["state"], row["party"], row["house"])
        entry = {
            "name_norm": row["candidate_name_norm"],
            "last_norm": row["candidate_last_norm"],
            "district": row["district"],
            "candidate_index": row["candidate_index"],
            "candidate_state": row["candidate_state"],
        }
        group_lookup.setdefault(group_key, []).append(entry)
    return group_lookup

def load_candidate_data_chunked(file_path, state, party, year, primary_date=None, chunksize=10000):
    """
    Memory-efficient: Load candidate data in chunks and filter immediately.
    Only keeps relevant records for the specific state-party-year.
    """
    party_code = '100' if party == 'DEM' else '200'
    filtered_chunks = []
    
    print(f"Loading candidate data in chunks of {chunksize}...")
    
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, low_memory=False, chunksize=chunksize)):
        # Filter this chunk immediately to save memory
        filters = (
            (chunk['cycle'] == year) &
            (chunk['recipient.state'] == state) &
            (chunk['recipient.party'] == party_code) &
            (chunk['seat'].isin(['state:upper', 'state:lower'])) &
            (chunk['recipient.type'] == 'CAND')
        )
        
        # Add date filter if primary date provided
        if primary_date is not None:
            chunk['date_parsed'] = pd.to_datetime(chunk['date'], errors='coerce')
            filters = filters & (chunk['date_parsed'] < primary_date) & (chunk['date_parsed'].notna())
        
        filtered_chunk = chunk[filters]
        
        if len(filtered_chunk) > 0:
            filtered_chunks.append(filtered_chunk)
        
        if chunk_num % 10 == 0:
            print(f"Processed {(chunk_num + 1) * chunksize} rows...")
    
    if filtered_chunks:
        result = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Final filtered candidate data: {len(result)} rows")
        return result
    else:
        return pd.DataFrame()

def load_party_data_chunked(file_path, state, party, year, primary_date=None, chunksize=10000):
    """
    Memory-efficient: Load party data in chunks and filter immediately.
    Only keeps relevant records for the specific state-party-year.
    Filters contributions before primary date if provided.
    """
    party_name = 'Democratic' if party == 'DEM' else 'Republican'
    filtered_chunks = []
    
    print(f"Loading party data in chunks of {chunksize}...")
    
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, low_memory=False, chunksize=chunksize)):
        # Filter this chunk immediately to save memory
        filters = (
            (chunk['ElectionYear'] == year) &
            (chunk['SAT_State'] == state) &
            (chunk['FSPC_PartyType'] == party_name)
        )
        
        # Add date filter if primary date provided
        if primary_date is not None:
            chunk['date_parsed'] = pd.to_datetime(chunk['CFS_Date'], errors='coerce')
            filters = filters & (chunk['date_parsed'] < primary_date) & (chunk['date_parsed'].notna())
        
        filtered_chunk = chunk[filters]
        
        if len(filtered_chunk) > 0:
            filtered_chunks.append(filtered_chunk)
        
        if chunk_num % 10 == 0:
            print(f"Processed {(chunk_num + 1) * chunksize} rows...")
    
    if filtered_chunks:
        result = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Final filtered party data: {len(result)} rows")
        return result
    else:
        return pd.DataFrame()

def extract_unique_combos(candidate_file_path, chunksize=10000, max_rows=100000):
    """
    Extract all unique (state, party, year) combinations from candidate data.
    Returns list of tuples: [(state, party, year), ...]
    """
    print("Extracting unique state/party/year combinations...")
    
    unique_combos = set()
    
    total_rows_processed = 0
    for chunk_num, chunk in enumerate(pd.read_csv(candidate_file_path, low_memory=False, chunksize=chunksize)):
        if total_rows_processed >= max_rows:
            print(f"Reached max_rows limit ({max_rows}). Stopping combo extraction.")
            break
        
        # Trim chunk if it would exceed max_rows
        if total_rows_processed + len(chunk) > max_rows:
            chunk = chunk.iloc[:max_rows - total_rows_processed]
        # Filter to state legislature candidates only
        filtered_chunk = chunk[
            (chunk['seat'].isin(['state:upper', 'state:lower'])) &
            (chunk['recipient.type'] == 'CAND') &
            (chunk['recipient.party'].isin(['100', '200']))  # DEM/REP only
        ]
        
        if len(filtered_chunk) > 0:
            # Extract combinations
            for _, row in filtered_chunk[['recipient.state', 'recipient.party', 'cycle']].drop_duplicates().iterrows():
                state = row['recipient.state']
                party_code = row['recipient.party']
                year = row['cycle']
                
                # Convert party code to name
                party = 'DEM' if party_code == '100' else 'REP'
                
                if pd.notna(state) and pd.notna(year):
                    unique_combos.add((state, party, int(year)))
        
        total_rows_processed += len(chunk)
        if chunk_num % 10 == 0:
            print(f"Processed {total_rows_processed} rows, found {len(unique_combos)} unique combos so far...")
    
    combos_list = sorted(list(unique_combos))
    print(f"\nFound {len(combos_list)} unique state/party/year combinations")
    return combos_list

def load_candidate_data(file_path, nrows=None):
    """
    Load DIME candidate donation data.
    Expected columns: cycle, contributor.state, recipient.party, recipient.type, 
                     recipient.state, seat, bonica.cid, bonica.rid, recipient.name, contributor.name
    
    Parameters:
    - file_path: Path to the CSV file
    - nrows: Number of rows to read (default None for all rows, or specify for testing)
    """
    df = pd.read_csv(file_path, low_memory=False, nrows=nrows)
    print(f"Loaded {len(df)} rows from candidate data")
    print(f"Candidate data columns: {list(df.columns)}")
    return df

def load_party_data(file_path, nrows=1000):
    """
    Load institutional/party donor data.
    Expected columns: ElectionYear, SAT_State, FSPC_PartyType, Name, CFS_Amount
    
    Parameters:
    - file_path: Path to the CSV file
    - nrows: Number of rows to read (default 1000 for testing)
    """
    df = pd.read_csv(file_path, low_memory=False, nrows=nrows)
    print(f"Loaded {len(df)} rows from party data")
    print(f"Party data columns: {list(df.columns)}")
    return df

def filter_state_party_year_candidates(candidate_df, state, party, year):
    """
    Filter candidates for specific state-party-year combination.
    Only includes state legislature candidates (state:upper/state:lower).
    
    Parameters:
    - candidate_df: DIME candidate data
    - state: Two-letter state code (e.g., 'CA')
    - party: Party code ('DEM' or 'REP')
    - year: Election year
    
    Returns filtered DataFrame
    """
    # Map party names to codes (they're stored as strings)
    party_code = '100' if party == 'DEM' else '200'
    
    filtered = candidate_df[
        (candidate_df['cycle'] == year) &
        (candidate_df['recipient.state'] == state) &
        (candidate_df['recipient.party'] == party_code) &
        (candidate_df['seat'].isin(['state:upper', 'state:lower'])) &
        (candidate_df['recipient.type'] == 'CAND')
    ]
    
    return filtered

def filter_state_party_year_donors(party_df, state, party, year):
    """
    Filter party donors for specific state-party-year combination.
    
    Parameters:
    - party_df: Institutional/party donor data
    - state: Two-letter state code (e.g., 'CA')
    - party: Party name ('Democratic' or 'Republican')
    - year: Election year
    
    Returns filtered DataFrame with unique donors
    """
    party_name = 'Democratic' if party == 'DEM' else 'Republican'
    
    filtered = party_df[
        (party_df['ElectionYear'] == year) &
        (party_df['SAT_State'] == state) &
        (party_df['FSPC_PartyType'] == party_name)
    ]
    
    return filtered

def analyze_candidate_party_overlap(candidate_df, party_df, state, party, year):
    """
    Count total donors per candidate AND party donor overlap for state-party-year combination.
    
    Returns DataFrame with columns:
    - candidate_id: CA-DEM-2000-H-DD-CC format
    - candidate_name: Name of candidate
    - party_donors_count: Number of party donors who also gave to candidate
    - total_party_donors: Total unique donors to state party
    - total_candidate_donors: Total unique donors to the candidate
    - percentage: Percentage (party_donors_count / total_party_donors * 100)
    """
    
    # Filter candidates and party donors
    candidates = filter_state_party_year_candidates(candidate_df, state, party, year)
    party_donors = filter_state_party_year_donors(party_df, state, party, year)
    
    print(f"\nAnalyzing {state}-{party}-{year}:")
    print(f"Candidates found: {len(candidates)}")
    print(f"Party donors found: {len(party_donors)}")
    
    if len(candidates) == 0:
        print("No candidates found!")
        return pd.DataFrame()
    
    if len(party_donors) == 0:
        print("No party donors found!")
        return pd.DataFrame()
    
    # Get unique party donor names (standardize for matching)
    party_donors['donor_name_clean'] = party_donors['Name'].str.upper().str.strip()
    unique_party_donors = set(party_donors['donor_name_clean'].dropna().unique())
    total_party_donors = len(unique_party_donors)
    
    # Get unique candidates
    unique_candidates = candidates['bonica.rid'].unique()
    
    results = []
    
    for candidate_rid in unique_candidates:
        # Get all donations for this specific candidate
        candidate_donations = candidates[candidates['bonica.rid'] == candidate_rid]
        
        # Get candidate info first
        candidate_name = candidate_donations['recipient.name'].iloc[0] if not candidate_donations.empty else 'Unknown'
        
        # Count total unique donors to this candidate
        total_candidate_donors = candidate_donations['bonica.cid'].nunique()
        
        # Clean candidate donor names for party matching
        candidate_donations['donor_name_clean'] = candidate_donations['contributor.name'].str.upper().str.strip()
        candidate_donors = set(candidate_donations['donor_name_clean'].dropna().unique())
        
        # Find overlap between candidate donors and party donors by name
        overlap_donors = candidate_donors.intersection(unique_party_donors)
        party_donors_count = len(overlap_donors)
        
        # Calculate percentage of party donors who gave to this candidate
        percentage = (party_donors_count / total_party_donors * 100) if total_party_donors > 0 else 0
        
        seat_type = candidate_donations['seat'].iloc[0] if not candidate_donations.empty else 'Unknown'
        
        # Extract house and district info
        if seat_type == 'state:upper':
            house = 'U'  # Upper house (Senate)
            district = 'AL'  # At-large for senate
        elif seat_type == 'state:lower':
            house = 'L'  # Lower house
            district = '01'  # Default - you might need to extract actual district
        else:
            house = 'X'  # Unknown
            district = '00'
        
        # Create candidate ID in format SS-PPP-YYYY-H-DD-CC
        candidate_count = len([r for r in results if r['seat_info'] == f"{house}-{district}"]) + 1
        candidate_id = f"{state}-{party}-{year}-{house}-{district}-{candidate_count:02d}"
        
        results.append({
            'candidate_id': candidate_id,
            'candidate_name': candidate_name,
            'party_donors_count': party_donors_count,
            'total_party_donors': total_party_donors,
            'total_candidate_donors': total_candidate_donors,
            'percentage': round(percentage, 2),
            'seat_info': f"{house}-{district}",
            'seat_type': seat_type
        })
    
    return pd.DataFrame(results)
    
def run_analysis_for_combo(candidate_file_path, party_file_path, state, party, year, primary_date=None, district_ref=None):
    """
    Memory-efficient analysis for a single state/party/year combination.
    Returns tuple: (results_df, unmatched_list)
    Filters contributions before primary date if provided.
    """
    date_info = f" (before {primary_date.strftime('%m/%d/%Y')})" if primary_date else ""
    print(f"=== ANALYZING {state}-{party}-{year}{date_info} ===")
    
    # Load data efficiently (filter while loading)
    candidate_df = load_candidate_data_chunked(candidate_file_path, state, party, year, primary_date)
    party_df = load_party_data_chunked(party_file_path, state, party, year, primary_date)
    
    # Load opposite party donors for cross-partisan detection
    opposite_party = 'REP' if party == 'DEM' else 'DEM'
    opposite_party_df = load_party_data_chunked(party_file_path, state, opposite_party, year, primary_date)
    
    # Analyze - returns tuple (results, unmatched)
    results, unmatched = analyze_candidate_party_overlap_efficient(
        candidate_df, party_df, state, party, year, district_ref, opposite_party_df
    )
    
    print(f"Analysis for {state}-{party}-{year}: {len(results)} candidates found, {len(unmatched)} unmatched")
    
    # Add combo identifier columns
    if len(results) > 0:
        results['state'] = state
        results['party'] = party
        results['year'] = year
    else:
        # Return empty DataFrame with proper columns for concatenation
        results = pd.DataFrame(columns=[
            'candidate_id', 'candidate_name', 'party_donors_count', 
            'total_party_donors', 'total_candidate_donors', 
            'cross_partisan_donations', 'percentage',
            'seat_info', 'seat_type', 'candidate_state', 'match_method', 'in_NIMSP',
            'state', 'party', 'year'
        ])
    
    return results, unmatched

def analyze_candidate_party_overlap_efficient(candidate_df, party_df, state, party, year, district_ref=None, opposite_party_df=None):
    """
    Efficient version that works with pre-filtered data.
    Matches candidates to NIMSP reference. Computes cross-partisan donations
    by checking candidate donors against the opposite party's NIMSP donor list.
    
    Returns a tuple: (results_df, unmatched_list)
    unmatched_list contains dicts with 'candidate_name', 'house', 'state', 'party', 'year'
    """
    print(f"\nAnalyzing {state}-{party}-{year}:")
    print(f"Candidates found: {len(candidate_df)}")
    print(f"Party donors found: {len(party_df)}")

    if len(candidate_df) == 0:
        print("No candidates found!")
        return pd.DataFrame(), []

    if len(party_df) == 0:
        print("No party donors found!")
        return pd.DataFrame(), []

    # Get unique party donor names (pre-process once)
    party_df['donor_name_clean'] = party_df['Name'].str.upper().str.strip()
    unique_party_donors = set(party_df['donor_name_clean'].dropna().unique())
    total_party_donors = len(unique_party_donors)

    # Get unique opposite-party donor names for cross-partisan detection
    unique_opposite_donors = set()
    if opposite_party_df is not None and len(opposite_party_df) > 0:
        opposite_party_df['donor_name_clean'] = opposite_party_df['Name'].str.upper().str.strip()
        unique_opposite_donors = set(opposite_party_df['donor_name_clean'].dropna().unique())
        print(f"Opposite party donors loaded: {len(unique_opposite_donors)}")

    # Get unique candidates
    unique_candidates = candidate_df['bonica.rid'].unique()

    results = []
    unmatched = []

    for candidate_rid in unique_candidates:
        # Get all donations for this specific candidate
        candidate_donations = candidate_df[candidate_df['bonica.rid'] == candidate_rid]

        # Get candidate info
        candidate_name = candidate_donations['recipient.name'].iloc[0] if not candidate_donations.empty else 'Unknown'

        # --- Match against reference data FIRST to determine TRUE party ---
        name_norm = normalize_name(candidate_name)
        seat_type = candidate_donations['seat'].iloc[0] if not candidate_donations.empty else 'Unknown'
        
        # Extract house
        if seat_type == 'state:upper':
            house = 'U'
            district = 'AL'
        elif seat_type == 'state:lower':
            house = 'L'
            district = '01'
        else:
            house = 'X'
            district = '00'

        candidate_index = None
        candidate_state = None
        match_method = "fallback"
        in_nimsp = False
        
        if district_ref:
            group_key = (state, party, house)
            ref_candidates = district_ref.get(group_key, [])
            if ref_candidates:
                matched_ref, method = match_candidate_name(name_norm, ref_candidates)
                if matched_ref:
                    district = matched_ref["district"]
                    candidate_index = matched_ref["candidate_index"]
                    candidate_state = matched_ref.get("candidate_state")
                    match_method = method
                    in_nimsp = True
            
            # If not matched to current party, check opposite party to avoid duplicates
            if not in_nimsp:
                opposite_party = 'REP' if party == 'DEM' else 'DEM'
                opposite_group_key = (state, opposite_party, house)
                opposite_ref_candidates = district_ref.get(opposite_group_key, [])
                if opposite_ref_candidates:
                    opposite_matched, _ = match_candidate_name(name_norm, opposite_ref_candidates)
                    if opposite_matched:
                        # This candidate belongs to opposite party - skip to avoid duplicate
                        continue

        # Count total unique donors to this candidate
        total_candidate_donors = candidate_donations['bonica.cid'].nunique()

        # Clean candidate donor names
        candidate_donations_copy = candidate_donations.copy()
        candidate_donations_copy['donor_name_clean'] = candidate_donations_copy['contributor.name'].str.upper().str.strip()
        candidate_donors = set(candidate_donations_copy['donor_name_clean'].dropna().unique())

        # Find overlap between candidate donors and same-party donors
        overlap_donors = candidate_donors.intersection(unique_party_donors)
        party_donors_count = len(overlap_donors)

        # Cross-partisan: candidate donors who also gave to the OPPOSITE party committee
        cross_partisan_count = len(candidate_donors.intersection(unique_opposite_donors))

        # Calculate percentage of party donors who gave to this candidate
        percentage = (party_donors_count / total_party_donors * 100) if total_party_donors > 0 else 0

        if candidate_index is None:
            candidate_count = len([r for r in results if r['seat_info'] == f"{house}-{district}"]) + 1
            candidate_index = f"{candidate_count:02d}"
        candidate_id = f"{state}-{party}-{year}-{house}-{district}-{candidate_index}"

        results.append({
            'candidate_id': candidate_id,
            'candidate_name': candidate_name,
            'party_donors_count': party_donors_count,
            'total_party_donors': total_party_donors,
            'total_candidate_donors': total_candidate_donors,
            'cross_partisan_donations': cross_partisan_count,
            'percentage': round(percentage, 2),
            'seat_info': f"{house}-{district}",
            'seat_type': seat_type,
            'candidate_state': candidate_state,
            'match_method': match_method,
            'in_NIMSP': in_nimsp
        })
        
        # Track unmatched candidates
        if not in_nimsp:
            unmatched.append({
                'candidate_name': candidate_name,
                'house': house,
                'state': state,
                'party': party,
                'year': year
            })

    return pd.DataFrame(results), unmatched

    return pd.DataFrame(results)

def run_analysis_for_year(candidate_file_path, party_file_path, primary_dates_path, year):
    """
    Main function: Extract all unique combos and process a single year.
    Outputs one CSV for the target year with all state/party combinations.
    Filters contributions to before primary date only.
    """
    start_time = time.time()
    
    # Load primary dates lookup
    primary_lookup = load_primary_dates(primary_dates_path)
    
    # Extract all unique combinations
    all_combos = extract_unique_combos(candidate_file_path)
    
    if not all_combos:
        print("No valid combinations found!")
        return

    combos_for_year = [(state, party) for state, party, combo_year in all_combos if combo_year == year]
    if not combos_for_year:
        print(f"No valid combinations found for year {year}.")
        return

    print(f"\n{'='*60}")
    print(f"PROCESSING YEAR {year}: {len(combos_for_year)} state/party combinations")
    print(f"{'='*60}")

    district_ref = load_district_reference(year)

    year_results = []
    all_unmatched = []

    for state, party in combos_for_year:
        try:
            # Get primary date for this state/year combo
            primary_date = primary_lookup.get((state, year))

            if primary_date is None:
                print(f"  {state}-{party}-{year}: SKIPPED - No primary date found")
                continue

            combo_result, unmatched_list = run_analysis_for_combo(
                candidate_file_path,
                party_file_path,
                state,
                party,
                year,
                primary_date,
                district_ref,
            )
            if len(combo_result) > 0:
                year_results.append(combo_result)
            all_unmatched.extend(unmatched_list)
        except Exception as e:
            print(f"  ERROR processing {state}-{party}-{year}: {str(e)}")
            continue

    if year_results:
        year_df = pd.concat(year_results, ignore_index=True)

        # Sort by state, then party for consistent output
        year_df = year_df.sort_values(['state', 'party', 'percentage'], ascending=[True, True, False])

        # Export to CSV
        output_path = OUTPUT_FILE(year)
        year_df.to_csv(output_path, index=False)

        # Generate matching summary
        total_candidates = len(year_df)
        matched_candidates = len(year_df[year_df['in_NIMSP'] == True])
        unmatched_candidates = total_candidates - matched_candidates
        match_percentage = (matched_candidates / total_candidates * 100) if total_candidates > 0 else 0
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60

        print(f"\n{'='*60}")
        print(f"MATCHING SUMMARY FOR YEAR {year}")
        print(f"{'='*60}")
        print(f"Total candidates found: {total_candidates}")
        print(f"Successfully matched to NIMSP: {matched_candidates} ({match_percentage:.1f}%)")
        print(f"Unmatched candidates: {unmatched_candidates} ({100-match_percentage:.1f}%)")
        print(f"Runtime: {minutes}m {seconds:.1f}s")
        print(f"\nResults saved to: {output_path}")
    else:
        print(f"\nYear {year}: No valid results found")


# Prompt for target years
TARGET_YEARS = input("Enter target years (comma-separated, e.g. 2000, 2002, 2004): ")

def parse_target_years(years_input):
    """
    Parse TARGET_YEARS string into list of integers.
    Accepts: "2000" or "2000, 2002, 2004"
    """
    if isinstance(years_input, list):
        return years_input
    
    years_str = str(years_input).strip()
    years = [int(y.strip()) for y in years_str.split(',')]
    return years

if __name__ == "__main__":
    # Process multiple years sequentially
    target_years = parse_target_years(TARGET_YEARS)
    
    # Ensure output directory exists
    ensure_output_dir_exists()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING YEARS: {target_years}")
    print(f"{'='*60}")
    print(f"Using NIMSP data: {NIMSP_PARTY_DATA}")
    print(f"Using primary dates: {PRIMARY_DATES_FILE}\n")
    
    for year in target_years:
        candidate_data_path = DIME_DATA_DIR(year)
        print(f"\n{'='*60}")
        print(f"STARTING YEAR {year}")
        print(f"{'='*60}")
        print(f"Using DIME data: {candidate_data_path}\n")
        
        try:
            run_analysis_for_year(candidate_data_path, str(NIMSP_PARTY_DATA), str(PRIMARY_DATES_FILE), year)
        except FileNotFoundError as e:
            print(f"ERROR: Missing required file for year {year}: {e}")
            print(f"Skipping year {year}...\n")
            continue
        except Exception as e:
            print(f"ERROR processing year {year}: {e}")
            print(f"Skipping year {year}...\n")
            continue
    
    print(f"\n{'='*60}")
    print("ALL PROCESSING COMPLETE")
    print(f"{'='*60}")
