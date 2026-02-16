import argparse
import re

import pandas as pd


PARTY_MAP = {
    "DEMOCRATIC": "DEM",
    "REPUBLICAN": "REP",
}


def normalize_name(name):
    if pd.isna(name):
        return ""
    text = str(name).upper()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in {"JR", "SR", "II", "III", "IV"}]
    return " ".join(tokens)


def map_election_status(election_status):
    """
    Map Election_Status from upper/lower house data to candidate outcome code.
    
    Returns:
    - W: Won general election
    - P: Won primary (lost general election)
    - H: Withdrew during general
    - L: Lost during primary
    """
    status = str(election_status).strip().upper() if pd.notna(election_status) else ""
    
    if status == "WON-GENERAL":
        return "W"
    elif status == "LOST-GENERAL":
        return "P"
    elif status == "WITHDREW-GENERAL":
        return "H"
    elif status == "LOST-PRIMARY":
        return "L"
    else:
        return "UNKNOWN"


def load_upper_lower(path, house_code):
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns=str.strip)
    df["house"] = house_code
    df["state"] = df["Election_Jurisdiction"].astype(str).str.upper().str.strip()
    df["party_raw"] = df["General_Party"].astype(str).str.upper().str.strip()
    df["party"] = df["party_raw"].map(PARTY_MAP)
    df["candidate_name"] = df["Candidate"].astype(str).str.strip()
    df["candidate_name_norm"] = df["candidate_name"].map(normalize_name)
    df["candidate_state"] = df["Election_Status"].map(map_election_status)
    df = df[df["party"].isin(["DEM", "REP"])].copy()
    return df


def build_status_lookup(df):
    lookup = {}
    for _, row in df.iterrows():
        key = (row["state"], row["party"], row["house"], row["candidate_name_norm"])
        lookup.setdefault(key, []).append(row["candidate_state"])
    return lookup


def main():
    parser = argparse.ArgumentParser(
        description="Add candidate electoral outcome to analysis CSV"
    )
    parser.add_argument("--analysis", default="outputs/2000-analysis-updated.csv")
    parser.add_argument("--upper", default="upperlower/2000_upper.csv")
    parser.add_argument("--lower", default="upperlower/2000_lower.csv")
    parser.add_argument("--output", default="outputs/updated-winners.csv")
    args = parser.parse_args()

    upper = load_upper_lower(args.upper, "U")
    lower = load_upper_lower(args.lower, "L")
    candidates = pd.concat([upper, lower], ignore_index=True)

    status_lookup = build_status_lookup(candidates)

    analysis = pd.read_csv(args.analysis, low_memory=False)
    analysis = analysis.rename(columns=str.strip)
    analysis["state"] = analysis["state"].astype(str).str.upper().str.strip()
    analysis["party_raw"] = analysis["party"].astype(str).str.upper().str.strip()
    analysis["party"] = analysis["party_raw"].map(PARTY_MAP)
    analysis["house"] = analysis["seat_type"].map(
        {"state:upper": "U", "state:lower": "L"}
    )
    analysis["candidate_name_norm"] = analysis["candidate_name"].map(normalize_name)

    analysis["candidate_state"] = None

    matched = 0
    for idx, row in analysis.iterrows():
        key = (row["state"], row["party"], row["house"], row["candidate_name_norm"])
        states = status_lookup.get(key, [])
        if len(states) >= 1:
            analysis.at[idx, "candidate_state"] = states[0]
            matched += 1

    unmatched = analysis[analysis["candidate_state"].isna()].copy()
    print(f"Exact match: {matched} matched")
    print(f"Exact match: {len(unmatched)} unmatched")

    analysis = analysis.drop(columns=["party_raw", "house", "candidate_name_norm"])
    analysis.to_csv(args.output, index=False)
    print(f"Wrote updated analysis to {args.output}")


if __name__ == "__main__":
    main()
