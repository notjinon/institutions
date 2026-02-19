import argparse
import re
from difflib import SequenceMatcher

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


def extract_last_name(name):
    if pd.isna(name):
        return ""
    text = str(name).upper().strip()
    if "," in text:
        last = text.split(",", 1)[0]
        return normalize_name(last)
    name_norm = normalize_name(text)
    parts = name_norm.split()
    return parts[-1] if parts else ""


def normalize_district(office_sought):
    if pd.isna(office_sought):
        return "UNKNOWN"
    text = str(office_sought).upper()
    if "AT LARGE" in text or "AT-LARGE" in text:
        return "AL"
    cleaned = re.sub(r"^(SENATE|HOUSE|ASSEMBLY)\s+DISTRICT\s+", "", text)
    cleaned = cleaned.strip()
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


def load_upper_lower(path, house_code):
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns=str.strip)
    df["house"] = house_code
    df["state"] = df["Election_Jurisdiction"].astype(str).str.upper().str.strip()
    df["party_raw"] = df["General_Party"].astype(str).str.upper().str.strip()
    df["party"] = df["party_raw"].map(PARTY_MAP)
    df["candidate_name"] = df["Candidate"].astype(str).str.strip()
    df["candidate_name_norm"] = df["candidate_name"].map(normalize_name)
    df["candidate_last_norm"] = df["candidate_name"].map(extract_last_name)
    df["district"] = df["Office_Sought"].map(normalize_district)
    df["total_amount"] = pd.to_numeric(df["Total_$"], errors="coerce").fillna(0.0)
    df = df[df["party"].isin(["DEM", "REP"])].copy()
    return df


def build_candidate_index(df):
    sort_cols = ["state", "party", "house", "district", "total_amount", "candidate_name"]
    df = df.sort_values(sort_cols, ascending=[True, True, True, True, False, True]).copy()
    df["candidate_index"] = (
        df.groupby(["state", "party", "house", "district"])["candidate_name"]
        .cumcount()
        .add(1)
        .map(lambda x: str(int(x)).zfill(2))
    )
    df["candidate_id_new"] = (
        df["state"]
        + "-"
        + df["party"]
        + "-2000-"
        + df["house"]
        + "-"
        + df["district"]
        + "-"
        + df["candidate_index"]
    )
    return df


def build_exact_lookup(df):
    lookup = {}
    for _, row in df.iterrows():
        key = (row["state"], row["party"], row["house"], row["candidate_name_norm"])
        lookup.setdefault(key, []).append(row["candidate_id_new"])
    return lookup


def find_best_fuzzy(name_norm, last_norm, candidate_rows, threshold):
    best_score = 0.0
    best_name = None
    best_detail = None
    for cand_name, cand_last in candidate_rows:
        last_score = SequenceMatcher(None, last_norm, cand_last).ratio()
        full_score = SequenceMatcher(None, name_norm, cand_name).ratio()
        score = (0.7 * last_score) + (0.3 * full_score)
        if score > best_score:
            best_score = score
            best_name = cand_name
            best_detail = (last_score, full_score)
    if best_score >= threshold:
        return best_name, best_score, best_detail
    return None, best_score, best_detail


def main():
    parser = argparse.ArgumentParser(
        description="Update 2000-analysis candidate_id using 2000 upper/lower candidate data."
    )
    parser.add_argument("--analysis", default="outputs/2000-analysis.csv")
    parser.add_argument("--upper", default="upperlower/2000_upper.csv")
    parser.add_argument("--lower", default="upperlower/2000_lower.csv")
    parser.add_argument("--output", default="outputs/2000-analysis-updated.csv")
    parser.add_argument("--unmatched", default="outputs/2000-analysis-unmatched.csv")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.9)
    args = parser.parse_args()

    upper = load_upper_lower(args.upper, "U")
    lower = load_upper_lower(args.lower, "L")
    candidates = pd.concat([upper, lower], ignore_index=True)
    candidates = build_candidate_index(candidates)

    exact_lookup = build_exact_lookup(candidates)
    grouped_candidates = {}
    for key, group in candidates.groupby(["state", "party", "house"]):
        rows = group[["candidate_name_norm", "candidate_last_norm"]].drop_duplicates()
        grouped_candidates[key] = list(rows.itertuples(index=False, name=None))

    analysis = pd.read_csv(args.analysis, low_memory=False)
    analysis = analysis.rename(columns=str.strip)
    analysis["state"] = analysis["state"].astype(str).str.upper().str.strip()
    analysis["party"] = analysis["party"].astype(str).str.upper().str.strip()
    analysis["house"] = analysis["seat_type"].map(
        {"state:upper": "U", "state:lower": "L"}
    )
    analysis["candidate_name_norm"] = analysis["candidate_name"].map(normalize_name)
    analysis["candidate_last_norm"] = analysis["candidate_name"].map(extract_last_name)

    analysis["candidate_id_old"] = analysis["candidate_id"]
    analysis["candidate_id"] = None

    unmatched_rows = []

    for idx, row in analysis.iterrows():
        key = (row["state"], row["party"], row["house"], row["candidate_name_norm"])
        candidate_ids = exact_lookup.get(key, [])
        if len(candidate_ids) == 1:
            analysis.at[idx, "candidate_id"] = candidate_ids[0]
        else:
            unmatched_rows.append(idx)

    unmatched = analysis.loc[unmatched_rows].copy()
    print(f"Exact match: {len(analysis) - len(unmatched_rows)} matched")
    print(f"Exact match: {len(unmatched_rows)} unmatched")

    if not unmatched.empty:
        for idx, row in unmatched.iterrows():
            group_key = (row["state"], row["party"], row["house"])
            candidate_rows = grouped_candidates.get(group_key, [])
            if not candidate_rows:
                continue
            best_name, _score, _detail = find_best_fuzzy(
                row["candidate_name_norm"],
                row["candidate_last_norm"],
                candidate_rows,
                args.fuzzy_threshold,
            )
            if best_name is None:
                continue
            lookup_key = (row["state"], row["party"], row["house"], best_name)
            candidate_ids = exact_lookup.get(lookup_key, [])
            if len(candidate_ids) == 1:
                analysis.at[idx, "candidate_id"] = candidate_ids[0]

    remaining_unmatched = analysis[analysis["candidate_id"].isna()].copy()
    remaining_unmatched.to_csv(args.unmatched, index=False)
    print(f"After fuzzy: {len(analysis) - len(remaining_unmatched)} matched")
    print(f"After fuzzy: {len(remaining_unmatched)} unmatched (saved to {args.unmatched})")

    analysis["candidate_id"] = analysis["candidate_id"].fillna(analysis["candidate_id_old"])
    analysis.to_csv(args.output, index=False)
    print(f"Wrote updated analysis to {args.output}")


if __name__ == "__main__":
    main()
