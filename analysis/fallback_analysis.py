"""
Fallback Analysis Script
Analyzes records where IN_NIMSP is False, providing descriptive statistics
focused on (year, state, party) combinations to diagnose matching degradation.
"""

import pandas as pd
from pathlib import Path

# Set up paths
OUTPUTS_DIR = Path(__file__).parent / "outputs"

W    = 72
SEP  = "=" * W
SEP2 = "-" * W


def load_analysis_data():
    """Load and combine all analysis CSV files from outputs folder."""
    dfs = []
    csv_files = sorted(OUTPUTS_DIR.glob("*-analysis.csv"))

    if not csv_files:
        print(f"No analysis CSV files found in {OUTPUTS_DIR}")
        return None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"  Loaded {csv_file.name:<35} ({len(df):>5} rows)")
        except Exception as e:
            print(f"  ERROR loading {csv_file.name}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total records loaded: {len(combined_df):,}")
    return combined_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Year-level matching rates + trend
# ─────────────────────────────────────────────────────────────────────────────

def analyze_matching_by_year(df):
    print(f"\n{SEP}")
    print(f"  MATCHING RATES BY YEAR")
    print(SEP)

    year_stats = df.groupby("year").apply(
        lambda x: pd.Series({
            "total":     len(x),
            "matched":   (x["in_NIMSP"] == True).sum(),
            "unmatched": (x["in_NIMSP"] == False).sum(),
        })
    ).astype(int).sort_index()

    year_stats["match_pct"]   = (year_stats["matched"]   / year_stats["total"] * 100).round(2)
    year_stats["unmatch_pct"] = (year_stats["unmatched"] / year_stats["total"] * 100).round(2)

    prev_pct = None
    print(f"  {'YEAR':<6} {'TOTAL':>6}  {'MATCHED':>8}  {'UNMATCHED':>9}  {'UNMATCH%':>8}  TREND")
    print(f"  {SEP2}")
    for year, row in year_stats.iterrows():
        if prev_pct is None:
            trend = "  —"
        elif row["unmatch_pct"] > prev_pct + 0.5:
            trend = f"  ▲ +{row['unmatch_pct'] - prev_pct:.1f}pp  (worse)"
        elif row["unmatch_pct"] < prev_pct - 0.5:
            trend = f"  ▼ {row['unmatch_pct'] - prev_pct:.1f}pp  (better)"
        else:
            trend = "  ≈ flat"
        print(f"  {int(year):<6} {int(row['total']):>6}  {int(row['matched']):>8}  "
              f"{int(row['unmatched']):>9}  {row['unmatch_pct']:>7.2f}%{trend}")
        prev_pct = row["unmatch_pct"]

    total     = year_stats["total"].sum()
    matched   = year_stats["matched"].sum()
    unmatched = year_stats["unmatched"].sum()
    print(f"\n  Overall — total: {total:,}  |  matched: {matched:,} ({matched/total*100:.2f}%)  "
          f"|  unmatched: {unmatched:,} ({unmatched/total*100:.2f}%)")
    return year_stats


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — (year, state, party) combo breakdown
# ─────────────────────────────────────────────────────────────────────────────

def analyze_combo_breakdown(df):
    print(f"\n{SEP}")
    print(f"  YEAR–STATE–PARTY COMBO BREAKDOWN")
    print(SEP)

    grp = df.groupby(["year", "state", "party"]).agg(
        total     = ("in_NIMSP", "count"),
        matched   = ("in_NIMSP", lambda x: (x == True).sum()),
        unmatched = ("in_NIMSP", lambda x: (x == False).sum()),
        pct_med   = ("percentage", "median"),
        pct_mean  = ("percentage", "mean"),
        pct_max   = ("percentage", "max"),
    ).reset_index()
    grp["unmatch_rate"]        = (grp["unmatched"] / grp["total"] * 100).round(2)
    grp["fully_unmatched"]     = grp["matched"] == 0
    grp["partially_unmatched"] = (grp["unmatched"] > 0) & (grp["matched"] > 0)

    # ── per-year summary ──────────────────────────────────────────────────
    print(f"\n  {'YEAR':<6} {'COMBOS':>7}  {'FULLY UNMTCH':>13}  {'PARTIALLY':>10}  {'ALL MATCHED':>11}")
    print(f"  {SEP2}")
    for year, g in grp.groupby("year"):
        total_combos = len(g)
        fully        = g["fully_unmatched"].sum()
        partial      = g["partially_unmatched"].sum()
        all_match    = total_combos - fully - partial
        print(f"  {int(year):<6} {total_combos:>7}  {fully:>13}  {partial:>10}  {all_match:>11}")

    # ── top 10 worst combos by raw unmatched record count ─────────────────
    print(f"\n{SEP}")
    print(f"  TOP 10 COMBOS (STATE–PARTY–YEAR) BY UNMATCHED RECORD COUNT")
    print(SEP)
    top10 = (grp[grp["unmatched"] > 0]
             .sort_values("unmatched", ascending=False)
             .head(10)
             .reset_index(drop=True))

    print(f"  {'#':<3} {'COMBO':<22} {'UNMTCH':>6}  {'TOTAL':>5}  "
          f"{'UNMTCH%':>7}  {'MED%':>7}  {'MEAN%':>7}  {'MAX%':>7}")
    print(f"  {SEP2}")
    for i, row in top10.iterrows():
        combo     = f"{row['state']}-{row['party']}-{int(row['year'])}"
        fully_tag = "  [FULL]" if row["fully_unmatched"] else ""
        print(f"  {i+1:<3} {combo:<22} {int(row['unmatched']):>6}  {int(row['total']):>5}  "
              f"{row['unmatch_rate']:>6.1f}%  "
              f"{row['pct_med']:>6.2f}%  {row['pct_mean']:>6.2f}%  {row['pct_max']:>6.2f}%"
              f"{fully_tag}")

    # ── fully-unmatched combos listed by year ─────────────────────────────
    print(f"\n{SEP}")
    print(f"  FULLY-UNMATCHED COMBOS LISTED BY YEAR")
    print(SEP)
    fully_df = grp[grp["fully_unmatched"]].sort_values(
        ["year", "unmatched"], ascending=[True, False]
    )
    if fully_df.empty:
        print("  None found.")
    else:
        cur_year = None
        for _, row in fully_df.iterrows():
            if row["year"] != cur_year:
                cur_year = row["year"]
                print(f"\n  [ {int(cur_year)} ]")
                print(f"    {'COMBO':<18} {'UNMTCH':>6}  {'MED%':>7}  {'MEAN%':>7}  {'MAX%':>7}")
                print(f"    {'-'*50}")
            combo = f"{row['state']}-{row['party']}"
            print(f"    {combo:<18} {int(row['unmatched']):>6}  "
                  f"{row['pct_med']:>6.2f}%  {row['pct_mean']:>6.2f}%  {row['pct_max']:>6.2f}%")

    return grp


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Match method breakdown by year
# ─────────────────────────────────────────────────────────────────────────────

def analyze_match_methods_by_year(df):
    print(f"\n{SEP}")
    print(f"  MATCH METHOD BREAKDOWN BY YEAR  (unmatched records only)")
    print(SEP)

    unmatched = df[df["in_NIMSP"] == False].copy()
    pivot = (unmatched.groupby(["year", "match_method"])
             .size()
             .unstack(fill_value=0)
             .sort_index())

    methods = list(pivot.columns)
    col_w   = max(len(m) for m in methods) + 2

    print(f"  {'YEAR':<6}" + "".join(f"{m:>{col_w}}" for m in methods))
    print(f"  {SEP2}")
    for year, row in pivot.iterrows():
        print(f"  {int(year):<6}" + "".join(f"{int(row[m]):>{col_w}}" for m in methods))

    totals    = pivot.sum()
    total_all = totals.sum()
    print(f"  {SEP2}")
    print(f"  {'TOTAL':<6}" + "".join(f"{int(totals[m]):>{col_w}}" for m in methods))
    print(f"\n  Method share of all unmatched records:")
    for m in methods:
        print(f"    {m:<25} {int(totals[m]):>6}  ({totals[m]/total_all*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Unmatched rate by party
# ─────────────────────────────────────────────────────────────────────────────

def analyze_by_party(df):
    print(f"\n{SEP}")
    print(f"  UNMATCHED RATE BY PARTY")
    print(SEP)

    party_stats = df.groupby("party").agg(
        total     = ("in_NIMSP", "count"),
        unmatched = ("in_NIMSP", lambda x: (x == False).sum()),
    )
    party_stats["unmatch_pct"] = (party_stats["unmatched"] / party_stats["total"] * 100).round(2)
    party_stats = party_stats.sort_values("unmatch_pct", ascending=False)

    print(f"  {'PARTY':<10} {'TOTAL':>7}  {'UNMATCHED':>10}  {'UNMATCH%':>9}")
    print(f"  {SEP2}")
    for party, row in party_stats.iterrows():
        print(f"  {party:<10} {int(row['total']):>7}  {int(row['unmatched']):>10}  {row['unmatch_pct']:>8.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — State-level degradation (unmatched rate change over time)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_state_degradation(df):
    print(f"\n{SEP}")
    print(f"  STATE DEGRADATION — UNMATCHED RATE CHANGE (earliest vs. latest year)")
    print(SEP)

    sy = df.groupby(["state", "year"]).agg(
        total     = ("in_NIMSP", "count"),
        unmatched = ("in_NIMSP", lambda x: (x == False).sum()),
    ).reset_index()
    sy["unmatch_pct"] = sy["unmatched"] / sy["total"] * 100

    results = []
    for state, g in sy.groupby("state"):
        g = g.sort_values("year")
        if len(g) < 2:
            continue
        first = g.iloc[0]
        last  = g.iloc[-1]
        delta = last["unmatch_pct"] - first["unmatch_pct"]
        results.append({
            "state":      state,
            "first_year": int(first["year"]),
            "last_year":  int(last["year"]),
            "first_pct":  first["unmatch_pct"],
            "last_pct":   last["unmatch_pct"],
            "delta_pp":   delta,
        })

    results_df = pd.DataFrame(results).sort_values("delta_pp", ascending=False)

    print(f"  {'STATE':<7} {'FROM':>6}→{'TO':<6}  {'FIRST%':>8}  {'LAST%':>7}  {'DELTA':>8}")
    print(f"  {SEP2}")
    for _, row in results_df.iterrows():
        arrow = "▲" if row["delta_pp"] > 0.5 else ("▼" if row["delta_pp"] < -0.5 else "≈")
        print(f"  {row['state']:<7} {row['first_year']:>6}→{row['last_year']:<6}  "
              f"{row['first_pct']:>7.1f}%  {row['last_pct']:>6.1f}%  "
              f"{arrow} {row['delta_pp']:>+6.1f}pp")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Combo size vs. match rate (do large combos match worse?)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_combo_size_vs_matchrate(grp):
    print(f"\n{SEP}")
    print(f"  COMBO SIZE vs. UNMATCHED RATE  (quintile buckets by total candidates)")
    print(SEP)

    g = grp.copy()
    g["size_quintile"] = pd.qcut(
        g["total"], q=5,
        labels=["Q1 (smallest)", "Q2", "Q3", "Q4", "Q5 (largest)"]
    )
    qstats = g.groupby("size_quintile", observed=True).agg(
        combos           = ("total", "count"),
        avg_total        = ("total", "mean"),
        avg_unmatch_rate = ("unmatch_rate", "mean"),
        med_unmatch_rate = ("unmatch_rate", "median"),
    ).round(2)

    print(f"  {'QUINTILE':<16} {'COMBOS':>7}  {'AVG SIZE':>9}  {'AVG UNMATCH%':>13}  {'MED UNMATCH%':>13}")
    print(f"  {SEP2}")
    for q, row in qstats.iterrows():
        print(f"  {str(q):<16} {int(row['combos']):>7}  {row['avg_total']:>9.1f}  "
              f"{row['avg_unmatch_rate']:>12.2f}%  {row['med_unmatch_rate']:>12.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print("  LOADING DATA")
    print(SEP)
    df = load_analysis_data()

    if df is not None:
        analyze_matching_by_year(df)
        grp = analyze_combo_breakdown(df)
        analyze_match_methods_by_year(df)
        analyze_by_party(df)
        analyze_state_degradation(df)
        analyze_combo_size_vs_matchrate(grp)
        print(f"\n{SEP}\n")
