"""
Fallback Analysis Script
Analyzes records where IN_NIMSP is False, providing descriptive statistics
by state including counts and average percentages.
"""

import pandas as pd
import os
from pathlib import Path

# Set up paths
OUTPUTS_DIR = Path(__file__).parent / "outputs"


def load_analysis_data():
    """Load and combine all analysis CSV files from outputs folder."""
    dfs = []
    
    # Find all analysis CSV files
    csv_files = sorted(OUTPUTS_DIR.glob("*-analysis.csv"))
    
    if not csv_files:
        print(f"No analysis CSV files found in {OUTPUTS_DIR}")
        return None
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"Loaded {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined_df)}")
    
    return combined_df


def analyze_fallback_cases(df):
    """
    Analyze records where in_NIMSP is False (fallback cases).
    Returns descriptive statistics by state.
    """
    # Filter for fallback cases
    fallback_df = df[df['in_NIMSP'] == False].copy()
    
    print(f"\n{'='*70}")
    print(f"FALLBACK ANALYSIS (in_NIMSP = False)")
    print(f"{'='*70}")
    print(f"Total fallback records: {len(fallback_df)}")
    print(f"Percentage of total: {len(fallback_df)/len(df)*100:.2f}%\n")
    
    # Group by state
    state_stats = fallback_df.groupby('state').agg({
        'candidate_id': 'count',          # Number of records
        'percentage': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    state_stats.columns = ['Count', 'Avg_Percentage', 'Std_Percentage', 'Min_Percentage', 'Max_Percentage']
    state_stats = state_stats.sort_values('Count', ascending=False)
    
    print(f"{'STATE':<6} {'COUNT':<8} {'AVG %':<10} {'STD %':<10} {'MIN %':<10} {'MAX %':<10}")
    print(f"{'-'*54}")
    
    for state, row in state_stats.iterrows():
        print(f"{state:<6} {int(row['Count']):<8} {row['Avg_Percentage']:<10.4f} "
              f"{row['Std_Percentage']:<10.4f} {row['Min_Percentage']:<10.4f} {row['Max_Percentage']:<10.4f}")
    
    print(f"\n{'='*70}")
    print(f"TOP 10 STATES BY FALLBACK RECORD COUNT")
    print(f"{'='*70}")
    top_10 = state_stats.head(10).copy()
    top_10['Aggregate_Avg_%'] = top_10['Avg_Percentage']
    
    for idx, (state, row) in enumerate(top_10.iterrows(), 1):
        print(f"{idx:2}. {state}: {int(row['Count']):4} records | Avg %: {row['Avg_Percentage']:7.4f}% | Agg Avg %: {row['Aggregate_Avg_%']:7.4f}%")
    
    # Overall aggregate average
    overall_avg = fallback_df['percentage'].mean()
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS FOR FALLBACK CASES")
    print(f"{'='*70}")
    print(f"Overall aggregate average percentage: {overall_avg:.4f}%")
    print(f"Median percentage: {fallback_df['percentage'].median():.4f}%")
    print(f"Std deviation: {fallback_df['percentage'].std():.4f}%")
    print(f"Min percentage: {fallback_df['percentage'].min():.4f}%")
    print(f"Max percentage: {fallback_df['percentage'].max():.4f}%")
    
    # Match method breakdown for fallback cases
    print(f"\n{'='*70}")
    print(f"MATCH METHOD BREAKDOWN FOR FALLBACK CASES")
    print(f"{'='*70}")
    match_counts = fallback_df['match_method'].value_counts()
    for method, count in match_counts.items():
        pct = count / len(fallback_df) * 100
        print(f"{method}: {count:5} records ({pct:6.2f}%)")
    
    return state_stats


def analyze_matching_by_year(df):
    """
    Analyze NIMSP matching rates by year.
    Shows breakdown of matched vs unmatched records per year.
    """
    print(f"\n{'='*70}")
    print(f"IN_NIMSP MATCHING RATES BY YEAR")
    print(f"{'='*70}")
    
    # Group by year and count matching status
    year_stats = df.groupby('year').apply(
        lambda x: pd.Series({
            'total': len(x),
            'in_nimsp': (x['in_NIMSP'] == True).sum(),
            'not_in_nimsp': (x['in_NIMSP'] == False).sum(),
        })
    ).astype(int)
    
    year_stats = year_stats.sort_index()
    year_stats['match_pct'] = (year_stats['in_nimsp'] / year_stats['total'] * 100).round(2)
    year_stats['fallback_pct'] = (year_stats['not_in_nimsp'] / year_stats['total'] * 100).round(2)
    
    for year, row in year_stats.iterrows():
        print(f"{int(year)}: {row['fallback_pct']:5.2f}% unmatched, {int(row['not_in_nimsp']):5} unmatched out of {int(row['total']):5} "
              f"({row['match_pct']:5.2f}% matched)")
    
    # Overall statistics
    total_records = year_stats['total'].sum()
    total_matched = year_stats['in_nimsp'].sum()
    total_unmatched = year_stats['not_in_nimsp'].sum()
    overall_match_pct = total_matched / total_records * 100
    overall_fallback_pct = total_unmatched / total_records * 100
    
    print(f"\n{'='*70}")
    print(f"OVERALL MATCHING STATISTICS")
    print(f"{'='*70}")
    print(f"Total records: {total_records:,}")
    print(f"Matched (in_NIMSP=True): {total_matched:,} ({overall_match_pct:.2f}%)")
    print(f"Unmatched (in_NIMSP=False): {total_unmatched:,} ({overall_fallback_pct:.2f}%)")
    print(f"\nOverall average matching percentage: {overall_match_pct:.2f}%")


if __name__ == "__main__":
    df = load_analysis_data()
    
    if df is not None:
        analyze_matching_by_year(df)
        state_stats = analyze_fallback_cases(df)
