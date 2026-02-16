"""
Quick validation script to test cspy-match.py improvements.
Run this after processing data to check match quality.
"""
import pandas as pd
import sys

def validate_output(csv_path):
    """Validate a cspy-match.py output CSV."""
    print(f"Validating: {csv_path}")
    print("=" * 60)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File not found: {csv_path}")
        return False
    
    # Check required columns
    required_cols = [
        'candidate_id', 'candidate_name', 'party_donors_count',
        'total_party_donors', 'total_candidate_donors', 'percentage',
        'seat_info', 'seat_type', 'candidate_state', 'match_method',
        'state', 'party', 'year'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return False
    
    print(f"✓ All required columns present")
    print(f"✓ Total candidates: {len(df)}")
    print()
    
    # Match method distribution
    print("Match Method Distribution:")
    print("-" * 40)
    match_counts = df['match_method'].value_counts()
    total = len(df)
    for method, count in match_counts.items():
        pct = 100 * count / total
        print(f"  {method:20s}: {count:4d} ({pct:5.1f}%)")
    
    fallback_count = len(df[df['match_method'] == 'fallback'])
    matched_count = total - fallback_count
    print(f"\n  Successfully matched: {matched_count}/{total} ({100*matched_count/total:.1f}%)")
    print()
    
    # Election outcome distribution
    print("Election Outcome Distribution:")
    print("-" * 40)
    outcome_counts = df['candidate_state'].value_counts()
    for outcome, count in outcome_counts.items():
        pct = 100 * count / total
        outcome_desc = {
            'W': 'Won general',
            'P': 'Lost general',
            'L': 'Lost primary',
            'H': 'Withdrew general',
            'WP': 'Won primary',
            'DG': 'Disqualified general',
            'XG': 'Deceased general',
            'DW': 'Default winner',
        }.get(str(outcome), str(outcome))
        print(f"  {str(outcome):8s} ({outcome_desc:20s}): {count:4d} ({pct:5.1f}%)")
    print()
    
    # District variety check
    print("District Variety Check:")
    print("-" * 40)
    unique_districts = df['seat_info'].nunique()
    print(f"  Unique seat_info values: {unique_districts}")
    
    # Check for problematic hardcoded values
    all_al = len(df[df['seat_info'].str.endswith('-AL')])
    all_01 = len(df[df['seat_info'].str.endswith('-01')])
    
    if all_al == total or all_01 == total:
        print(f"  ⚠ WARNING: All districts are hardcoded to {df['seat_info'].iloc[0]}")
    else:
        print(f"  ✓ Districts are varied (not all hardcoded)")
    
    print("\n  Top 10 districts:")
    for district, count in df['seat_info'].value_counts().head(10).items():
        print(f"    {district}: {count}")
    print()
    
    # Sample output
    print("Sample Output (First 5 High-Match Candidates):")
    print("-" * 40)
    sample = df.nlargest(5, 'percentage')[
        ['candidate_id', 'candidate_name', 'percentage', 'match_method', 'candidate_state']
    ]
    for _, row in sample.iterrows():
        print(f"  {row['candidate_id']}")
        print(f"    Name: {row['candidate_name']}")
        print(f"    Party overlap: {row['percentage']:.1f}%")
        print(f"    Match: {row['match_method']}, Outcome: {row['candidate_state']}")
        print()
    
    print("=" * 60)
    print("✓ Validation complete")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "outputs/2000-analysis.csv"
    
    validate_output(csv_path)
