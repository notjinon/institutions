#!/usr/bin/env python3
"""
convert_party_data.py
One-time conversion of NIMSP party_donor.csv to Parquet format using DuckDB.

This converts the party donor CSV to Parquet for faster loading in cspy-match2.py.

Usage:
    python convert_party_data.py
    python convert_party_data.py --overwrite  # Force re-conversion
"""

import argparse
import shutil
from pathlib import Path

import duckdb


def sql_quote(s: str) -> str:
    """Escape single quotes for SQL."""
    return s.replace("'", "''")


def sql_path(p: Path) -> str:
    """Convert Path to SQL-safe POSIX string."""
    return sql_quote(p.resolve().as_posix())


def rm_any(path: Path) -> None:
    """Remove file or directory if it exists."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def convert_party_data_to_parquet(overwrite: bool = False) -> None:
    """
    Convert NIMSP party_donor.csv to Parquet format.
    
    Parameters:
    - overwrite: Whether to overwrite existing parquet file
    """
    # Define paths
    workspace_root = Path(__file__).parent.resolve()
    csv_path = workspace_root / "NIMSP data" / "party_donor.csv"
    parquet_path = workspace_root / "NIMSP data" / "party_donor.parquet"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Check if output already exists
    if parquet_path.exists():
        if not overwrite:
            print(f"✅ Parquet file already exists: {parquet_path}")
            print("   Use --overwrite to force re-conversion")
            return
        rm_any(parquet_path)
    
    # Build DuckDB SQL for conversion
    read_sql = f"""
    read_csv_auto(
        '{sql_path(csv_path)}',
        delim=',',
        header=true,
        sample_size=200000,
        ignore_errors=true
    )
    """
    
    target_sql = sql_path(parquet_path)
    
    copy_sql = f"""
    COPY (
        SELECT *
        FROM {read_sql}
    )
    TO '{target_sql}'
    (FORMAT PARQUET, COMPRESSION snappy);
    """
    
    print(f"Converting NIMSP party data to Parquet...")
    print(f"  Input:  {csv_path}")
    print(f"  Output: {parquet_path}")
    print()
    
    con = duckdb.connect()
    try:
        con.execute("PRAGMA enable_progress_bar")
        con.execute("PRAGMA enable_print_progress_bar")
        con.execute(copy_sql)
        
        # Get file sizes for comparison
        csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
        parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - parquet_size_mb / csv_size_mb) * 100
        
        print(f"\n✅ Successfully converted party data to Parquet")
        print(f"   CSV size:     {csv_size_mb:.2f} MB")
        print(f"   Parquet size: {parquet_size_mb:.2f} MB")
        print(f"   Compression:  {compression_ratio:.1f}% smaller")
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert NIMSP party_donor.csv to Parquet format using DuckDB."
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet file.",
    )

    args = ap.parse_args()
    
    try:
        convert_party_data_to_parquet(overwrite=args.overwrite)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Conversion complete!")
    print("\nNext steps:")
    print("  1. Update cspy-match2.py to use Parquet for party data (optional)")
    print("  2. Run: python cspy-match2.py")


if __name__ == "__main__":
    main()
