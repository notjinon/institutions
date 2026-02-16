#!/usr/bin/env python3
"""
csv2parquet_duckdb.py
Fast CSV → Parquet conversion using DuckDB *without* pulling data through pandas/Python.

Key idea:
- Keep the whole pipeline inside DuckDB: read_csv_auto(...) → COPY ... TO parquet
- Optional: write a multi-file dataset (parallel) via PER_THREAD_OUTPUT + FILE_SIZE_BYTES

Outputs to: DIME data/{year}_parquet/{year}_candidate_donor.parquet
"""

import argparse
import shutil
from pathlib import Path

import duckdb

# Import path management
from paths import DIME_DATA_DIR, DIME_PARQUET_FILE, ensure_parquet_dir_exists


def sql_quote(s: str) -> str:
    return s.replace("'", "''")


def sql_path(p: Path) -> str:
    return sql_quote(p.resolve().as_posix())


def as_dataset_dir(out: Path) -> Path:
    # If user gives foo.parquet but wants a dataset directory, use foo/
    return out.with_suffix("") if out.suffix.lower() == ".parquet" else out


def rm_any(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def build_read_csv_auto_sql(
    csv_path: Path,
    delimiter: str,
    header: bool,
    sample_size: int,
    nullstr: str | None,
    ignore_errors: bool = False,
    column_types: dict | None = None,
) -> str:
    # Only include options we actually need (keeps SQL cleaner).
    parts = [
        f"'{sql_path(csv_path)}'",
        f"delim='{sql_quote(delimiter)}'",
        f"header={'true' if header else 'false'}",
    ]

    # DuckDB default sample_size is usually fine; sample_size=-1 forces full scan (often slower).
    if sample_size is not None:
        parts.append(f"sample_size={int(sample_size)}")

    if nullstr:
        parts.append(f"nullstr='{sql_quote(nullstr)}'")
    
    # Ignore unicode/encoding errors
    if ignore_errors:
        parts.append("ignore_errors=true")
    
    # Override column types to avoid inference issues
    if column_types:
        types_str = ", ".join([f"'{k}': '{v}'" for k, v in column_types.items()])
        parts.append(f"types={{{types_str}}}")

    return f"read_csv_auto({', '.join(parts)})"


def parse_partition_by(s: str | None) -> list[str]:
    if not s:
        return []
    cols = [c.strip() for c in s.split(",")]
    return [c for c in cols if c]


def convert_year_to_parquet(year: int, overwrite: bool = False) -> None:
    """
    Convert DIME candidate-donor CSV to Parquet for a specific year.
    
    Parameters:
    - year: Election year (e.g., 2000, 2002, 2004)
    - overwrite: Whether to overwrite existing parquet file
    """
    # Get input and output paths
    csv_path = DIME_DATA_DIR(year)
    parquet_file = DIME_PARQUET_FILE(year)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Ensure parquet directory exists
    ensure_parquet_dir_exists(year)
    
    # Check if output already exists
    if parquet_file.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists: {parquet_file} (use --overwrite to replace)"
            )
        rm_any(parquet_file)
    
    # Force problematic columns to VARCHAR to avoid type inference issues
    # contributor.zipcode: Contains state codes like "WA", "MI" in some rows
    # recipient.party: Contains text codes like "UNK", "GRE" in addition to numeric codes
    column_types = {
        'contributor.zipcode': 'VARCHAR',
        'recipient.party': 'VARCHAR',
    }
    
    # Build DuckDB SQL with error handling and type overrides
    read_sql = build_read_csv_auto_sql(
        csv_path=csv_path,
        delimiter=",",
        header=True,
        sample_size=200_000,  # Increased from 100k to catch more edge cases
        nullstr=None,
        ignore_errors=True,  # Skip rows with unicode/encoding errors
        column_types=column_types,
    )
    
    target_sql = sql_path(parquet_file)
    
    copy_sql = f"""
    COPY (
        SELECT *
        FROM {read_sql}
    )
    TO '{target_sql}'
    (FORMAT PARQUET, COMPRESSION snappy);
    """
    
    print(f"Converting {year} DIME data to Parquet...")
    print(f"  Input:  {csv_path}")
    print(f"  Output: {parquet_file}")
    print()
    
    con = duckdb.connect()
    try:
        con.execute("PRAGMA enable_progress_bar")
        con.execute("PRAGMA enable_print_progress_bar")
        con.execute(copy_sql)
        print(f"✅ Successfully converted {year} data to parquet")
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast CSV → Parquet conversion using DuckDB (no pandas roundtrips). "
                    "Automatically reads from DIME data folder and writes to {year}_parquet subfolder."
    )
    ap.add_argument(
        "year",
        type=int,
        nargs="+",
        help="Election year(s) to convert (e.g., 2000 2002 2004)"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet files.",
    )
    ap.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,)",
    )

    args = ap.parse_args()
    
    for year in args.year:
        try:
            convert_year_to_parquet(year, overwrite=args.overwrite)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            continue
        except FileExistsError as e:
            print(f"❌ Error: {e}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error converting {year}: {e}")
            continue
    
    print("\n✅ All conversions complete!")


if __name__ == "__main__":
    main()
