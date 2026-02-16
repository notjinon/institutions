#!/usr/bin/env python3
"""
download_data.py
Download DIME data files from Box cloud storage using shared links.

Usage:
    python download_data.py --year 2000 2002 2004    # Download specific years
    python download_data.py --all                    # Download all configured files
    python download_data.py --csv-only --year 2000   # Only download CSVs
    python download_data.py --parquet-only --all     # Only download Parquets
    python download_data.py --overwrite --year 2000  # Re-download existing files
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("❌ Missing required packages. Install with:")
    print("   pip install requests tqdm")
    sys.exit(1)

from paths import DIME_DATA_DIR, DIME_PARQUET_FILE, ensure_parquet_dir_exists

# Config file location
CONFIG_FILE = Path(__file__).parent / "data_sources.json"


def load_config() -> Dict:
    """Load download URLs from data_sources.json"""
    if not CONFIG_FILE.exists():
        print(f"❌ Config file not found: {CONFIG_FILE}")
        print("Create data_sources.json with Box shared links first.")
        sys.exit(1)
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    return config


def box_link_to_direct_download(shared_link: str) -> Optional[str]:
    """
    Convert Box shared link to direct download URL.
    
    Box shared links are in format:
        https://app.box.com/s/xxxxxxxxxxxxx
    
    To download, we need to use:
        https://app.box.com/shared/static/xxxxxxxxxxxxx
    
    Or append ?dl=1 for direct download
    """
    if "REPLACE_WITH_BOX_SHARED_LINK" in shared_link:
        return None
    
    # Handle different Box URL formats
    if "/s/" in shared_link:
        # Standard shared link format
        # Just append the download parameter
        if "?" in shared_link:
            return f"{shared_link}&dl=1"
        else:
            return f"{shared_link}?dl=1"
    
    return shared_link


def download_file(url: str, dest_path: Path, overwrite: bool = False) -> bool:
    """
    Download a file from Box with progress bar.
    
    Returns:
        True if downloaded successfully, False otherwise
    """
    if dest_path.exists() and not overwrite:
        print(f"⏭️  Skipping (already exists): {dest_path.name}")
        return True
    
    download_url = box_link_to_direct_download(url)
    if not download_url:
        print(f"❌ Invalid URL (not configured): {dest_path.name}")
        return False
    
    try:
        # Make request with streaming
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        with open(dest_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=dest_path.name,
                ncols=80
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {dest_path.name}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed for {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial download
        return False
    except Exception as e:
        print(f"❌ Unexpected error downloading {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_csv(year: int, config: Dict, overwrite: bool) -> bool:
    """Download DIME CSV file for a specific year"""
    year_str = str(year)
    if year_str not in config.get("dime_csv", {}):
        print(f"❌ No CSV configured for year {year}")
        return False
    
    url = config["dime_csv"][year_str]
    dest = DIME_DATA_DIR(year)
    
    return download_file(url, dest, overwrite)


def download_parquet(year: int, config: Dict, overwrite: bool) -> bool:
    """Download DIME Parquet file for a specific year"""
    year_str = str(year)
    if year_str not in config.get("dime_parquet", {}):
        print(f"❌ No Parquet configured for year {year}")
        return False
    
    url = config["dime_parquet"][year_str]
    ensure_parquet_dir_exists(year)
    dest = DIME_PARQUET_FILE(year)
    
    return download_file(url, dest, overwrite)


def download_other_files(config: Dict, overwrite: bool) -> None:
    """Download NIMSP and other supporting files"""
    # NIMSP party donor file
    if "nimsp" in config and "party_donor" in config["nimsp"]:
        url = config["nimsp"]["party_donor"]
        dest = Path("NIMSP data") / "party_donor.csv"
        download_file(url, dest, overwrite)
    
    # Primary dates file
    if "primary_dates" in config and "legislative_primary_dates_full" in config["primary_dates"]:
        url = config["primary_dates"]["legislative_primary_dates_full"]
        dest = Path("primarydates") / "legislative_primary_dates_full.csv"
        download_file(url, dest, overwrite)


def main():
    parser = argparse.ArgumentParser(
        description="Download DIME data files from Box cloud storage"
    )
    parser.add_argument(
        "--year",
        type=int,
        nargs="+",
        help="Specific year(s) to download (e.g., 2000 2002 2004)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all configured files"
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only download CSV files (not Parquet)"
    )
    parser.add_argument(
        "--parquet-only",
        action="store_true",
        help="Only download Parquet files (not CSV)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they exist locally"
    )
    parser.add_argument(
        "--other",
        action="store_true",
        help="Download supporting files (NIMSP, primary dates)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.year and not args.all and not args.other:
        parser.error("Specify --year, --all, or --other")
    
    if args.csv_only and args.parquet_only:
        parser.error("Cannot use both --csv-only and --parquet-only")
    
    # Load configuration
    config = load_config()
    
    # Determine which years to download
    years: List[int] = []
    if args.all:
        # Get all available years from config
        csv_years = set(config.get("dime_csv", {}).keys())
        parquet_years = set(config.get("dime_parquet", {}).keys())
        all_years = csv_years.union(parquet_years)
        years = sorted([int(y) for y in all_years if y.isdigit()])
    elif args.year:
        years = args.year
    
    # Download files
    success_count = 0
    fail_count = 0
    
    print(f"\n{'='*60}")
    print(f"DIME Data Downloader")
    print(f"{'='*60}\n")
    
    for year in years:
        print(f"\n📅 Year {year}")
        print(f"{'-'*60}")
        
        if not args.parquet_only:
            if download_csv(year, config, args.overwrite):
                success_count += 1
            else:
                fail_count += 1
        
        if not args.csv_only:
            if download_parquet(year, config, args.overwrite):
                success_count += 1
            else:
                fail_count += 1
    
    # Download other supporting files if requested
    if args.other or args.all:
        print(f"\n📚 Supporting Files")
        print(f"{'-'*60}")
        download_other_files(config, args.overwrite)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ {success_count} files downloaded successfully")
    if fail_count > 0:
        print(f"❌ {fail_count} files failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
