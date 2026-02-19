"""
Centralized path management for the institutions project.
All paths are relative to the workspace root, eliminating need for global paths.
"""

from pathlib import Path


def get_workspace_root() -> Path:
    """Get the workspace root directory (where this script is located)."""
    return Path(__file__).parent.parent.resolve()


def get_data_path(relative_path: str) -> Path:
    """
    Get absolute path for any data file relative to workspace root.
    
    Examples:
        get_data_path("DIME data/2000_candidate_donor.csv")
        get_data_path("outputs/2000-analysis.csv")
    """
    return get_workspace_root() / relative_path


# Data file paths
# Exception: only raw DIME CSV inputs live outside the repo.
DIME_CSV_BASE = Path(r"D:\projects\institutions csv storage")
DIME_DATA_DIR = lambda year: DIME_CSV_BASE / f"{year}_candidate_donor.csv"
DIME_PARQUET_DIR = lambda year: get_data_path(f"DIME data/{year}_parquet")
DIME_PARQUET_FILE = lambda year: DIME_PARQUET_DIR(year) / f"{year}_candidate_donor.parquet"

NIMSP_PARTY_DATA = get_data_path("NIMSP data/party_donor.csv")
NIMSP_PARTY_DATA_PARQUET = get_data_path("NIMSP data/party_donor.parquet")

PRIMARY_DATES_FILE = get_data_path("primarydates/legislative_primary_dates_full.csv")

UPPER_HOUSE_FILE = lambda year: get_data_path(f"upperlower/{year}_upper.csv")
LOWER_HOUSE_FILE = lambda year: get_data_path(f"upperlower/{year}_lower.csv")

OUTPUT_DIR = get_data_path("outputs")
OUTPUT_FILE = lambda year: OUTPUT_DIR / f"{year}-analysis.csv"


def ensure_output_dir_exists() -> Path:
    """Create outputs directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def ensure_parquet_dir_exists(year: int) -> Path:
    """Create parquet directory for a given year if it doesn't exist."""
    parquet_dir = DIME_PARQUET_DIR(year)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    return parquet_dir


if __name__ == "__main__":
    # Test paths
    print(f"Workspace root: {get_workspace_root()}")
    print(f"DIME data (2000): {DIME_DATA_DIR(2000)}")
    print(f"Parquet output (2000): {DIME_PARQUET_FILE(2000)}")
    print(f"NIMSP party data: {NIMSP_PARTY_DATA}")
    print(f"Primary dates: {PRIMARY_DATES_FILE}")
    print(f"Output file (2000): {OUTPUT_FILE(2000)}")
