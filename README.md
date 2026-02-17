# Project Path Management & File Organization

## Overview

This project analyzes campaign finance data, matching state legislative candidates from DIME data against party donors from NIMSP data. The pipeline uses centralized path management (`paths.py`) to eliminate hardcoded paths and ensure portability.

## Data Management & Version Control

This project uses a size-based storage policy for data and code.

### Storage Policy

- **Small CSVs** → kept in **Git**
- **Medium Parquet files** → kept in **Box** (downloaded via `download_data.py`)
- **Large raw CSVs (uncompressed Parquet-scale)** → **not in Git or Box**; source from Stanford DIME: [https://data.stanford.edu/dime](https://data.stanford.edu/dime)
- **Code and scripts** → kept in **Git**

### Quick Setup

#### 1. First-Time Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd institutions

# Install dependencies
pip install requests tqdm duckdb pandas

# Get private Box links config from repository owner
# Place data_sources.json in the project root (same folder as download_data.py)
```

`data_sources.json` is required to download source files and is intentionally not committed to Git. Get this file from the project owner and place it in the working directory before running `download_data.py`.

#### 2. Upload Medium Parquet Data to Box (One-Time)

1. Go to [box.com](https://box.com) and upload your Parquet files
2. For each file, right-click → **Share** → **Create Shared Link**
3. Set permissions to "People with the link can **download**"
4. Copy the shared link (format: `https://app.box.com/s/xxxxx`)
5. Paste links into `data_sources.json`

#### 3. Download Parquet Data Files

```bash
# Download specific years
python download_data.py --year 2000 2002 2004

# Download everything
python download_data.py --all

# Download supporting parquet files
python download_data.py --other

# Re-download existing files
python download_data.py --overwrite --year 2000
```

### Workflow

**Option A: Download pre-made Parquet files** (default, fastest)
```bash
python download_data.py --all
python cspy-match.py
```

**Option B: Download large raw CSV zip from Stanford DIME, generate Parquet locally**
```bash
# Source: https://data.stanford.edu/dime
# Download CSV zip and extract into DIME data/
python convert_year.py 2000 2002 2004
python cspy-match.py
```

**Option C: Selective download**
```bash
python download_data.py --year 2000
python cspy-match.py
# Enter years: 2000
```

### What's Tracked in Git?

✅ **Tracked:**
- All Python scripts (`*.py`)
- Configuration files (`paths.py`)
- Documentation (`README.md`, `*.md`)
- Directory structure markers

❌ **Not Tracked (in `.gitignore`):**
- `data_sources.json` (private Box links; request from project owner)
- CSV files (`DIME data/**/*.csv`)
- Parquet files (`DIME data/**/*.parquet`)
- Analysis outputs (`outputs/`)
- Debug files (`debug_output.txt`, `result.txt`)

### Sharing Your Work

**To share code changes:**
```bash
git add cspy-match.py
git commit -m "Improved name matching algorithm"
git push
```

**To share new medium parquet data:**
1. Upload Parquet to Box
2. Get shared link
3. Update your local `data_sources.json`
4. Send updated `data_sources.json` to teammates through a private channel

**To share analysis results:**
- Small files (<10MB): Can commit to `outputs/` (update `.gitignore`)
- Large files: Upload to Box and share link

## Directory Structure

```
institutions/
├── paths.py                          # Path management (DO NOT MOVE)
├── download_data.py                 # Download data from Box cloud storage
├── data_sources.json                # Box shared links configuration
├── .gitignore                       # Exclude data files from Git
├── cspy-match.py                    # Main analysis script
├── convert_year.py                  # CSV to Parquet converter
├── extract_rows.py                  # Row extraction utility
│
├── DIME data/
│   ├── 2000_candidate_donor.csv
│   ├── 2002_candidate_donor.csv
│   ├── 2004_candidate_donor.csv
│   ├── 2000_parquet/               # Auto-created by convert_year.py
│   │   └── 2000_candidate_donor.parquet
│   ├── 2002_parquet/
│   │   └── 2002_candidate_donor.parquet
│   └── 2004_parquet/
│       └── 2004_candidate_donor.parquet
│
├── NIMSP data/
│   ├── party_donor.csv
│   └── notes.txt
│
├── outputs/
│   ├── 2000-analysis.csv
│   ├── 2002-analysis.csv
│   └── 2004-analysis.csv
│
├── primarydates/
│   └── legislative_primary_dates_full.csv
│
└── upperlower/
    ├── 2000_upper.csv
    ├── 2000_lower.csv
    ├── 2002_upper.csv
    └── ...
```

## Key Features

### 1. **Automatic Path Resolution**
- All paths are relative to the workspace root
- No global file paths needed in scripts
- Works regardless of where you run the script from

### 2. **Parquet Output Structure**
- Parquet files are organized in year-specific folders
- Format: `DIME data/{year}_parquet/{year}_candidate_donor.parquet`
- Example: `DIME data/2000_parquet/2000_candidate_donor.parquet`

## Usage

### Quick Start

1. **Convert CSV to Parquet** (faster processing, optional but recommended):
```bash
python convert_year.py 2000 2002 2004
```

2. **Run Analysis**:
```bash
python cspy-match.py
# When prompted, enter: 2000, 2002, 2004
```

3. **Validate Output**:
```bash
python validate_output.py outputs/2000-analysis.csv
```

### Convert CSV to Parquet

```bash
# Convert single year
python convert_year.py 2000

# Convert multiple years
python convert_year.py 2000 2002 2004

# Overwrite existing parquet files
python convert_year.py 2000 --overwrite
```

**What it does:**
- Reads: `DIME data/2000_candidate_donor.csv`
- Writes: `DIME data/2000_parquet/2000_candidate_donor.parquet`
- Directories are created automatically

### Run Main Analysis

```bash
python cspy-match.py
```

**Interactive prompt:**
```
Enter target years (comma-separated, e.g. 2000, 2002, 2004): 2000, 2002
```

**What it does:**
- Reads from all required folders (DIME data, NIMSP data, etc.)
- Writes analysis results to: `outputs/{year}-analysis.csv`
- All paths are managed by `paths.py`

### Extract Rows from CSV

```bash
python extract_rows.py
```

**Interactive prompt:**
```
Enter CSV file path (relative to workspace root): DIME data/2000_candidate_donor.csv
Number of rows to extract (default 100): 1000
```

**What it does:**
- Extracts N rows from any CSV
- Creates output in same directory as input
- Supports relative paths from workspace root

## Path Configuration

All paths are defined in `paths.py`. To modify paths, edit that file:

```python
# Example: Change output directory
OUTPUT_DIR = get_data_path("my_outputs")  # Instead of "outputs"

# Example: Add new data source
MY_DATA = get_data_path("my_folder/my_file.csv")
```

## Important Notes

1. **`paths.py` must stay in the workspace root** - All path resolution depends on it
2. **No global paths in scripts** - If you see hardcoded paths, they should be moved to `paths.py`
3. **Relative paths work everywhere** - Pass relative paths to utility functions; they'll be converted to absolute paths automatically
4. **Output directories auto-create** - No need to manually create `outputs/`, `{year}_parquet/`, etc.
5. **Name matching is intelligent** - Uses 3-tier strategy (exact → token-subset → fuzzy) to achieve ~90% match rate
6. **One-pass processing** - `cspy-match.py` now produces correct output directly; retcon scripts no longer needed

## Output Format

The analysis CSV contains:
- `candidate_id`: Format SS-PPP-YYYY-H-DD-CC (e.g., CA-DEM-2000-L-42-01)
- `candidate_name`: Full name from DIME data
- `party_donors_count`: Number of party donors who gave to this candidate
- `total_party_donors`: Total unique party donors in state
- `total_candidate_donors`: Total unique donors to candidate
- `percentage`: % of party donors who gave to candidate
- `candidate_state`: Election outcome (W=won general, P=lost general, L=lost primary, etc.)
- `match_method`: How name was matched (exact, token_subset, fuzzy_best, fallback)
- Plus: `seat_info`, `seat_type`, `state`, `party`, `year`

## Troubleshooting

**FileNotFoundError when running scripts:**
- Ensure you're in the workspace directory or have the correct working directory
- Check that all input data files exist in their expected locations
- Verify file names match exactly (case-sensitive on Linux/Mac)

**Parquet files not creating:**
- Run: `python convert_year.py 2000 --overwrite` to force re-creation
- Check that `DIME data/` folder contains CSV files
- Ensure DuckDB is installed: `pip install duckdb`

**Paths module not found:**
- Make sure `paths.py` is in the same directory as the script you're running
- Verify `__init__.py` is NOT needed (this is a single-file module)
