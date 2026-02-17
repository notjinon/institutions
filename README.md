# Repo "institutions"

## Overview

This project analyzes campaign finance data, matching state legislative candidates from DIME data against party donors from NIMSP data. 

## Data Management & Version Control

This project separates **code** (tracked in Git) from **data** (stored in Box cloud storage).

### Why This Setup?

- **CSV/Parquet files are too large for Git** (dozens of GB each)
- **Code is version-controlled** - easy to collaborate and track changes
- **Data is centralized in Box** - single source of truth, accessible to team
- **Local analysis** - download once, work offline

### Quick Setup

#### 1. First-Time Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/notjinon/institutions.git
cd institutions
pip install requests tqdm duckdb pandas
```

#### 2. Download Data Files

There are two steps. You need to download the DIME candidate contribution files, and the NIMSP party contribution files/other helpers.

Step 1:

**Download Helper (NIMSP, Primary Dates, etc.) CSV files** 
```bash
python download_data.py --other
```
Step 2:

**Option A: Download pre-made Parquet files** (most straightforward)
```bash
python download_data.py --all
python cspy-match.py
```

**Option B: Selective download** (recommended)
```bash
python download_data.py --year 2000
python cspy-match.py
# Enter years: 2000
```

### What's Tracked in Git?

✅ **Tracked:**
- All Python scripts (`*.py`)
- Configuration files (`data_sources.json`, `paths.py`)
- Documentation (`README.md`, `*.md`)
- Directory structure markers

❌ **Not Tracked (in `.gitignore`):**
- CSV files (`DIME data/**/*.csv`)
- Parquet files (`DIME data/**/*.parquet`)
- Analysis outputs (`outputs/`)
- Debug files (`debug_output.txt`, `result.txt`)

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
│   ├── YYYY_candidate_donor.csv
│   ├── YYYY_parquet/               # Auto-created by convert_year.py
│   │   └── YYYY_candidate_donor.parquet
│
├── NIMSP data/
│   ├── party_donor.parquet
│   └── notes.txt
│
├── outputs/
│   ├── YYYY-analysis.csv
│
├── primarydates/
│   └── legislative_primary_dates_full.csv
│
└── upperlower/
    ├── YYYY_upper.csv
    ├── YYYY_lower.csv
    └── ...
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
