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
python processing/download_data.py --other
```
Step 2:

**Option A: Download pre-made Parquet files** (most straightforward)
```bash
python processing/download_data.py --all
python analysis/cspy-match.py
```

**Option B: Selective download** (recommended)
```bash
python processing/download_data.py --year 2000
python analysis/cspy-match.py
# Enter years: 2000
```

### What's Tracked in Git?

✅ **Tracked:**
- All Python scripts in `processing/`, `analysis/`, `utils/`
- Configuration files (`data/data_sources.json`, `utils/paths.py`)
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
├── utils/
│   └── paths.py                     # Central path management
│
├── data/
│   └── data_sources.json            # Box shared links configuration
│
├── processing/                      # Data ETL & transformation scripts
│   ├── download_data.py             # Download data from Box cloud storage
│   ├── convert_year.py              # CSV to Parquet converter
│   ├── convert_party_data.py        # Party data conversion helpers
│   ├── extract_rows.py              # Row extraction utility
│   ├── update_candidate_ids_2000.py
│   └── update_candidate_state_2000.py
│
├── analysis/                        # Analysis & matching scripts
│   ├── cspy-match.py                # Main candidate-party matching script
│   ├── cspy-match2.py               # Variant matching logic
│   ├── cspy-match3.py               # Variant matching logic
│   ├── nul_recipients.py            # Recipients analysis
│   ├── nul_combo_hotspots.py        # Hotspot analysis
│   ├── bonica_rid_counts.py         # Bonica ID counting
│   ├── party_registration_trend.py  # Registration trends
│   ├── fallback_analysis.py         # Fallback analysis
│   ├── validate_output.py           # Output validation
│   ├── source_compare.py            # Source comparison
│   ├── source_integrity_checks.py   # Data integrity checks
│   ├── source_rowsize_comparison.py # Row size analysis
│   └── source_year_breakdown.py     # Year-wise breakdown
│
├── .gitignore                       # Exclude data files from Git
├── README.md                        # This file
│
├── DIME data/
│   ├── YYYY_candidate_donor.csv
│   ├── YYYY_parquet/               # Auto-created by convert_year.py
│   │   └── YYYY_candidate_donor.parquet
│
├── NIMSP data/
│   ├── party_donor.csv
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

1. **Central path module** - All scripts import from `utils.paths` for path resolution
2. **Config in data folder** - `data/data_sources.json` stores Box download links
3. **Organized by purpose** - `processing/` for ETL, `analysis/` for matching & inspection scripts
4. **Relative paths work everywhere** - Scripts accept relative paths and resolve them correctly
5. **Output directories auto-create** - No need to manually create `outputs/`, `{year}_parquet/`, etc.
6. **Name matching is intelligent** - Uses 3-tier strategy (exact → token-subset → fuzzy) to achieve ~90% match rate
7. **One-pass processing** - `cspy-match.py` now produces correct output directly; retcon scripts no longer needed

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
- Ensure you're in the workspace root directory when running scripts
- Check that all input data files exist in their expected locations
- Verify file names match exactly (case-sensitive on Linux/Mac)

**Parquet files not creating:**
- Run: `python processing/convert_year.py 2000 --overwrite` to force re-creation
- Check that `DIME data/` folder contains CSV files
- Ensure DuckDB is installed: `pip install duckdb`

**Import errors for utils.paths:**
- Make sure you're running scripts from the workspace root directory
- Verify that the `utils/` and `utils/paths.py` exist
- Check PYTHONPATH includes the workspace root if running from subdirectories
