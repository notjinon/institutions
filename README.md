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

# institutions — Technical Reference

Purpose
-------
Provide reproducible processing and analysis of DIME candidate-donor files and NIMSP party-donor data. This README is a concise operator/developer manual: prerequisites, configuration, data layout, script reference, workflows, expected outputs, and troubleshooting.

Prerequisites
-------------
- Python 3.8+
- Install minimal runtime dependencies used by scripts:

```bash
python -m pip install --upgrade pip
python -m pip install pandas duckdb requests tqdm python-dateutil
```

- Workspace: clone the repository and run commands from the repository root.

Configuration
-------------
- `data_sources.json` — required for `download_data.py`. Contains private Box shared links. This file is excluded from Git; acquire it from the project owner and place it at the repository root.
- `paths.py` — single source of truth for all file locations. Do not move or duplicate. Modify only when you want to change directory locations used by all scripts.

Data layout (canonical)
-----------------------
- `DIME data/` — raw CSVs and generated per-year parquet directories `YYYY_parquet/`
- `NIMSP data/` — `party_donor.parquet` and supporting files
- `outputs/` — generated analysis CSVs named `{year}-analysis.csv`
- `primarydates/`, `upperlower/` — auxiliary CSVs used in matching

Quick operational commands
--------------------------
Use these exact commands from the repo root (Windows PowerShell or POSIX shell):

- Download medium/parquet files from Box (requires `data_sources.json`):

```bash
python download_data.py --year 2000 2002 2004      # download specific years
python download_data.py --all                      # download all configured years
python download_data.py --other                    # download NIMSP party_donor.parquet
python download_data.py --overwrite --year 2000    # force re-download
```

- Convert raw DIME CSVs to Parquet (local conversion):

```bash
python convert_year.py 2000 2002 2004              # generate year parquet dirs
python convert_year.py 2000 --overwrite            # regenerate a single year
```

- Run the main analysis pipeline (matching):

```bash
python cspy-match.py                                 # interactive: supply years
python cspy-match2.py                                # alternate or batch mode
```

- Validate output:

```bash
python validate_output.py outputs/2000-analysis.csv
```

Script reference (purpose and CLI)
----------------------------------
- `download_data.py` — downloads medium parquet assets and supporting files from Box. Requires `data_sources.json`.
    - Flags: `--year`, `--all`, `--other`, `--overwrite`

- `convert_year.py` — converts DIME CSV years to DuckDB/parquet layout used by analysis. Idempotent unless `--overwrite`.
    - Usage: `python convert_year.py <year> [<year> ...] [--overwrite]`

- `cspy-match.py` — primary analysis driver. Reads DIME parquet and `NIMSP data/party_donor.parquet`, writes `{year}-analysis.csv` in `outputs/`.
    - Interactive by default; years may be provided via prompt or refactor for batch invocation.

- `cspy-match2.py` — variant/batch mode of matching; consult inline header comments for differences.

- `extract_rows.py` — utility to sample rows from CSVs for quick inspection.

- `validate_output.py` — lightweight validator for output schema and basic integrity checks.

- `update_candidate_ids_2000.py`, `update_candidate_state_2000.py` — back-compat / fix-up utilities for legacy ID/state corrections.

Typical workflows
-----------------
1) Fast path (recommended when parquet files are available):

```bash
python download_data.py --all
python download_data.py --other
python cspy-match.py
```

2) Recompute from raw CSVs (if you obtained raw DIME zips):

```bash
# extract raw CSVs into `DIME data/`
python convert_year.py 2000 2002 2004
python cspy-match.py
```

3) Minimal debug run for a single year:

```bash
python convert_year.py 2000 --overwrite
python cspy-match.py   # select 2000
```

Outputs and schema
------------------
Outputs are CSV files written to `outputs/{year}-analysis.csv`. Expected core columns:

- `candidate_id`, `candidate_name`, `state`, `year`, `party`, `seat_type`, `party_donors_count`, `total_party_donors`, `total_candidate_donors`, `percentage`, `match_method`

Operational notes and expectations
----------------------------------
- All scripts use `paths.py` — do not hardcode alternate paths inside scripts.
- Parquet generation via `convert_year.py` uses DuckDB; ensure `duckdb` is installed.
- Name-matching strategy: exact → token-subset → fuzzy; `match_method` indicates which branch succeeded.

Troubleshooting (quick checks)
------------------------------
- FileNotFoundError: confirm current working directory is repo root and `paths.py` exists.
- Missing `data_sources.json`: obtain from owner; without it `download_data.py` will fail.
- Parquet not created: verify raw CSV exists in `DIME data/` and run `convert_year.py` with `--overwrite`.
- Performance: increase available memory or run per-year to reduce memory footprint.

Developer notes
---------------
- Keep `paths.py` as canonical path resolver; any new script must use it for file IO.
- Add tests for `cspy-match` logic before refactoring matching heuristics.
- When adding new external deps, update the top `pip install` list and document version constraints.

Next steps
----------
- If you want, I can:
    - Open a PR with this README change.
    - Create a minimal `requirements.txt` or `pyproject.toml`.
    - Add a non-interactive CLI wrapper for `cspy-match.py`.
