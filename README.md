# institutions — Reproducible guide for R-first researchers

This guide tells you exactly what to run (and why) so political-science researchers who prefer R can reproduce the pipeline. Short explanations are included where a choice affects reproducibility or performance.

1. Essentials (why these matter)
--------------------------------
- Parquet files are used for performance and reproducibility: smaller I/O, columnar reads, friendly to both Python and R (`arrow`, `duckdb`).
- `paths.py` centralizes file locations so scripts are portable; do not hardcode file locations.
- Medium-sized files are stored on Box (private links in `data_sources.json`) to avoid committing large binaries to Git.

2. Environment (minimal, step-by-step)
--------------------------------------
Follow these commands from the repository root.

2.1. System tools

Install Git and make sure you can run `python` and `R` from the shell.

2.2. Python (needed by provided scripts)

```powershell
# Windows PowerShell (single line copy-paste)
python -m pip install --upgrade pip
python -m pip install pandas duckdb requests tqdm python-dateutil
```

2.3. R (for users who will inspect results or run parts in R)

Open R or RStudio and install required packages:

```r
install.packages(c("arrow","duckdb","httr","readr","dplyr"))
# Optional: reticulate to call Python directly from R
install.packages("reticulate")
```

3. Acquire repository and config (exact commands)
------------------------------------------------

```bash
- FileNotFoundError: confirm current working directory is repo root and `paths.py` exists.
cd institutions
```

Get `data_sources.json` from the project owner and place it in the repository root (same folder as `download_data.py`). This file contains private Box links required by `download_data.py`.

4. Reproducible workflows — choose one (with R-friendly options)
----------------------------------------------------------------

Option A — Use Python scripts (recommended; exact commands):

```bash
# 1) Download medium files that live on Box
python download_data.py --all
python download_data.py --other

# 2) Run the analysis (interactive by default)
python cspy-match.py

# 3) Validate one output
python validate_output.py outputs/2000-analysis.csv
```

Why this path: uses the code as written, avoids reimplementing matching logic in R, and produces the canonical `outputs/` CSVs.

Option B — Use R to drive the existing Python scripts (no Python coding required inside R)

In R, use `system2()` or `reticulate` to execute the exact commands above. Example using `system2()`:

```r
# From the repo root in R
system2("python", args = c("download_data.py", "--all"), stdout=TRUE)
system2("python", args = c("download_data.py", "--other"), stdout=TRUE)
system2("python", args = c("cspy-match.py"), stdout=TRUE)
```

Option C — Process DIME CSVs in R (if you cannot run Python)

1) Place raw DIME CSVs into `DIME data/`.
2) Convert CSV to Parquet in R and follow the pipeline's expected layout:

```r
library(arrow)
library(readr)

# read CSV and write parquet for year 2000 (example)
df <- read_csv("DIME data/2000_candidate_donor.csv")
write_parquet(df, "DIME data/2000_parquet/2000_candidate_donor.parquet")
```

3) Copy or create `NIMSP data/party_donor.parquet` similarly so the Python `cspy-match.py` can read it, or port the matching logic to R (not recommended unless necessary).

Notes: converting CSV→parquet in R is straightforward but the matching logic lives in `cspy-match.py`. For reproducible results prefer Option A or B.

5. Inspecting outputs in R
---------------------------
Once the pipeline finishes, open outputs in R:

```r
library(readr)
res <- read_csv("outputs/2000-analysis.csv")
summary(res)
head(res)
```

6. Quick troubleshooting (what to check, in order)
------------------------------------------------
- Confirm current working directory is the repository root and `paths.py` exists.
- Confirm `data_sources.json` is present for `download_data.py` runs.
- If `convert_year.py` fails, ensure the expected raw CSV is at `DIME data/<year>_candidate_donor.csv`.
- If Python scripts fail with missing modules, run the Python install commands in section 2.2.
- If outputs are empty, rerun for a single year with `--overwrite` and watch the logs:

```bash
python convert_year.py 2000 --overwrite
python cspy-match.py   # then select 2000
```

7. Short rationale notes (helpful context)
----------------------------------------
- Why Box links: keeps repository small and avoids distributing potentially sensitive data through Git.
- Why `paths.py`: makes scripts portable across OSes and avoids hidden hardcoded paths.
- Why prefer Python execution: the matching code is implemented and tested in Python; calling it from R (via `system2()` or `reticulate`) preserves behavior and reproducibility.

8. Next actions I can take (pick one)
-----------------------------------
- Add a one-line wrapper R script that runs the canonical pipeline and then reads `outputs/` into R.
- Produce `requirements.txt` and a short `install.sh` / `install.ps1` to install Python deps.
- Add a non-interactive flag to `cspy-match.py` so it accepts `--years 2000,2002`.

If you want the R wrapper, I will add it now.
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
