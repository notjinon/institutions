#!/usr/bin/env python3
"""
Migrate scripts from root to organized folders with import updates.
Skips if exact copy already exists in destination.
"""

import hashlib
import shutil
from pathlib import Path

# Map of script -> destination folder
SCRIPT_MOVES = {
    # Processing scripts
    "download_data.py": "processing",
    "convert_year.py": "processing",
    "convert_party_data.py": "processing",
    "extract_rows.py": "processing",
    "update_candidate_ids_2000.py": "processing",
    "update_candidate_state_2000.py": "processing",
    
    # Analysis scripts
    "cspy-match.py": "analysis",
    "cspy-match2.py": "analysis",
    "cspy-match3.py": "analysis",
    "nul_recipients.py": "analysis",
    "nul_combo_hotspots.py": "analysis",
    "bonica_rid_counts.py": "analysis",
    "party_registration_trend.py": "analysis",
    "fallback_analysis.py": "analysis",
    
    # Data quality & inspection
    "source_compare.py": "analysis",
    "source_integrity_checks.py": "analysis",
    "source_rowsize_comparison.py": "analysis",
    "source_year_breakdown.py": "analysis",
    "validate_output.py": "analysis",
}

ROOT = Path(__file__).parent
PROCESSING = ROOT / "processing"
ANALYSIS = ROOT / "analysis"

PROCESSING.mkdir(exist_ok=True)
ANALYSIS.mkdir(exist_ok=True)


def file_hash(p: Path) -> str:
    """Compute SHA256 of file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def update_imports(content: str) -> str:
    """Replace 'from paths import' with 'from utils.paths import'."""
    lines = content.split("\n")
    updated = []
    for line in lines:
        if line.strip().startswith("from paths import"):
            line = line.replace("from paths import", "from utils.paths import")
        elif line.strip().startswith("import paths"):
            line = line.replace("import paths", "import utils.paths as paths")
        updated.append(line)
    return "\n".join(updated)


def migrate_script(script_name: str, dest_folder: str) -> bool:
    """
    Copy script to dest_folder with updated imports.
    Returns True if copied, False if skipped (already exists).
    """
    src = ROOT / script_name
    if not src.exists():
        print(f"  SKIP {script_name}: source not found")
        return False
    
    if dest_folder == "processing":
        dst = PROCESSING / script_name
    else:
        dst = ANALYSIS / script_name
    
    # Check if already copied (exact match)
    if dst.exists():
        if file_hash(src) == file_hash(dst):
            print(f"  SKIP {script_name}: exact copy already exists at {dst.relative_to(ROOT)}")
            return False
        else:
            print(f"  OVER {script_name}: different copy exists, replacing")
    
    # Read source
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    # Update imports
    updated = update_imports(content)
    
    # Write to destination
    with open(dst, "w", encoding="utf-8") as f:
        f.write(updated)
    
    print(f"  MOVE {script_name} -> {dst.relative_to(ROOT)}")
    return True


def main():
    print("=" * 70)
    print("SCRIPT MIGRATION TOOL")
    print("=" * 70)
    
    moved_count = 0
    for script, folder in sorted(SCRIPT_MOVES.items()):
        try:
            if migrate_script(script, folder):
                moved_count += 1
        except Exception as e:
            print(f"  ERROR {script}: {e}")
    
    print("=" * 70)
    print(f"Migration complete: {moved_count} script(s) moved/updated")
    print("Originals left in root for manual cleanup.")
    print("=" * 70)


if __name__ == "__main__":
    main()
