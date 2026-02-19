#!/usr/bin/env python3
"""
Verify imports in moved scripts.
Check that all moved scripts use 'from utils.paths import' and no 'from paths import'.
"""

from pathlib import Path
import re

ROOT = Path(__file__).parent
PROCESSING = ROOT / "processing"
ANALYSIS = ROOT / "analysis"
UTILS = ROOT / "utils"


def check_imports(script_path: Path) -> tuple[bool, list[str]]:
    """Check if script has correct imports. Returns (is_good, issues)."""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        issues = []
        
        # Check for old imports
        if re.search(r"from\s+paths\s+import", content):
            issues.append("Still has 'from paths import'")
        
        if re.search(r"import\s+paths\b", content):
            issues.append("Still has 'import paths'")
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [str(e)]


def main():
    print("=" * 70)
    print("IMPORT VERIFICATION TOOL")
    print("=" * 70)
    
    all_good = True
    total = 0
    
    for folder in [PROCESSING, ANALYSIS]:
        for script in sorted(folder.glob("*.py")):
            total += 1
            is_good, issues = check_imports(script)
            rel_path = script.relative_to(ROOT)
            
            if is_good:
                print(f"  ✓ {rel_path}")
            else:
                print(f"  ✗ {rel_path}")
                for issue in issues:
                    print(f"      - {issue}")
                all_good = False
    
    print("=" * 70)
    if all_good:
        print(f"✓ All {total} scripts have correct imports!")
    else:
        print(f"✗ Some scripts need import fixes")
    print("=" * 70)


if __name__ == "__main__":
    main()
