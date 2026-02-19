#!/usr/bin/env python3
"""
Fix path references in migrated scripts.
Adjust file path calculations for scripts moved to subdirectories.
"""

from pathlib import Path
import re

ROOT = Path(__file__).parent
PROCESSING = ROOT / "processing"
ANALYSIS = ROOT / "analysis"


def fix_config_file_ref(content: str, script_folder: str) -> str:
    """
    Convert CONFIG_FILE references to point correctly from subdirectory.
    E.g., inside processing/, data_sources.json is at ../data/data_sources.json
    """
    if "CONFIG_FILE = Path(__file__).parent / \"data_sources.json\"" in content:
        new_ref = "CONFIG_FILE = Path(__file__).parent.parent / \"data\" / \"data_sources.json\""
        content = content.replace(
            "CONFIG_FILE = Path(__file__).parent / \"data_sources.json\"",
            new_ref
        )
    return content


def fix_relative_paths(script_path: Path) -> bool:
    """Fix path references in a script. Returns True if modified."""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        original = content
        
        # Fix CONFIG_FILE reference
        if "CONFIG_FILE" in content and "data_sources.json" in content:
            content = fix_config_file_ref(content, str(script_path.parent.name))
        
        if content != original:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"  ERROR fixing {script_path}: {e}")
    
    return False


def main():
    print("=" * 70)
    print("PATH REFERENCE FIX TOOL")
    print("=" * 70)
    
    modified = 0
    for folder in [PROCESSING, ANALYSIS]:
        for script in folder.glob("*.py"):
            if fix_relative_paths(script):
                print(f"  FIXED {script.relative_to(ROOT)}")
                modified += 1
    
    print("=" * 70)
    print(f"Fixed: {modified} script(s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
