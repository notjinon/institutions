import pandas as pd
from pathlib import Path
from utils.paths import get_workspace_root


def extract_first_rows(input_path, num_rows=100):
    """
    Extract first N rows from a CSV file and save to a new file.
    
    Parameters:
    - input_path: Path to input CSV file (relative to workspace root)
    - num_rows: Number of rows to extract (default 100)
    """
    # Convert to Path object if string
    if isinstance(input_path, str):
        # If relative path, resolve from workspace root
        if not Path(input_path).is_absolute():
            input_path = get_workspace_root() / input_path
        else:
            input_path = Path(input_path)
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # Read first N rows
    df = pd.read_csv(input_path, nrows=num_rows)
    
    # Create output filename
    input_dir = input_path.parent
    input_filename = input_path.name
    name, ext = input_path.stem, input_path.suffix
    output_filename = f"{name}_first_{num_rows}_rows{ext}"
    output_path = input_dir / output_filename
    
    # Save to new file
    df.to_csv(output_path, index=False)
    
    print(f"Extracted {len(df)} rows from: {input_filename}")
    print(f"Saved to: {output_path}")
    print(f"Columns: {list(df.columns)}")
    
    return output_path


if __name__ == "__main__":
    # Get file path from user (can be relative to workspace root)
    file_path = input("Enter CSV file path (relative to workspace root): ").strip().strip('"')
    
    # Ask for number of rows (optional)
    rows_input = input("Number of rows to extract (default 100): ").strip()
    num_rows = int(rows_input) if rows_input else 100
    
    try:
        extract_first_rows(file_path, num_rows)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
