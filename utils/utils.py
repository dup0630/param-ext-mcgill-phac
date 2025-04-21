import json
import os
import re
import pandas as pd

def load_config(file_path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def cleanup_dir(dir: str) -> None:
    """
    Cleans up a directory by removing all files and subdirectories.
    """
    if not os.path.isdir(dir) or not os.path.exists(dir): raise ValueError(f"Could not find {dir}.")
    
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isdir(file_path):
            cleanup_dir(file_path)
            os.rmdir(file_path)
        elif os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(dir)

def extract_numeric(paper_filename: str):
    """
    Extract the numeric ID from paper names (e.g., '12.pdf' → 12).
    Used for sorting and comparison.
    """
    match = re.search(r'\d+', paper_filename)
    return int(match.group()) if match else float('inf')

def extract_first_number(entry):
    if pd.isnull(entry):
        return None
    # Find all numbers (integers or decimals)
    numbers = re.findall(r'\d+(?:\.\d+)?', str(entry))
    return float(numbers[0]) if numbers else None
