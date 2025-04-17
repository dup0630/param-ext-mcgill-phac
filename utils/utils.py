import json
import os

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
