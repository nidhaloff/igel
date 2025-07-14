import shutil
import os

# List of paths to remove
paths = [
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".coverage",
    "coverage.xml",
    "htmlcov",
    "*.egg-info",
    "__pycache__",
    ".cache",
    ".venv",
    "venv",
]

def remove_path(path):
    if os.path.isdir(path):
        print(f"Removing directory: {path}")
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        print(f"Removing file: {path}")
        os.remove(path)

for path in paths:
    # Handle wildcards
    for match in [p for p in os.listdir('.') if p.startswith(path.replace("*", ""))]:
        remove_path(match)
    # Try to remove the path directly
    remove_path(path)
