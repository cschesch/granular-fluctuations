"""Current project WRDS snapshots, selected explicitly by the source inventory."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_YEAR = 2025
ANALYSIS_LAST_DAY = "2025-12-31"
ANALYSIS_LAST_QUARTER = "2025Q4"


@lru_cache(None)
def wrds_path(dataset: str) -> Path:
    """Resolve a pinned private snapshot or its path in a clean checkout."""
    inventory = ROOT / "data/wrds_inventory.json"
    if inventory.is_file():
        records = json.loads(inventory.read_text())["datasets"]
        if record := next((row for row in records if row["dataset"] == dataset), None):
            return ROOT / record["file"]
    source = "mutual_funds" if dataset.startswith("mutual_fund_") else dataset.split("_", 1)[0]
    directory = ROOT / "data" / source
    canonical = directory / f"{dataset}.csv.gz"
    if canonical.is_file():
        return canonical
    dated = sorted(directory.glob(f"{dataset}_wrds_*.csv.gz"), reverse=True)
    return dated[0] if dated else canonical
