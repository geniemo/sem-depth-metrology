from pathlib import Path

import pandas as pd


def append_result(csv_path: Path, row: dict) -> None:
    """Append one experiment-result row; creates file and unions columns as needed."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(csv_path, index=False)
