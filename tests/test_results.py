from pathlib import Path

import pandas as pd

from semdepth.results import append_result


def test_append_creates_and_appends(tmp_path: Path):
    csv = tmp_path / "results.csv"
    append_result(csv, {"run": "a", "rmse": 1.0})
    append_result(csv, {"run": "b", "rmse": 2.0, "note": "extra col"})
    df = pd.read_csv(csv)
    assert list(df["run"]) == ["a", "b"]
    assert "note" in df.columns
    assert pd.isna(df.loc[0, "note"])
