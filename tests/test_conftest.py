import numpy as np
import pandas as pd
from PIL import Image


def test_synth_layout(synth_root):
    sems = sorted((synth_root / "simulation_data" / "SEM").rglob("*.png"))
    depths = sorted((synth_root / "simulation_data" / "Depth").rglob("*.png"))
    assert len(sems) == 2 * len(depths) == 24  # 2 cases x 2 buckets x 3 structs x itr0/itr1
    img = np.array(Image.open(sems[0]))
    assert img.shape == (72, 48) and img.dtype == np.uint8
    # the same structure names repeat across both cases (real-data property)
    names = {p.name for p in depths}
    assert len(names) == 6
    for n in names:
        assert len([p for p in depths if p.name == n]) == 2
    df = pd.read_csv(synth_root / "train" / "average_depth.csv")
    assert list(df.columns) == ["0", "1"]
    assert len(df) == 3  # one label row per site, not per image
    assert str(df.iloc[0, 0]).startswith("depth_")
    assert len(list((synth_root / "train" / "SEM").rglob("*.png"))) == 9
    assert len(list((synth_root / "test" / "SEM").glob("*.png"))) == 4
