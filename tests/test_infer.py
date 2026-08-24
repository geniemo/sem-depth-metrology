import zipfile

import numpy as np
from PIL import Image

from semdepth.infer import make_submission_zip, predict_dir
from semdepth.model import UnetTimm


def test_predict_dir_and_zip(synth_root, tmp_path):
    model = UnetTimm("resnet18", pretrained=False).eval()
    n = predict_dir(model, synth_root / "test" / "SEM", tmp_path / "pred", device="cpu")
    assert n == 4
    in_names = sorted(p.name for p in (synth_root / "test" / "SEM").glob("*.png"))
    out_names = sorted(p.name for p in (tmp_path / "pred").glob("*.png"))
    assert in_names == out_names
    arr = np.array(Image.open(tmp_path / "pred" / out_names[0]))
    assert arr.dtype == np.uint8 and arr.shape == (72, 48)
    n_zip = make_submission_zip(tmp_path / "pred", tmp_path / "submission.zip")
    with zipfile.ZipFile(tmp_path / "submission.zip") as zf:
        assert sorted(zf.namelist()) == in_names
    assert n_zip == 4
