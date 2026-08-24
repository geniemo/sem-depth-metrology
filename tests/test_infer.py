import zipfile

import numpy as np
import torch
from PIL import Image

from semdepth.infer import make_submission_zip, predict_dir
from semdepth.model import UnetTimm


class _IdentityModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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


def test_predict_tta_unflips_before_averaging(synth_root, tmp_path):
    n = predict_dir(
        _IdentityModel(), synth_root / "test" / "SEM", tmp_path / "pred_tta",
        device="cpu", flip_tta=True,
    )
    assert n == 4
    for p in sorted((synth_root / "test" / "SEM").glob("*.png")):
        out = np.array(Image.open(tmp_path / "pred_tta" / p.name))
        src = np.array(Image.open(p))
        # identity model + correct flip->unflip == exact uint8 roundtrip;
        # a missing/misaxised unflip would average a mirrored copy in and break this
        assert np.array_equal(out, src)


def test_predict_dir_recursive_mirrors_tree(synth_root, tmp_path):
    n = predict_dir(
        _IdentityModel(), synth_root / "train" / "SEM", tmp_path / "pseudo",
        device="cpu", recursive=True,
    )
    assert n == 9
    outs = sorted(p.relative_to(tmp_path / "pseudo").as_posix()
                  for p in (tmp_path / "pseudo").rglob("*.png"))
    ins = sorted(p.relative_to(synth_root / "train" / "SEM").as_posix()
                 for p in (synth_root / "train" / "SEM").rglob("*.png"))
    assert outs == ins
    assert outs[0].count("/") == 2  # Depth_XXX/site_XXXXX/SEM_XXXXXX.png
