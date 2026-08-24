import numpy as np
import torch

from semdepth.classify import (
    BucketClassifier,
    BucketDataset,
    list_real_labeled,
    predict_buckets,
    split_by_site,
)


def test_list_real_labeled_and_site_split(synth_root):
    recs = list_real_labeled(synth_root / "train" / "SEM")
    assert len(recs) == 9
    labels = {r[2]: r[1] for r in recs}
    assert labels["110/site_00000"] == 0 and labels["120/site_00002"] == 1
    tr, va = split_by_site(recs, val_fraction=0.34, seed=5)
    tr_sites, va_sites = {r[2] for r in tr}, {r[2] for r in va}
    assert tr_sites.isdisjoint(va_sites)
    assert len(tr) + len(va) == 9


def test_bucket_dataset_and_classifier_forward(synth_root):
    recs = list_real_labeled(synth_root / "train" / "SEM")
    ds = BucketDataset(recs)
    item = ds[0]
    assert item["image"].shape == (1, 72, 48) and item["label"] == 0
    model = BucketClassifier(pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.stack([ds[i]["image"] for i in range(3)]))
    assert logits.shape == (3, 4)


def test_predict_buckets_shapes(synth_root):
    recs = list_real_labeled(synth_root / "train" / "SEM")
    model = BucketClassifier(pretrained=False)
    preds = predict_buckets(model, [r[0] for r in recs], device="cpu", batch_size=4)
    assert preds.shape == (9,)
    assert set(preds.tolist()) <= {0, 1, 2, 3}
