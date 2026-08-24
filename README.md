# semdepth — Depth Metrology from Top-down SEM Images

Predicting per-pixel depth maps of semiconductor hole structures from top-down
SEM images — a sim-to-real domain adaptation problem from the
[2022 Samsung AI Challenge: 3D Metrology](https://dacon.io/competitions/official/235954)
(practiced post-competition on the live scoring server).

**Result: Public RMSE 4.597 / Private 4.618 — rank 26 of 172 submitting teams**
(790 registered), starting from a pure-simulation baseline of 8.000 (rank 118).

## The problem

Pixel-level depth ground truth exists only for **simulated** SEM images
(21,663 structures × 4 imaging conditions × 2 noise realizations); the
**real** training images carry only one *average-depth* scalar per site, and
the test set is real. Three mismatches must be crossed: appearance (sim is
sharp/noisy, real is smooth), label encoding (sim depth pixels are
smaller-is-deeper; real averages are physical larger-is-deeper), and label
shift (sim covers only the middle of the real depth range).

## What worked (and what didn't)

| Step | Idea | Public LB |
|---|---|---|
| S1 | Supervised U-Net (timm resnet34) on simulation only | 8.000 |
| E2 | **Weak supervision with EM-refit affine** — closed-form (α,β) refit per epoch links the unknown label encodings; real site means constrain the prediction range | 5.325 |
| E6 | **Frame-averaging hypothesis** — train on the mean of the two simulated noise realizations, matching the real acquisition look without touching brightness | 5.113 |
| E9 | Spectrum-calibrated residual blur (σ*=0.7 fitted by radial power-spectrum matching) | member |
| E8 | 3-member ensemble (2 recipes × resnet34 + convnext_tiny) + flip TTA | **4.597** |

Negative results that shaped the method: global intensity alignment and
brightness/contrast randomization both *hurt* (absolute brightness is the
primary transferable depth cue, r = −0.977 — standard DA intuition deletes the
signal); self-training on pseudo labels improved every local metric while
degrading the leaderboard (confirmation bias on real pixel shapes); a 6-member
ensemble diluted a 3-member one. Details: [report/report.md](report/report.md) (Korean).

## Repository

```
src/semdepth/     data (leak-safe structure-level splits), model, train,
                  finetune (EM-affine weak supervision), infer (ensemble/TTA), proxy
scripts/          train / finetune / predict / predict_ensemble / make_pseudo /
                  eval_real_proxy / inspect_data / eda / site_consistency
configs/          one YAML per experiment (S1, E1–E11)
experiments/      results.csv, submissions.csv — every run and submission logged
report/           EDA + final report (Korean), figures
docs/             verified dataset notes, design spec, plans
tests/            32 tests incl. split-leak guards, TTA roundtrip, EM-affine refit
```

## Reproduce

```bash
uv sync                                            # torch cu128 (Blackwell-ready)
uv run pytest -q                                   # 32 passed
uv run python scripts/inspect_data.py              # verify dataset layout (data/raw/)
uv run python scripts/train.py    -c configs/e6a_itrmean_base.yaml
uv run python scripts/finetune.py -c configs/e6b_itrmean_weak.yaml
uv run python scripts/predict_ensemble.py --tta \
  --member configs/e6b_itrmean_weak.yaml:experiments/runs/e6b_itrmean_weak/best.pt \
  --member configs/e7b_itrmean_blur_weak.yaml:experiments/runs/e7b_itrmean_blur_weak/best.pt \
  --member configs/e4c_convnext_weak.yaml:experiments/runs/e4c_convnext_weak/best.pt \
  --input data/raw/test/SEM --out /tmp/pred --zip submission.zip
```

Dataset: download from the competition page (login required; redistribution is
not permitted, so it is not included here).
