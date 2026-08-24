# Stage 0–1: Foundation + Supervised Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SEM→Depth 회귀 파이프라인의 기반(저장소·환경·데이터·모델·학습·추론·제출)을 구축하고, 시뮬레이션 지도학습 베이스라인으로 첫 리더보드 점수를 얻는다.

**Architecture:** `src/semdepth` 파이썬 패키지에 데이터/모델/학습/추론 모듈을 두고, 실험은 YAML config 1개 = 실험 1개로 관리하며 결과를 `experiments/results.csv`에 축적한다. 모델은 timm 사전학습 인코더 + 직접 구현한 U-Net 디코더. 데이터 의존 작업(Task 9+) 전까지는 합성 픽스처로 전체 파이프라인을 테스트한다.

**Tech Stack:** Python 3.12, uv, PyTorch(cu128, RTX 5070 Ti/sm_120), timm, numpy, pandas, Pillow, PyYAML, TensorBoard, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-samsung-3d-metrology-design.md`

## Global Constraints

- Python `>=3.12`, 패키지 관리는 **uv** (venv/lock 포함). 실행은 항상 `uv run ...`.
- PyTorch는 **cu128 인덱스**(`https://download.pytorch.org/whl/cu128`)에서 설치 — RTX 5070 Ti(Blackwell, sm_120) 지원 필수.
- 데이터(`data/`)와 대용량 산출물(`experiments/runs/`, `*.pt`, `*.zip`)은 **절대 커밋 금지** (gitignore).
- 모든 난수는 시드 고정(`seed_all(seed)`), config에 seed 명시.
- 각 태스크 커밋 전 `uv run pytest -q` 통과 필수.
- 깊이 값 규약: 내부 연산은 `[0,1]` 정규화(float32), 리더보드/리포트 지표는 **0–255 스케일 RMSE**(`rmse_255`)로 통일.
- Task 9~12는 실제 데이터(`data/raw/`) 필요 — 데이터 도착 전에는 Task 8까지만 진행.
- **실측 데이터 레이아웃(2026-08-24 압축 해제로 확정)** — 모든 데이터 코드는 이 구조를 따른다:
  - `simulation_data/SEM/Case_{1..4}/{80..84}/<base>_itr{0,1}.png`, `simulation_data/Depth/Case_{1..4}/{80..84}/<base>.png` (SEM 173,304 / Depth 86,652 — 대회 페이지의 "259,956"은 SEM+Depth 합계)
  - **동일 `<base>` 이름이 Case_1~4에 전부 반복**(21,663개 완전 중복) → 스플릿은 반드시 `<base>`(구조 정체성, group_id) 단위. 버킷(80~84)은 케이스 내에서 서로소.
  - `train/SEM/Depth_{110,120,130,140}/site_XXXXX/SEM_XXXXXX.png` 60,665장(페이지의 60,664는 오기), 라벨은 **site 단위**: `train/average_depth.csv` 2,059행, 헤더 `0,1`, 키 `depth_140_site_00233` ↔ 폴더 `Depth_140/site_00233`
  - `test/SEM/XXXXXX.png` 25,988장(평면), sample_submission 없음
  - 모든 이미지 48×72(폭×높이, 배열 shape (72,48)), 8-bit grayscale
- 대회 데이터는 재배포 금지 약관 — 리포트 그림에 원본 영상을 쓸 때도 소량 예시로 제한.

---

### Task 1: 저장소 스캐폴드 + uv 프로젝트 + pytest

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.python-version`, `src/semdepth/__init__.py`, `tests/test_scaffold.py`, `README.md`, `configs/.gitkeep`, `experiments/.gitkeep`, `scripts/.gitkeep`, `report/.gitkeep`, `notebooks/.gitkeep`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `semdepth` 패키지(이후 모든 태스크가 `from semdepth...`로 import), `uv run pytest` 실행 환경

- [ ] **Step 1: uv 프로젝트 초기화**

```bash
cd /home/park/workspace/2022-samsung-ai-challenge
uv init --bare --python 3.12
uv python pin 3.12
```

- [ ] **Step 2: pyproject.toml 작성** (uv init 결과를 아래로 교체)

```toml
[project]
name = "semdepth"
version = "0.1.0"
description = "SEM top-down image to depth map regression (Dacon 235954 practice)"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.7",
    "timm>=1.0",
    "numpy>=1.26",
    "pandas>=2.0",
    "pillow>=10.0",
    "pyyaml>=6.0",
    "tensorboard>=2.16",
    "tqdm>=4.66",
    "matplotlib>=3.8",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/semdepth"]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: .gitignore 작성**

```gitignore
.venv/
__pycache__/
*.pyc
data/
experiments/runs/
*.pt
*.zip
nohup.out
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: 패키지/디렉토리 뼈대 생성**

```bash
mkdir -p src/semdepth tests configs experiments scripts report notebooks data/raw
touch src/semdepth/__init__.py configs/.gitkeep experiments/.gitkeep scripts/.gitkeep report/.gitkeep notebooks/.gitkeep
```

`README.md` (초안 — Stage 4에서 완성):

```markdown
# semdepth — SEM-to-Depth Metrology (Dacon 235954 practice)

Predicting per-pixel depth maps from top-down SEM images
(2022 Samsung AI Challenge: 3D Metrology, practiced post-hoc).

Work in progress. See `docs/superpowers/specs/` for the design document.
```

- [ ] **Step 5: 실패하는 테스트 작성** — `tests/test_scaffold.py`

```python
def test_package_imports():
    import semdepth  # noqa: F401
```

- [ ] **Step 6: 테스트 실행 (실패 확인 → 통과 확인)**

Run: `uv sync && uv run pytest -q`
Expected: `test_package_imports` PASS (src 레이아웃이 pyproject에 등록돼 있으므로 sync 후 통과해야 함. import 실패 시 `[tool.hatch.build.targets.wheel]` 확인)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold uv project with semdepth package skeleton"
```

---

### Task 2: 의존성 설치 + GPU 스모크 테스트

**Files:**
- Create: `scripts/check_env.py`
- Modify: `uv.lock` (sync 산출물)

**Interfaces:**
- Consumes: Task 1의 pyproject
- Produces: CUDA 동작이 검증된 환경. 이후 모든 학습 태스크의 전제.

- [ ] **Step 1: 의존성 설치**

Run: `uv sync`
Expected: torch가 `pytorch-cu128` 인덱스에서 설치됨 (`uv pip list | grep torch`로 `+cu128` 또는 cu 표기 확인)

- [ ] **Step 2: scripts/check_env.py 작성**

```python
"""GPU/PyTorch smoke test: prints env info and times a matmul on CUDA."""
import time

import torch


def main() -> None:
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(dev)}")
    print(f"capability: {torch.cuda.get_device_capability(dev)}")
    x = torch.randn(4096, 4096, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        x = x @ x
        x = x / x.norm()
    torch.cuda.synchronize()
    print(f"10x 4096^2 matmul: {time.perf_counter() - t0:.3f}s")
    print("bf16 autocast:", end=" ")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = (x @ x).float().sum()
    print(f"ok ({y.item():.3e})")
    print("ENV OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 실행 및 증거 확인**

Run: `uv run python scripts/check_env.py`
Expected: `capability: (12, 0)` (sm_120), matmul 수 초 내 완료, 마지막 줄 `ENV OK`.
실패(`no kernel image` 등) 시: torch nightly cu128로 교체(`uv add torch --index pytorch-cu128=https://download.pytorch.org/whl/nightly/cu128`) 후 재시도하고, 사용한 버전을 커밋 메시지에 기록.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: install deps (torch cu128) and add GPU smoke test"
```

---

### Task 3: 지표(RMSE) + 결과 로거

**Files:**
- Create: `src/semdepth/metrics.py`, `src/semdepth/results.py`
- Test: `tests/test_metrics.py`, `tests/test_results.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `metrics.rmse(pred: torch.Tensor, target: torch.Tensor) -> float` — 같은 스케일의 두 텐서
  - `metrics.rmse_255(pred01: torch.Tensor, target01: torch.Tensor) -> float` — [0,1] 입력을 0–255 스케일로 환산한 RMSE
  - `results.append_result(csv_path: Path, row: dict) -> None` — 헤더 자동 생성/컬럼 합집합 유지 append

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_metrics.py`

```python
import torch

from semdepth.metrics import rmse, rmse_255


def test_rmse_known_value():
    pred = torch.tensor([0.0, 0.0])
    target = torch.tensor([3.0, 4.0])
    assert abs(rmse(pred, target) - (12.5 ** 0.5)) < 1e-6


def test_rmse_zero_for_identical():
    x = torch.rand(2, 1, 8, 8)
    assert rmse(x, x) == 0.0


def test_rmse_255_scales_unit_interval():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.full((1, 1, 4, 4), 0.5)
    assert abs(rmse_255(pred, target) - 127.5) < 1e-4
```

`tests/test_results.py`

```python
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
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `uv run pytest tests/test_metrics.py tests/test_results.py -q`
Expected: FAIL — `ModuleNotFoundError: semdepth.metrics`

- [ ] **Step 3: 구현** — `src/semdepth/metrics.py`

```python
import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMSE between two tensors of identical shape/scale."""
    return torch.sqrt(torch.mean((pred.float() - target.float()) ** 2)).item()


def rmse_255(pred01: torch.Tensor, target01: torch.Tensor) -> float:
    """RMSE in 0-255 units for tensors normalized to [0,1] (leaderboard scale)."""
    return 255.0 * rmse(pred01, target01)
```

`src/semdepth/results.py`

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_metrics.py tests/test_results.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/semdepth/metrics.py src/semdepth/results.py tests/test_metrics.py tests/test_results.py
git commit -m "feat: add rmse metrics and experiment results logger"
```

---

### Task 4: 합성 데이터 픽스처 (실데이터 폴더 구조 모사)

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_conftest.py`

**Interfaces:**
- Consumes: 없음
- Produces: pytest fixture `synth_root: Path` — 아래 구조의 임시 데이터셋. 이후 데이터/학습/추론 테스트가 전부 이것을 사용.

```
<root>/
  simulation_data/SEM/Case_{1,2}/{80,81}/struct_<bucket>_<j>_itr{0,1}.png  # 8-bit L, (72,48)
  simulation_data/Depth/Case_{1,2}/{80,81}/struct_<bucket>_<j>.png         # depth map
  train/SEM/Depth_{110,120}/site_XXXXX/SEM_XXXXXX.png                      # site당 3장
  train/average_depth.csv   # 헤더 "0,1"; 행: depth_110_site_00000,<float> (site 단위 라벨)
  test/SEM/000000.png ...   # 평면
```

실데이터의 핵심 성질을 그대로 재현한다: **같은 구조 이름이 두 Case에 반복**되고(스플릿
누수 테스트의 근거), 같은 Case 안에서 버킷(80/81)끼리는 이름이 서로소이며, depth map은
Case가 달라도 동일하고, 실제 train 라벨은 이미지가 아니라 site 단위다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_conftest.py`

```python
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
        same = [p for p in depths if p.name == n]
        assert len(same) == 2
        a, b = (np.array(Image.open(p)) for p in same)
        assert np.array_equal(a, b)  # identical depth map regardless of case
    df = pd.read_csv(synth_root / "train" / "average_depth.csv")
    assert list(df.columns) == ["0", "1"]
    assert len(df) == 3  # one label row per site, not per image
    assert str(df.iloc[0, 0]).startswith("depth_")
    assert len(list((synth_root / "train" / "SEM").rglob("*.png"))) == 9
    assert len(list((synth_root / "test" / "SEM").glob("*.png"))) == 4
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `uv run pytest tests/test_conftest.py -q`
Expected: FAIL — `fixture 'synth_root' not found`

- [ ] **Step 3: 구현** — `tests/conftest.py`

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

HW = (72, 48)  # (rows, cols) — matches the real data (PIL reports size 48x72)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _depth_pattern(rng: np.random.Generator) -> np.ndarray:
    """Smooth radial 'hole' pattern with random depth scale — learnable signal."""
    h, w = HW
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2 + rng.uniform(-5, 5), w / 2 + rng.uniform(-8, 8)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    depth = 255 - rng.uniform(100, 220) * np.exp(-(r ** 2) / rng.uniform(80, 300))
    return np.clip(depth, 0, 255)


def make_synth_data(root: Path) -> Path:
    rng = np.random.default_rng(0)
    # simulation: same structure names (and identical depth maps) repeated across cases
    for bucket in ["80", "81"]:
        for j in range(3):
            name = f"struct_{bucket}_{j}"
            depth = _depth_pattern(rng)
            for case in ["Case_1", "Case_2"]:
                _save_gray(
                    root / "simulation_data" / "Depth" / case / bucket / f"{name}.png", depth
                )
                for k in range(2):  # two noise realizations of the same structure
                    sem = np.clip(depth + rng.normal(0, 12, HW), 0, 255)
                    _save_gray(
                        root / "simulation_data" / "SEM" / case / bucket / f"{name}_itr{k}.png",
                        sem,
                    )
    # real train: 3 sites x 3 images, one average-depth label per SITE
    rows, n = [], 0
    for bucket, site in [("110", "site_00000"), ("110", "site_00001"), ("120", "site_00002")]:
        depth = _depth_pattern(rng)
        for _ in range(3):
            sem = np.clip(depth + rng.normal(0, 20, HW), 0, 255)
            _save_gray(
                root / "train" / "SEM" / f"Depth_{bucket}" / site / f"SEM_{n:06d}.png", sem
            )
            n += 1
        rows.append({"0": f"depth_{bucket}_{site}", "1": float(depth.mean())})
    pd.DataFrame(rows).to_csv(root / "train" / "average_depth.csv", index=False)
    for i in range(4):
        sem = np.clip(_depth_pattern(rng) + rng.normal(0, 20, HW), 0, 255)
        _save_gray(root / "test" / "SEM" / f"{i:06d}.png", sem)
    return root


@pytest.fixture()
def synth_root(tmp_path: Path) -> Path:
    return make_synth_data(tmp_path / "data")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_conftest.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_conftest.py
git commit -m "test: add synthetic dataset fixture mirroring competition layout"
```

---

### Task 5: 데이터 모듈 (페어링·그룹 스플릿·Dataset)

**Files:**
- Create: `src/semdepth/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `synth_root` 픽스처 (Task 4)
- Produces:
  - `SimPair` dataclass: `group_id: str`(구조 정체성 = depth 파일 stem — Case가 달라도 동일), `case: str`, `bucket: str`, `sem_paths: tuple[Path, ...]`(itr 순서), `depth_path: Path`
  - `list_sim_pairs(sem_root: Path, depth_root: Path) -> list[SimPair]` — 재귀 탐색; SEM `<case>/<bucket>/<base>_itr<k>.png` ↔ Depth `<case>/<bucket>/<base>.png`를 상대 경로로 매칭; 이름 규칙 위반·짝 없는 SEM/Depth는 ValueError
  - `split_pairs(pairs: list[SimPair], val_fraction: float, seed: int) -> tuple[list[SimPair], list[SimPair]]` — **group_id(구조 이름) 단위 분할**: 같은 구조는 Case·itr가 달라도 전부 같은 쪽 (실데이터에서 동일 구조가 Case_1~4에 반복되므로 그보다 가는 단위는 전부 누수)
  - `load_image01(path: Path) -> np.ndarray` — float32 [H,W], [0,1]
  - `SimDataset(pairs, augment: bool = False)` — 항목: `{"image": FloatTensor[1,H,W], "target": FloatTensor[1,H,W]}`; 길이 = itr 포함 SEM 장수
  - `RealDataset(sem_root: Path, csv_path: Path)` — csv 키 `depth_<bucket>_site_<id>` ↔ 폴더 `Depth_<bucket>/site_<id>`; 항목: `{"image": FloatTensor[1,H,W], "avg_depth": float(site 평균 깊이, 0-255 스케일 원값), "name": str}`; 길이 = 모든 site의 모든 이미지 수 (site 라벨이 그 site의 각 이미지에 공유됨)
  - `ImageDirDataset(sem_dir)` — 항목: `{"image": FloatTensor[1,H,W], "name": str}` (추론용, 평면 폴더, 이름순 정렬)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_data.py`

```python
import pytest
import torch

from semdepth.data import (
    ImageDirDataset,
    RealDataset,
    SimDataset,
    list_sim_pairs,
    split_pairs,
)


def _pairs(synth_root):
    return list_sim_pairs(
        synth_root / "simulation_data" / "SEM", synth_root / "simulation_data" / "Depth"
    )


def test_pairing(synth_root):
    pairs = _pairs(synth_root)
    assert len(pairs) == 12  # 2 cases x 2 buckets x 3 structs
    assert all(len(p.sem_paths) == 2 for p in pairs)
    assert all(p.depth_path.stem == p.group_id for p in pairs)
    by_group: dict[str, set] = {}
    for p in pairs:
        by_group.setdefault(p.group_id, set()).add(p.case)
    assert len(by_group) == 6  # same structure appears once per case
    assert all(cases == {"Case_1", "Case_2"} for cases in by_group.values())


def test_pairing_rejects_orphan_sem(synth_root):
    orphan = synth_root / "simulation_data" / "SEM" / "Case_1" / "80" / "orphan_itr0.png"
    orphan.write_bytes((synth_root / "test" / "SEM" / "000000.png").read_bytes())
    with pytest.raises(ValueError):
        _pairs(synth_root)


def test_pairing_rejects_bad_sem_name(synth_root):
    bad = synth_root / "simulation_data" / "SEM" / "Case_1" / "80" / "noitr.png"
    bad.write_bytes((synth_root / "test" / "SEM" / "000000.png").read_bytes())
    with pytest.raises(ValueError, match="no _itr suffix"):
        _pairs(synth_root)


def test_pairing_rejects_misplaced_depth(synth_root):
    depth_dir = synth_root / "simulation_data" / "Depth" / "Case_1" / "80"
    victim = sorted(depth_dir.glob("*.png"))[0]
    victim.rename(depth_dir / "renamed.png")  # count unchanged, per-item lookup fails
    with pytest.raises(ValueError, match="depth map missing"):
        _pairs(synth_root)


def test_augment_flips_image_and_target_together(synth_root, monkeypatch):
    aug_ds = SimDataset(_pairs(synth_root), augment=True)
    base_ds = SimDataset(_pairs(synth_root), augment=False)
    monkeypatch.setattr("semdepth.data.random.random", lambda: 0.0)  # force both flips
    aug, orig = aug_ds[0], base_ds[0]
    assert torch.equal(aug["image"], torch.flip(orig["image"], [1, 2]))
    assert torch.equal(aug["target"], torch.flip(orig["target"], [1, 2]))


def test_group_split_no_leak(synth_root):
    pairs = _pairs(synth_root)
    tr, va = split_pairs(pairs, val_fraction=0.34, seed=7)
    assert len(tr) + len(va) == 12
    tr_groups = {p.group_id for p in tr}
    va_groups = {p.group_id for p in va}
    assert tr_groups.isdisjoint(va_groups)  # a structure never crosses the boundary
    assert len(va_groups) == 2 and len(va) == 4  # 2 groups x 2 cases
    tr2, va2 = split_pairs(pairs, val_fraction=0.34, seed=7)
    assert sorted(p.depth_path for p in va2) == sorted(p.depth_path for p in va)


def test_sim_dataset_items(synth_root):
    ds = SimDataset(_pairs(synth_root))
    assert len(ds) == 24
    item = ds[0]
    assert item["image"].shape == (1, 72, 48) and item["image"].dtype == torch.float32
    assert item["target"].shape == (1, 72, 48)
    assert 0.0 <= item["image"].min() and item["image"].max() <= 1.0


def test_real_dataset_site_labels(synth_root):
    ds = RealDataset(synth_root / "train" / "SEM", synth_root / "train" / "average_depth.csv")
    assert len(ds) == 9  # every image of every site
    items = [ds[i] for i in range(len(ds))]
    assert all(it["image"].shape == (1, 72, 48) for it in items)
    assert all(0.0 <= it["avg_depth"] <= 255.0 for it in items)
    assert len({round(it["avg_depth"], 6) for it in items}) == 3  # one label per site
    assert all(it["name"].endswith(".png") for it in items)


def test_image_dir_dataset_sorted(synth_root):
    ds = ImageDirDataset(synth_root / "test" / "SEM")
    names = [ds[i]["name"] for i in range(len(ds))]
    assert names == sorted(names) and len(names) == 4
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `uv run pytest tests/test_data.py -q`
Expected: FAIL — `ModuleNotFoundError: semdepth.data`

- [ ] **Step 3: 구현** — `src/semdepth/data.py`

```python
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

_ITR_RE = re.compile(r"^(?P<base>.+)_itr(?P<k>\d+)$")
_SITE_RE = re.compile(r"^depth_(?P<bucket>\d+)_site_(?P<site>\d+)$")


@dataclass(frozen=True)
class SimPair:
    group_id: str  # structure identity: depth-map stem, shared across cases
    case: str
    bucket: str
    sem_paths: tuple[Path, ...]
    depth_path: Path


def load_image01(path: Path) -> np.ndarray:
    """Load 8-bit grayscale PNG as float32 [H,W] in [0,1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def list_sim_pairs(sem_root: Path, depth_root: Path) -> list[SimPair]:
    """Pair SEM <case>/<bucket>/<base>_itr<k>.png with Depth <case>/<bucket>/<base>.png."""
    sem_root, depth_root = Path(sem_root), Path(depth_root)
    by_rel: dict[Path, list[Path]] = {}
    for p in sorted(sem_root.rglob("*.png")):
        m = _ITR_RE.match(p.stem)
        if m is None:
            raise ValueError(f"unexpected SEM name (no _itr suffix): {p}")
        rel = p.relative_to(sem_root).parent / f"{m['base']}.png"
        by_rel.setdefault(rel, []).append(p)
    n_depth = sum(1 for _ in depth_root.rglob("*.png"))
    if n_depth != len(by_rel):
        raise ValueError(f"{n_depth} depth maps but {len(by_rel)} SEM groups")
    pairs = []
    for rel, sems in sorted(by_rel.items()):
        depth = depth_root / rel
        if not depth.exists():
            raise ValueError(f"depth map missing: {depth}")
        parts = rel.parts
        case = parts[0] if len(parts) > 1 else ""
        bucket = parts[1] if len(parts) > 2 else ""
        pairs.append(SimPair(rel.stem, case, bucket, tuple(sorted(sems)), depth))
    return pairs


def split_pairs(
    pairs: list[SimPair], val_fraction: float, seed: int
) -> tuple[list[SimPair], list[SimPair]]:
    """Structure-level split: a group_id never appears on both sides.

    The same structure is simulated under every case (and twice per case via
    itr0/itr1); splitting by anything finer would leak it across the boundary.
    """
    groups = sorted({p.group_id for p in pairs})
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction))
    val_groups = set(groups[:n_val])
    train = [p for p in pairs if p.group_id not in val_groups]
    val = [p for p in pairs if p.group_id in val_groups]
    return train, val


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.copy()).unsqueeze(0)


class SimDataset(Dataset):
    """(SEM image, depth map) pairs; one item per SEM realization (itr)."""

    def __init__(self, pairs: list[SimPair], augment: bool = False):
        self.items = [(p.sem_paths[k], p.depth_path) for p in pairs for k in range(len(p.sem_paths))]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        sem_path, depth_path = self.items[i]
        image, target = load_image01(sem_path), load_image01(depth_path)
        if self.augment:
            if random.random() < 0.5:
                image, target = image[:, ::-1], target[:, ::-1]
            if random.random() < 0.5:
                image, target = image[::-1, :], target[::-1, :]
        return {"image": _to_tensor(image), "target": _to_tensor(target)}


class RealDataset(Dataset):
    """Real-domain SEM images; the average-depth label is shared per site."""

    def __init__(self, sem_root: Path, csv_path: Path):
        df = pd.read_csv(csv_path)
        self.records: list[tuple[Path, float]] = []
        for key, avg in zip(df.iloc[:, 0], df.iloc[:, 1]):
            m = _SITE_RE.match(str(key))
            if m is None:
                raise ValueError(f"unexpected site key in csv: {key}")
            site_dir = Path(sem_root) / f"Depth_{m['bucket']}" / f"site_{m['site']}"
            pngs = sorted(site_dir.glob("*.png"))
            if not pngs:
                raise ValueError(f"no images for site {key}: {site_dir}")
            self.records += [(p, float(avg)) for p in pngs]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        path, avg = self.records[i]
        return {"image": _to_tensor(load_image01(path)), "avg_depth": avg, "name": path.name}


class ImageDirDataset(Dataset):
    """All PNGs in a directory, sorted by name (inference input)."""

    def __init__(self, sem_dir: Path):
        self.paths = sorted(Path(sem_dir).glob("*.png"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        p = self.paths[i]
        return {"image": _to_tensor(load_image01(p)), "name": p.name}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_data.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/semdepth/data.py tests/test_data.py
git commit -m "feat: add sim pairing, leak-safe group split, and datasets"
```

---

### Task 6: 모델 — timm 인코더 + U-Net 디코더 (크기 무관)

**Files:**
- Create: `src/semdepth/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: 없음 (독립)
- Produces: `UnetTimm(encoder_name: str = "resnet18", pretrained: bool = True)` — `forward(x: FloatTensor[B,1,H,W]) -> FloatTensor[B,1,H,W]`, 출력은 sigmoid로 [0,1]. 임의 H,W 지원(내부에서 32의 배수로 replicate 패딩 후 원복).

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_model.py`

```python
import torch

from semdepth.model import UnetTimm


def _model():
    return UnetTimm(encoder_name="resnet18", pretrained=False)


def test_output_shape_and_range():
    m = _model().eval()
    for h, w in [(48, 72), (64, 64), (40, 56)]:
        x = torch.rand(2, 1, h, w)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 1, h, w)
        assert 0.0 <= y.min() and y.max() <= 1.0


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = _model().train()
    x = torch.rand(4, 1, 48, 72)
    t = torch.rand(4, 1, 48, 72) * 0.5 + 0.25
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(m(x), t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"no learning: {losses[0]:.4f} -> {losses[-1]:.4f}"
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `uv run pytest tests/test_model.py -q`
Expected: FAIL — `ModuleNotFoundError: semdepth.model`

- [ ] **Step 3: 구현** — `src/semdepth/model.py`

```python
import timm
import torch
import torch.nn.functional as F
from torch import nn


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UnetTimm(nn.Module):
    """U-Net with a timm backbone encoder; grayscale in, [0,1] depth map out."""

    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, features_only=True, pretrained=pretrained, in_chans=1
        )
        enc_chs = self.encoder.feature_info.channels()  # shallow -> deep
        dec_chs = [256, 128, 64, 32, 16][-len(enc_chs) :]
        blocks, in_ch = [], enc_chs[-1]
        skips = enc_chs[:-1][::-1] + [0]  # deepest skip first, last block has none
        for skip_ch, out_ch in zip(skips, dec_chs):
            blocks.append(_DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        feats = self.encoder(x)
        out, skips = feats[-1], feats[:-1][::-1] + [None]
        for block, skip in zip(self.decoder, skips):
            out = block(out, skip)
        if out.shape[-2:] != x.shape[-2:]:  # encoders whose first feature is stride>2
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.sigmoid(self.head(out))
        return out[..., :h, :w]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_model.py -q`
Expected: PASS (2 tests; overfit 테스트는 CPU에서 1분 내)

- [ ] **Step 5: Commit**

```bash
git add src/semdepth/model.py tests/test_model.py
git commit -m "feat: add size-agnostic UnetTimm depth regression model"
```

---

### Task 7: 학습 루프 (config 주도, AMP, 체크포인트, 결과 기록)

**Files:**
- Create: `src/semdepth/train.py`, `scripts/train.py`, `configs/baseline.yaml`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Task 3 `rmse_255`, `append_result` / Task 5 `list_sim_pairs`, `split_pairs`, `SimDataset` / Task 6 `UnetTimm`
- Produces:
  - `train.seed_all(seed: int) -> None`
  - `train.run_training(cfg: dict) -> dict` — 학습 실행 후 결과 행(dict) 반환.
    부수효과: `<runs_dir>/<run_name>/best.pt` (최저 val RMSE 시점 state_dict), TensorBoard 로그, `results_csv`에 행 append.
  - CLI: `uv run python scripts/train.py -c configs/baseline.yaml`
  - config 스키마(YAML) — 아래 `configs/baseline.yaml`이 규범.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_train.py`

```python
from pathlib import Path

import pandas as pd
import pytest

from semdepth.train import run_training


def _tiny_cfg(synth_root: Path, tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "data": {
            "sim_sem_dir": str(synth_root / "simulation_data" / "SEM"),
            "sim_depth_dir": str(synth_root / "simulation_data" / "Depth"),
            "val_fraction": 0.34,
        },
        "model": {"encoder": "resnet18", "pretrained": False},
        "train": {
            "epochs": 2, "batch_size": 4, "lr": 1.0e-3, "weight_decay": 0.0,
            "num_workers": 0, "amp": False, "augment": True, "device": "cpu",
        },
        "out": {
            "run_name": "smoke",
            "runs_dir": str(tmp_path / "runs"),
            "results_csv": str(tmp_path / "results.csv"),
        },
    }


def test_run_training_end_to_end(synth_root, tmp_path):
    row = run_training(_tiny_cfg(synth_root, tmp_path))
    assert (tmp_path / "runs" / "smoke" / "best.pt").exists()
    df = pd.read_csv(tmp_path / "results.csv")
    assert df.loc[0, "run_name"] == "smoke"
    assert 0.0 < row["best_val_rmse255"] < 255.0


def test_run_training_rejects_empty_train_loader(synth_root, tmp_path):
    cfg = _tiny_cfg(synth_root, tmp_path)
    cfg["train"]["batch_size"] = 999  # larger than the whole synthetic train set
    with pytest.raises(ValueError, match="no training batches"):
        run_training(cfg)
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `uv run pytest tests/test_train.py -q`
Expected: FAIL — `ModuleNotFoundError: semdepth.train`

- [ ] **Step 3: 구현** — `src/semdepth/train.py`

```python
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from semdepth.data import SimDataset, list_sim_pairs, split_pairs
from semdepth.metrics import rmse_255
from semdepth.model import UnetTimm
from semdepth.results import append_result


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    se_sum, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        t = batch["target"].to(device, non_blocking=True)
        p = model(x)
        se_sum += ((p - t) ** 2).sum().item()
        n += t.numel()
    return 255.0 * (se_sum / n) ** 0.5


def run_training(cfg: dict) -> dict:
    seed_all(cfg["seed"])
    d, tr, m, out = cfg["data"], cfg["train"], cfg["model"], cfg["out"]
    device = tr.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    pairs = list_sim_pairs(Path(d["sim_sem_dir"]), Path(d["sim_depth_dir"]))
    train_pairs, val_pairs = split_pairs(pairs, d["val_fraction"], cfg["seed"])
    train_ds = SimDataset(train_pairs, augment=tr["augment"])
    val_ds = SimDataset(val_pairs, augment=False)
    train_dl = DataLoader(
        train_ds, batch_size=tr["batch_size"], shuffle=True,
        num_workers=tr["num_workers"], pin_memory=(device == "cuda"), drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=tr["batch_size"], shuffle=False,
        num_workers=tr["num_workers"], pin_memory=(device == "cuda"),
    )
    if len(train_dl) == 0:
        raise ValueError(
            f"no training batches: {len(train_ds)} images < batch_size {tr['batch_size']}"
        )

    model = UnetTimm(m["encoder"], pretrained=m["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tr["epochs"] * max(1, len(train_dl))
    )

    run_dir = Path(out["runs_dir"]) / out["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    best, t0 = float("inf"), time.time()

    for epoch in range(tr["epochs"]):
        model.train()
        for step, batch in enumerate(tqdm(train_dl, desc=f"epoch {epoch}", leave=False)):
            x = batch["image"].to(device, non_blocking=True)
            t = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=tr["amp"]):
                loss = torch.nn.functional.l1_loss(model(x), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 50 == 0:
                writer.add_scalar("train/l1", loss.item(), epoch * len(train_dl) + step)
        val_rmse = evaluate(model, val_dl, device)
        writer.add_scalar("val/rmse255", val_rmse, epoch)
        print(f"epoch {epoch}: val rmse255 = {val_rmse:.4f}")
        if val_rmse < best:
            best = val_rmse
            torch.save(model.state_dict(), run_dir / "best.pt")

    row = {
        "run_name": out["run_name"],
        "encoder": m["encoder"],
        "epochs": tr["epochs"],
        "batch_size": tr["batch_size"],
        "lr": tr["lr"],
        "augment": tr["augment"],
        "seed": cfg["seed"],
        "n_train_imgs": len(train_ds),
        "n_val_imgs": len(val_ds),
        "best_val_rmse255": round(best, 4),
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    append_result(Path(out["results_csv"]), row)
    writer.close()
    return row
```

`scripts/train.py`

```python
"""Run one training experiment from a YAML config."""
import argparse

import yaml

from semdepth.train import run_training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    row = run_training(cfg)
    print(row)


if __name__ == "__main__":
    main()
```

`configs/baseline.yaml` (경로는 Task 9에서 실데이터 레이아웃에 맞게 확정)

```yaml
seed: 42
data:
  sim_sem_dir: data/raw/simulation_data/SEM
  sim_depth_dir: data/raw/simulation_data/Depth
  val_fraction: 0.1
model:
  encoder: resnet34
  pretrained: true
train:
  epochs: 8
  batch_size: 256
  lr: 3.0e-4
  weight_decay: 1.0e-4
  num_workers: 8
  amp: true
  augment: true
out:
  run_name: s1_baseline_r34
  runs_dir: experiments/runs
  results_csv: experiments/results.csv
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_train.py -q`
Expected: PASS (CPU 2 epoch, 수십 초 내)

- [ ] **Step 5: 전체 테스트 + Commit**

```bash
uv run pytest -q
git add src/semdepth/train.py scripts/train.py configs/baseline.yaml tests/test_train.py
git commit -m "feat: add config-driven training loop with checkpoint and results logging"
```

---

### Task 8: 추론·제출 zip·Real 프록시 지표

**Files:**
- Create: `src/semdepth/infer.py`, `src/semdepth/proxy.py`, `scripts/predict.py`
- Test: `tests/test_infer.py`, `tests/test_proxy.py`

**Interfaces:**
- Consumes: Task 5 `ImageDirDataset`, `RealDataset` / Task 6 `UnetTimm`
- Produces:
  - `infer.predict_dir(model, sem_dir: Path, out_dir: Path, device: str, batch_size: int = 256, flip_tta: bool = False) -> int` — 입력 PNG마다 같은 파일명의 uint8 depth PNG 저장, 개수 반환
  - `infer.make_submission_zip(pred_dir: Path, zip_path: Path) -> int` — pred_dir의 PNG를 평평하게 zip, 개수 반환
  - `proxy.real_proxy_rmse(model, dataset: RealDataset, device: str, batch_size: int = 256) -> float` — `mean(pred)*255` vs `avg_depth` 의 RMSE (제출 없는 도메인 갭 지표)
  - CLI: `uv run python scripts/predict.py -c configs/baseline.yaml --ckpt experiments/runs/<run>/best.pt --input <sem_dir> --out <pred_dir> [--zip <zip_path>] [--tta]`

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_infer.py`

```python
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
```

`tests/test_proxy.py`

```python
import torch

from semdepth.data import RealDataset
from semdepth.proxy import real_proxy_rmse


class _ConstModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


def test_proxy_exact_for_constant_model(synth_root):
    ds = RealDataset(synth_root / "train" / "SEM", synth_root / "train" / "average_depth.csv")
    model = _ConstModel(0.5)  # predicts mean depth 127.5 everywhere
    got = real_proxy_rmse(model, ds, device="cpu")
    avgs = torch.tensor([ds[i]["avg_depth"] for i in range(len(ds))])
    expected = torch.sqrt(torch.mean((127.5 - avgs) ** 2)).item()
    assert abs(got - expected) < 1e-3
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

Run: `uv run pytest tests/test_infer.py tests/test_proxy.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현** — `src/semdepth/infer.py`

```python
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from semdepth.data import ImageDirDataset


@torch.no_grad()
def predict_dir(
    model: torch.nn.Module,
    sem_dir: Path,
    out_dir: Path,
    device: str,
    batch_size: int = 256,
    flip_tta: bool = False,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    ds = ImageDirDataset(Path(sem_dir))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4)
    n = 0
    for batch in tqdm(dl, desc="predict"):
        x = batch["image"].to(device)
        pred = model(x)
        if flip_tta:
            pred = (pred + torch.flip(model(torch.flip(x, [-1])), [-1])) / 2
        arr = (pred.clamp(0, 1) * 255).round().byte().cpu().numpy()
        for name, a in zip(batch["name"], arr):
            Image.fromarray(a[0].astype(np.uint8), mode="L").save(out_dir / name)
            n += 1
    return n


def make_submission_zip(pred_dir: Path, zip_path: Path) -> int:
    paths = sorted(Path(pred_dir).glob("*.png"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    return len(paths)
```

`src/semdepth/proxy.py`

```python
import torch
from torch.utils.data import DataLoader

from semdepth.data import RealDataset


@torch.no_grad()
def real_proxy_rmse(
    model: torch.nn.Module, dataset: RealDataset, device: str, batch_size: int = 256
) -> float:
    """RMSE between predicted-map means (0-255) and given average depths.

    Submission-free proxy that quantifies the sim-to-real domain gap.
    """
    model = model.to(device).eval()
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    se_sum, n = 0.0, 0
    for batch in dl:
        x = batch["image"].to(device)
        pred_mean = model(x).mean(dim=(1, 2, 3)) * 255.0
        avg = batch["avg_depth"].to(device).float()
        se_sum += ((pred_mean - avg) ** 2).sum().item()
        n += len(avg)
    return (se_sum / n) ** 0.5
```

`scripts/predict.py`

```python
"""Predict depth maps for a directory of SEM PNGs; optionally build submission zip."""
import argparse
from pathlib import Path

import torch
import yaml

from semdepth.infer import make_submission_zip, predict_dir
from semdepth.model import UnetTimm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = UnetTimm(cfg["model"]["encoder"], pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = predict_dir(model, Path(args.input), Path(args.out), device, flip_tta=args.tta)
    print(f"wrote {n} depth maps to {args.out}")
    if args.zip:
        nz = make_submission_zip(Path(args.out), Path(args.zip))
        print(f"zipped {nz} files -> {args.zip}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인 + Commit**

Run: `uv run pytest -q` → 전체 PASS

```bash
git add src/semdepth/infer.py src/semdepth/proxy.py scripts/predict.py tests/test_infer.py tests/test_proxy.py
git commit -m "feat: add inference, submission zip writer, and real-domain proxy metric"
```

---

### Task 9: [데이터 필요] 데이터 인테이크 & 가정 검증

**선행 조건: 사용자가 `data/raw/`에 대회 zip을 놓아두었을 것.**

**Files:**
- Create: `scripts/inspect_data.py`, `docs/data-notes.md`
- Modify: `configs/baseline.yaml` (실제 경로/파일명 반영), 필요시 `tests/conftest.py`·`src/semdepth/data.py` (실제 레이아웃과 가정이 다를 경우)

**Interfaces:**
- Consumes: Task 5 `list_sim_pairs` (실데이터 검증에 사용)
- Produces: 확정된 데이터 레이아웃(`configs/baseline.yaml`의 `data:` 경로가 실데이터를 가리킴), `docs/data-notes.md` (이후 모든 태스크·리포트가 참조하는 데이터 사실 기록)

- [ ] **Step 1: 압축 해제**

```bash
cd data/raw && for z in *.zip; do unzip -n -q "$z"; done && find . -maxdepth 3 -type d | head -50
```

- [ ] **Step 2: scripts/inspect_data.py 작성**

```python
"""Inventory the raw competition data and verify plan assumptions."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def describe_image_dir(d: Path, sample: int = 200) -> str:
    paths = sorted(d.glob("*.png"))
    sizes, modes = Counter(), Counter()
    step = max(1, len(paths) // sample)
    for p in paths[::step]:
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
    return f"{d}: {len(paths)} png | sizes {dict(sizes)} | modes {dict(modes)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    root = Path(args.root)
    print("== directory tree (depth<=3) ==")
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts) <= 3):
        n_png = len(list(d.glob("*.png")))
        print(f"  {d.relative_to(root)}  ({n_png} png)" if n_png else f"  {d.relative_to(root)}/")
    print("== image dirs ==")
    for d in sorted({p.parent for p in root.rglob("*.png")}):
        print(" ", describe_image_dir(d))
    print("== csv files ==")
    for c in sorted(root.rglob("*.csv")):
        df = pd.read_csv(c)
        print(f"  {c.relative_to(root)}: shape={df.shape} cols={list(df.columns)[:8]}")
        print(df.head(3).to_string(index=False))
    print("== sample pixel stats ==")
    for d in sorted({p.parent for p in root.rglob("*.png")}):
        ps = sorted(d.glob("*.png"))[:50]
        arrs = np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.float32) for p in ps])
        print(f"  {d.relative_to(root)}: mean={arrs.mean():.1f} std={arrs.std():.1f} "
              f"min={arrs.min():.0f} max={arrs.max():.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 실행하고 스펙의 가정과 대조**

Run: `uv run python scripts/inspect_data.py | tee /tmp/inspect_out.txt`
확인 항목(각각 결과를 `docs/data-notes.md`에 기록):
1. 시뮬레이션 SEM 259,956장 / Depth 129,978장(=SEM의 절반), 실제 train 60,664장, test 25,988장 — 수량 일치?
2. SEM↔Depth 매칭이 예외 없이 되는지 (`list_sim_pairs`를 실데이터에 실행) + **케이스 내 고유 stem 수 == 해당 케이스 depth 수(21,663)** 로 버킷 간 group_id 충돌이 없음을 확인 (group_id가 bucket을 버리므로, 충돌 시 서로 다른 구조가 한 그룹으로 묶임)
3. 이미지 크기·비트심도, `average_depth.csv`의 실제 컬럼명
4. 제출 형식: sample_submission 존재 여부, zip 내부 구조(평면/폴더)
5. depth 값 스케일(0–255인지), 값이 작을수록 깊다는 방향성

- [ ] **Step 4: 가정이 다르면 코드·픽스처를 실데이터에 맞게 수정**

수정 대상은 세 곳으로 국한된다: `configs/baseline.yaml`의 `data:` 경로, `RealDataset`의 기본 컬럼명 인자, `tests/conftest.py`의 폴더/파일명 규칙. 수정 후 `uv run pytest -q` 전체 통과 확인.

- [ ] **Step 5: docs/data-notes.md 작성 후 Commit**

`docs/data-notes.md`에 Step 3의 확인 항목 5개의 실측 결과를 표로 기록.

```bash
git add scripts/inspect_data.py docs/data-notes.md configs/baseline.yaml tests/ src/
git commit -m "docs: record verified dataset layout and align config/fixtures to it"
```

---

### Task 10: [데이터 필요] EDA 리포트 (도메인 갭 정량화 포함)

**Files:**
- Create: `scripts/eda.py`, `report/figures/` (산출 그림), `report/eda.md`

주의(레이아웃 확정 반영): 아래 스크립트 코드에서 이미지 샘플링은 중첩 구조를 위해
`glob` 대신 `rglob`을 사용하고, 그림 4(밝기 vs 평균깊이)는 **site 단위**(site 이미지들의
평균 밝기 vs site 평균 깊이)로 그린다. 그림 5의 itr 페어는 `list_sim_pairs` 결과를
그대로 사용하므로 변경 없음. 실행은 controller가 인라인으로 수행하며 이때 반영한다.

**Interfaces:**
- Consumes: Task 5 `load_image01`, `list_sim_pairs`, `RealDataset` / Task 9의 확정 경로
- Produces: `report/figures/*.png` 6종 + `report/eda.md` — Stage 2 기법 선택과 리포트 3장(데이터 분석)의 근거 자료

- [ ] **Step 1: scripts/eda.py 작성** — 아래 그림을 `report/figures/`에 저장

```python
"""EDA: quantify the sim-to-real domain gap and basic physics sanity checks."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from semdepth.data import list_sim_pairs, load_image01  # noqa: E402


def sample_arrays(d: Path, n: int = 2000, seed: int = 0) -> np.ndarray:
    paths = sorted(d.glob("*.png"))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return np.stack([np.asarray(Image.open(paths[i]).convert("L"), np.float32) for i in pick])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/baseline.yaml")
    ap.add_argument("--real-sem-dir", required=True)
    ap.add_argument("--real-csv", required=True)
    ap.add_argument("--test-sem-dir", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = Path("report/figures")
    out.mkdir(parents=True, exist_ok=True)

    sim = sample_arrays(Path(cfg["data"]["sim_sem_dir"]))
    real = sample_arrays(Path(args.real_sem_dir))
    test = sample_arrays(Path(args.test_sem_dir))

    # 1. intensity histograms: the domain gap, visualized
    plt.figure(figsize=(7, 4))
    for arr, label in [(sim, "sim SEM"), (real, "real train SEM"), (test, "real test SEM")]:
        plt.hist(arr.ravel(), bins=64, range=(0, 255), density=True, alpha=0.5, label=label)
    plt.legend(); plt.title("Pixel intensity distribution by domain")
    plt.savefig(out / "01_intensity_hist.png", dpi=150, bbox_inches="tight"); plt.close()

    # 2. per-image mean brightness distributions
    plt.figure(figsize=(7, 4))
    for arr, label in [(sim, "sim"), (real, "real train"), (test, "real test")]:
        plt.hist(arr.mean(axis=(1, 2)), bins=50, density=True, alpha=0.5, label=label)
    plt.legend(); plt.title("Per-image mean brightness by domain")
    plt.savefig(out / "02_mean_brightness.png", dpi=150, bbox_inches="tight"); plt.close()

    # 3. average_depth label distribution
    df = pd.read_csv(args.real_csv)
    depth_col = [c for c in df.columns if "depth" in c.lower()][0]
    plt.figure(figsize=(7, 4))
    plt.hist(df[depth_col], bins=60)
    plt.title(f"Real train {depth_col} distribution")
    plt.savefig(out / "03_avg_depth_hist.png", dpi=150, bbox_inches="tight"); plt.close()

    # 4. physics hook: mean brightness vs average depth (secondary-electron escape)
    name_col = [c for c in df.columns if c != depth_col][0]
    sub = df.sample(min(3000, len(df)), random_state=0)
    means = [load_image01(Path(args.real_sem_dir) / str(n)).mean() * 255 for n in sub[name_col]]
    plt.figure(figsize=(5, 5))
    plt.scatter(means, sub[depth_col], s=2, alpha=0.3)
    plt.xlabel("image mean brightness"); plt.ylabel(depth_col)
    corr = np.corrcoef(means, sub[depth_col])[0, 1]
    plt.title(f"Brightness vs avg depth (r={corr:.3f})")
    plt.savefig(out / "04_brightness_vs_depth.png", dpi=150, bbox_inches="tight"); plt.close()

    # 5. itr0 vs itr1: noise level between realizations of the same structure
    pairs = list_sim_pairs(Path(cfg["data"]["sim_sem_dir"]), Path(cfg["data"]["sim_depth_dir"]))
    diffs = []
    for p in pairs[:: max(1, len(pairs) // 500)]:
        a, b = (load_image01(q) * 255 for q in p.sem_paths[:2])
        diffs.append(np.abs(a - b).mean())
    plt.figure(figsize=(7, 4))
    plt.hist(diffs, bins=50)
    plt.title("Mean |itr0 - itr1| per case (sim noise level)")
    plt.savefig(out / "05_itr_noise.png", dpi=150, bbox_inches="tight"); plt.close()

    # 6. example grid: sim SEM / sim depth / real SEM / test SEM
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    for j, p in enumerate(pairs[:6]):
        axes[0, j].imshow(load_image01(p.sem_paths[0]), cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(load_image01(p.depth_path), cmap="viridis", vmin=0, vmax=1)
    for j, p in enumerate(sorted(Path(args.real_sem_dir).glob("*.png"))[:6]):
        axes[2, j].imshow(load_image01(p), cmap="gray", vmin=0, vmax=1)
    for j, p in enumerate(sorted(Path(args.test_sem_dir).glob("*.png"))[:6]):
        axes[3, j].imshow(load_image01(p), cmap="gray", vmin=0, vmax=1)
    for ax in axes.ravel():
        ax.axis("off")
    for i, label in enumerate(["sim SEM", "sim Depth", "real SEM", "test SEM"]):
        axes[i, 0].set_ylabel(label)
    plt.savefig(out / "06_examples.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

Run: `uv run python scripts/eda.py --real-sem-dir <실경로> --real-csv <실경로> --test-sem-dir <실경로>` (경로는 Task 9의 data-notes 기준)
Expected: `report/figures/`에 그림 6장 생성

- [ ] **Step 3: report/eda.md 작성**

그림 6장을 임베드하고 각 그림당 2~4문장으로 실측 소견을 기록한다. 반드시 답할 질문:
1. sim vs real 강도 분포가 얼마나 다른가 (도메인 갭의 크기) — Stage 2 입력 정렬 기법 선택의 근거
2. 밝기-평균깊이 상관계수 r 값 — "밝기만으로 깊이가 풀리는가"에 대한 답과 물리적 해석(이차전자 방출)
3. itr 노이즈 수준 — consistency 학습의 여지
4. train(real) vs test(real) 분포가 같은가 — 검증 전략의 타당성
5. flip 증강이 안전한가 (예시 그림의 구조 대칭성 확인)

- [ ] **Step 4: Commit**

```bash
git add scripts/eda.py report/figures report/eda.md
git commit -m "feat: add EDA report quantifying the sim-to-real domain gap"
```

---

### Task 11: [데이터 필요] 베이스라인 학습 실행 (GPU, 백그라운드)

**Files:**
- Modify: `configs/baseline.yaml` (EDA 결과 반영: 배치 크기/에폭은 실데이터 크기·VRAM 실측으로 조정)
- 산출: `experiments/runs/s1_baseline_r34/best.pt`, `experiments/results.csv` 행 1개

**Interfaces:**
- Consumes: Task 7 `run_training` CLI, Task 9 확정 경로
- Produces: 학습된 베이스라인 체크포인트(Task 12의 입력), sim hold-out RMSE 실측값

- [ ] **Step 1: 1-epoch 파일럿으로 VRAM·시간 실측**

`configs/baseline.yaml`을 복사해 `configs/pilot.yaml` 생성(`epochs: 1`, `run_name: s1_pilot`).
Run: `uv run python scripts/train.py -c configs/pilot.yaml` (포그라운드, `nvidia-smi`로 VRAM 관찰)
Expected: OOM 없이 완료. OOM 시 batch_size를 절반으로 낮춰 재시도하고 baseline.yaml에 반영. 1 epoch 소요 시간을 기록해 본 학습 에폭 수(총 2~4시간 목표)를 정한다.

- [ ] **Step 2: 본 학습 백그라운드 실행**

Run (백그라운드): `uv run python scripts/train.py -c configs/baseline.yaml`
모니터링: TensorBoard 로그의 `val/rmse255` 추이, `nvidia-smi` VRAM. 발산(loss NaN/정체) 시 lr을 1/3로 낮춰 재시작.

- [ ] **Step 3: 결과 확인 및 real 프록시 측정**

학습 완료 후:

```bash
uv run python - <<'EOF'
import torch, yaml
from pathlib import Path
from semdepth.model import UnetTimm
from semdepth.data import RealDataset
from semdepth.proxy import real_proxy_rmse
from semdepth.results import append_result

cfg = yaml.safe_load(open("configs/baseline.yaml"))
model = UnetTimm(cfg["model"]["encoder"], pretrained=False)
ckpt = Path(cfg["out"]["runs_dir"]) / cfg["out"]["run_name"] / "best.pt"
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
ds = RealDataset("<real_sem_dir>", "<real_csv>")  # data-notes의 실경로로 치환
proxy = real_proxy_rmse(model, ds, device="cuda")
print(f"real proxy rmse: {proxy:.4f}")
append_result(Path("experiments/results.csv"),
              {"run_name": cfg["out"]["run_name"] + "_proxy", "real_proxy_rmse": round(proxy, 4)})
EOF
```

Expected: sim val RMSE와 real 프록시 RMSE 두 수치 확보 — 이 차이가 도메인 갭의 첫 정량 측정값.

- [ ] **Step 4: Commit**

```bash
git add configs/ experiments/results.csv
git commit -m "exp: train supervised sim baseline (record sim-val and real-proxy rmse)"
```

---

### Task 12: [데이터 필요] 첫 제출 + 리더보드 캘리브레이션

**Files:**
- Create: `experiments/submissions.csv`
- 산출: `experiments/submission_s1_baseline.zip` (커밋 금지 — gitignore가 *.zip 차단)

**Interfaces:**
- Consumes: Task 8 `scripts/predict.py`, Task 11 체크포인트
- Produces: 첫 리더보드 점수, (sim val, real proxy, LB) 3점 세트의 첫 행 — 이후 모든 실험의 캘리브레이션 기준

- [ ] **Step 1: 테스트 셋 추론 + zip 생성**

```bash
uv run python scripts/predict.py -c configs/baseline.yaml \
  --ckpt experiments/runs/s1_baseline_r34/best.pt \
  --input <test_sem_dir> --out /tmp/pred_s1 --zip experiments/submission_s1_baseline.zip
```

Expected: `wrote 25988 depth maps`, zip 파일 생성. 개수가 25,988이 아니면 중단하고 원인 파악. zip 내부 구조가 Task 9에서 확인한 제출 형식과 일치하는지 `unzip -l`로 확인.

- [ ] **Step 2: 사용자 제출 요청 (수동 개입 지점)**

사용자에게 zip 경로를 안내하고 Dacon 제출을 요청한다. **종료된 대회 제출이 가능한지가 여기서 판명** — 불가하면 스펙 8장 리스크 대응(프록시 기반 성능 주장으로 전환)을 발동하고 이후 태스크의 "제출" 단계를 프록시 평가로 대체한다.

- [ ] **Step 3: 결과 기록**

`experiments/submissions.csv`에 행 추가 (컬럼: `date, run_name, zip, sim_val_rmse255, real_proxy_rmse, lb_public_rmse, note`).

- [ ] **Step 4: Commit**

```bash
git add experiments/submissions.csv
git commit -m "exp: record first leaderboard submission and local-metric calibration"
```

---

## 이 계획 이후 (별도 계획으로 작성)

- **Stage 2 계획**: Task 10(EDA)과 Task 12(캘리브레이션) 결과가 나온 뒤 작성 — 입력 정렬/약지도/self-training 실험군의 우선순위를 실측 도메인 갭 크기에 근거해 결정한다.
- **Stage 3~4 계획**: 앙상블·TTA·최종 제출, 리포트·README·이력서 문구.

## Self-Review 결과

- 스펙 커버리지: 스펙 4장(접근법)의 Stage 1, 5장(검증 전략) 1·2항, 6장(인프라), 7장(로드맵 Stage 0~1)을 Task 1~12가 구현. 스펙 5장 3항(상관 추적)은 Task 12의 submissions.csv가 시작점. Stage 2 이후는 의도적으로 후속 계획으로 분리(스펙 7장과 일치).
- 플레이스홀더: `<실경로>`/`<real_sem_dir>` 표기는 Task 9의 data-notes에서 확정되는 값을 가리키는 명시적 참조로, Task 9 완료 전에는 알 수 없는 값이다(가짜 경로를 적는 것보다 정직).
- 타입 일관성: `rmse_255`, `append_result`, `SimPair`, `list_sim_pairs`, `split_pairs`, `SimDataset`, `RealDataset`, `ImageDirDataset`, `UnetTimm`, `run_training`, `predict_dir`, `make_submission_zip`, `real_proxy_rmse` — 정의 태스크와 사용 태스크 간 시그니처 일치 확인 완료.
