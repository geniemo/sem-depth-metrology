# SEM Top-down 영상 기반 깊이 계측 — sim-to-real 도메인 갭의 진단과 공략

Predicting per-pixel depth maps of semiconductor hole structures from top-down
SEM images ([2022 Samsung AI Challenge: 3D Metrology](https://dacon.io/competitions/official/235954)).
Pixel-level ground truth exists only for simulated images; the real domain
carries one average-depth scalar per site. This repository documents a
hypothesis-driven campaign across that gap — including the negative results
that shaped the method.

> **한 줄 요약** — 표준 도메인 어댑테이션 레시피가 이 문제의 물리(밝기 = 깊이 신호,
> r = −0.977)와 충돌해 전부 실패했고, 그 실패를 근거로 설계한 **약지도 EM-아핀 캘리브레이션**과
> **밝기 보존형 입력 정합**이 점수를 만들었다.

## 핵심 결과

**Public RMSE 4.597 / Private 4.618 — 제출 172팀 중 26위** (등록 790명).
순수 시뮬레이션 지도학습 베이스라인 8.000(118위)에서 가설-검증 사이클 6회로
−3.40을 만들었다. 본대회 최종 리더보드의 20위 컷이 4.796이므로, 그 기준으로는
20위 안쪽에 해당하는 점수다.

| | 기법 | Public LB | 순위 |
|---|---|---|---|
| 베이스라인 | sim 지도학습 (U-Net) | 8.000 | 118 |
| +약지도 | **EM-아핀 평균 제약** | 5.325 | 39 |
| +입력 정합 | **itr-평균(프레임 평균 가설)** | 5.113 | 35 |
| +앙상블 | 3멤버 + flip TTA | **4.597** | **26** |

모든 수치는 [`experiments/results.csv`](experiments/results.csv) ·
[`experiments/submissions.csv`](experiments/submissions.csv)에 실측 기록으로 남아 있다.

## 왜 어려운 문제인가 — 3중 어긋남

| 데이터 | 규모 | 라벨 |
|---|---|---|
| 시뮬레이션 | 구조 21,663종 × 4촬영조건(Case) × 2노이즈(itr) = 173,304장 | **픽셀별** Depth Map |
| 실제 train | 2,059 site × ~29장 = 60,664장 | site당 **평균 깊이 1개** |
| 실제 test | 25,988장 | 없음 — Depth Map PNG 제출, RMSE 채점 |

1. **외형 갭** — sim은 거친 노이즈(픽셀 μ/σ 99.9/57.8), real은 부드럽다(115.6/65.8)
2. **라벨 인코딩 불일치** — sim Depth 픽셀은 "작을수록 깊음", real `average_depth`는
   "클수록 깊음"(물리 스케일); 둘을 잇는 사상이 미지수
3. **라벨 시프트** — sim depth 맵의 영상 평균은 111±7.5에 몰려 있고 real은 105~142의
   4군집; sim은 real 깊이 범위의 가운데만 커버한다

![examples](report/figures/06_examples.png)

*위에서부터 sim SEM / sim Depth GT / real train / real test. 타원형 홀 내부가 어둡고
(깊음) 테두리가 밝은(edge effect) 전형적 top-down SEM. sim은 입자성 노이즈가 강하고
real은 부드럽다.*

## 데이터 사실 (EDA)

전수 검증과 시드 고정 샘플링으로 확정한 사실들이다
(상세: [`report/eda.md`](report/eda.md) · [`docs/data-notes.md`](docs/data-notes.md)).

**밝기가 곧 깊이 신호다.** site 평균 밝기와 평균 깊이의 상관 **r = −0.977** —
깊은 홀일수록 이차전자가 탈출하지 못해 어두워지는 SEM 물리가 데이터에 거의
결정론적으로 새겨져 있다. 이 한 장이 이후 모든 방법 선택의 분기점이 된다.

![brightness-depth](report/figures/04_brightness_vs_depth.png)

**두 도메인의 라벨 분포는 다르게 생겼다.** sim depth 맵의 영상 평균은 폴더
버킷(80~84)과 무관하게 111±7.5에 몰려 있는 반면, real은 명목값(110/120/130/140)에
정합하는 4군집이다 — 지도학습만으로는 깊은 시편에서 계통적 과소 예측이 남는다.

![labels](report/figures/03_label_distributions.png)

**real train ↔ test는 사실상 같은 분포다** (픽셀 μ/σ 115.6/65.8 vs 115.6/65.9).
제출 없이 real train으로 계산하는 프록시 지표가 test를 대변할 수 있다는 근거다.

![intensity](report/figures/01_intensity_hist.png)

그 외: 동일 구조가 4개 Case에 완전 반복(21,663개 파일명 100% 중복)되어 **구조 단위
그룹 스플릿**이 아니면 검증이 누수로 부풀며(테스트로 고정), 같은 구조의 노이즈 실현
쌍(itr0/itr1)의 차이는 평균 9.89±0.74다.

## 방법

### 검증 체계 — 제출 없이 무엇을 알 수 있는가

- **sim hold-out RMSE**: 픽셀 수준이지만 sim 도메인. 구조 단위 그룹 스플릿으로 누수 차단.
- **cal-proxy**: real 6만 장에서 `avg_depth ≈ α + β·mean(pred)`를 최소제곱 적합한 잔차
  RMSE — 인코딩 미지수를 우회해 평균 수준의 real 전이력을 잰다.
- 제출은 가설 검증 단위로만 사용(총 6회), 매 제출마다 (sim, proxy, LB) 3점을 기록해
  로컬 지표의 예측력 자체를 검증했다.

### 핵심 기여 ① — 약지도 EM-아핀 캘리브레이션 (8.00 → 5.33)

real 라벨과 예측의 사상 (α,β)가 미지수이므로, 이를 자유 학습하면 β→0으로 퇴화한다.
대신 **매 epoch (α,β)를 현재 예측에 대한 폐형 최소제곱으로만 재적합**하고(역전파 없음),
모델은 고정된 사상 아래에서 학습한다:

```
loss = L1_sim  +  λ · | α + β·mean(pred_real)·255 − avg_depth | / 255
```

sim L1이 픽셀 형상의 앵커, real 항이 예측 대역을 라벨 분포로 확장한다.
cal-proxy 5.37 → 1.84, corr 0.986.

### 핵심 기여 ② — 밝기 보존형 입력 정합 (5.33 → 5.11)

"real 영상이 부드러운 이유 = 다중 프레임 평균"이라는 가설 아래, sim이 제공하는
노이즈 실현 쌍의 **평균을 학습 입력**으로 사용했다 — 절대 밝기를 전혀 훼손하지 않고
real의 look에 접근한다. 약지도 이전 시점의 전이력이 즉시 개선(cal 5.41→3.83)되어
가설을 로컬에서 먼저 검증했고, 잔여 블러는 방사형 파워 스펙트럼 매칭으로
**σ = 0.7로 계측**해 증강 범위를 데이터로 정했다.

### 마무리 — 앙상블 구성 실험 (5.11 → 4.597)

가설이 다른 3개 모델(itr-평균 r34 / +블러 r34 / convnext_tiny)의 평균 + flip TTA.
6멤버로 확장하면 4.657로 오히려 후퇴한다 — 동일 가중 평균에서 약한 멤버가 강한
멤버를 희석하며, **앙상블은 질이 양을 이긴다**를 수치로 남겼다.

## 결과 전체 표

| 실험 | 가설 | 로컬 (cal-proxy / corr / sim-val) | Public / Private | 판정 |
|---|---|---|---|---|
| S1 | 순수 sim 지도학습 | 5.37 / 0.875 / 2.08 | 8.000 / — | 기준점 (118위) |
| E0 | 전역 입력 정렬 | 6.50 / 0.810 / — | 미제출 | ❌ 밝기 신호 왜곡 |
| E1 | 밝기·대비 랜덤화 | 7.13 / 0.766 / 4.64 | 미제출 | ❌ 신호 제거 |
| E2 | 약지도 EM-아핀 | 1.84 / 0.986 / 2.64 | 5.325 / — | ✅ 39위 |
| E3 | self-training | 1.61 / 0.989 / 2.34 | 5.628 / 5.609 | ❌ 로컬 全개선·LB 악화 |
| E6 | itr-평균 + 약지도 | 1.93 / 0.985 / 3.09 | 5.113 / 5.110 | ✅ 35위 |
| E7·E9 | +블러 지터 / 스펙트럼 σ | 2.00 / 1.92 (sample) | 앙상블 멤버 | 중립 |
| E4c | convnext + 동일 레시피 | **1.58** (sample) / 2.46 | 앙상블 멤버 | 최강 단일 멤버 |
| **E8** | **3멤버 앙상블 + TTA** | 1.96 / 0.984 | **4.597 / 4.618** | ✅ **26위** |
| E10 | 6멤버 앙상블 | — | 4.657 / 4.677 | ❌ 희석 |

## 실패가 가르쳐준 것

> **밝기는 노이즈가 아니라 신호다.** 표준 도메인 어댑테이션 직관(정규화·랜덤화)은
> 외형 변인을 지우라고 말하지만, 이 문제에서 절대 밝기는 깊이 신호의 주 운반자다.
> E0·E1은 그 신호를 지웠기 때문에 실패했고, 이후의 모든 성공은 밝기를 보존하는
> 정합 위에 세워졌다.

**E3(self-training)은 로컬 지표가 전부 개선되면서 LB만 악화된 사례다** (5.33→5.63).
pseudo 라벨이 모델 자신의 예측이라 real 픽셀 형상의 계통 오차가 재학습되는 확증
편향이며, 동시에 "cal-proxy는 평균이 풀린 뒤 픽셀 오차를 보지 못한다"는 지표의
사각지대를 확정해 준 실험이기도 하다.

## 해석과 한계

- 잔여 오차는 real 픽셀 형상에 있다. real 픽셀 GT가 없는 한 이를 로컬에서 재는
  지표는 만들지 못했다 — 실무라면 소량의 TEM/AFM 크로스 캘리브레이션 셋이 이 병목을 푼다.
- 약지도의 (α,β)는 이 장비·레시피 조합에 붙는 상수로, 장비 이관 시 소량 라벨로
  재캘리브레이션하는 절차가 전제된다.
- 상위권(1~2점대)과의 격차 원인은 부록에서 별도로 다뤘다.

## 부록 — 상위권 솔루션 리버스 엔지니어링

공개된 [1위 솔루션](https://github.com/lastdefiance20/2022-Samsung-AI-Challenge-3D-Metrology-1st-place-Solution)(LB 1.22,
CycleGAN + KNN 견본 검색)을 분석·재구현했다. 케이스 분류기는 재현에 성공했고
(site 홀드아웃 98.7%, 테스트 분포가 공개 수치와 0.5% 내 일치), GT 구조를 완전
해독했다(배경 = 케이스별 상수 140/150/160/170, 홀 바닥 = 30, `배경−바닥` = real 버킷
폴더명). 그러나 견본 검색의 핵심 메커니즘은 공개 스펙(WGAN-GP, 300/100 epoch,
동일 정규화)대로 재현해도 성립하지 않았고, 서로 다른 랭커 3종이 동일한 ~6.3에
수렴하는 것으로부터 **현 채점 환경에서 라이브러리 견본 출력의 상한이 ~6.3**이라는
결론을 남겼다. 전체 증거 사슬과 방법론적 기록(프록시 함정 3연속 검증 포함)은
[`report/report.md`](report/report.md)의 부록 장에 있다.

## 재현

```bash
uv sync                                            # PyTorch cu128
uv run pytest -q                                   # 56 passed
uv run python scripts/inspect_data.py              # 데이터 레이아웃 전수 검증 (data/raw/)
uv run python scripts/eda.py                       # report/figures/ 재생성

# 최고 제출(E8) 재현: 멤버 3개 학습 후 앙상블 + TTA
uv run python scripts/train.py    -c configs/e6a_itrmean_base.yaml
uv run python scripts/finetune.py -c configs/e6b_itrmean_weak.yaml
uv run python scripts/train.py    -c configs/e7a_itrmean_blur_base.yaml
uv run python scripts/finetune.py -c configs/e7b_itrmean_blur_weak.yaml
uv run python scripts/train.py    -c configs/e4b_convnext_itrmean_base.yaml
uv run python scripts/finetune.py -c configs/e4c_convnext_weak.yaml
uv run python scripts/predict_ensemble.py --tta \
  --member configs/e6b_itrmean_weak.yaml:experiments/runs/e6b_itrmean_weak/best.pt \
  --member configs/e7b_itrmean_blur_weak.yaml:experiments/runs/e7b_itrmean_blur_weak/best.pt \
  --member configs/e4c_convnext_weak.yaml:experiments/runs/e4c_convnext_weak/best.pt \
  --input data/raw/test/SEM --out /tmp/pred --zip submission.zip
```

데이터는 [대회 페이지](https://dacon.io/competitions/official/235954)에서 받는다
(재배포 금지 약관으로 저장소에는 포함하지 않는다).

## 저장소 구조

```
src/semdepth/     data(누수 방지 그룹 스플릿) · model · train · finetune(EM-아핀 약지도)
                  · infer(앙상블/TTA) · proxy · retrieval/classify/embed(부록 트랙)
scripts/          train / finetune / predict(+ensemble) / eval_real_proxy / inspect_data / eda ...
configs/          실험 1개 = YAML 1개
experiments/      results.csv · submissions.csv — 모든 런과 제출의 원장
report/           기술 리포트(부록 포함) · EDA · 그림
docs/             데이터 실측 노트 · 프로젝트 요약 · [학습 가이드](docs/study-guide.md)(도메인 배경부터 실험 서사까지 자습용)
tests/            56개 — 스플릿 누수 가드, TTA 라운드트립, EM 재적합 폐형해 등 규약 고정
```

## 참고

- 1위 솔루션 (분석·인용): lastdefiance20, [*2022 Samsung AI Challenge (3D Metrology) 1st place Solution*](https://github.com/lastdefiance20/2022-Samsung-AI-Challenge-3D-Metrology-1st-place-Solution)
- 대회: [DACON 235954 — 2022 Samsung AI Challenge (3D Metrology)](https://dacon.io/competitions/official/235954)
