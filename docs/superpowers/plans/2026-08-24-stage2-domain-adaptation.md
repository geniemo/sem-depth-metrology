# Stage 2: 도메인 갭 공략 실험 계획

**Goal:** 순수 sim 베이스라인(LB 8.00, 118위/172팀)을 도메인 어댑테이션으로 LB ≤ 3.8(10위 상당)까지 끌어올린다. 스트레치 ≤ 2.6.

**Spec:** `docs/superpowers/specs/2026-08-24-samsung-3d-metrology-design.md` (4장 Stage 2)
**실행 모드:** controller 인라인 (GPU 실험 반복; Stage 0-1의 서브에이전트 경로는 API 과부하로 불안정했음). 라이브러리 코드 변경은 테스트 동반, 실험은 config+commit 단위.

## 증거 기반 (Stage 0-1 실측)

| 지표 | 값 | 함의 |
|---|---|---|
| LB (순수 sim) | 8.00021 | 도메인 갭이 점수 지배 |
| sim hold-out RMSE | 2.085 | 과제 자체는 학습 가능 |
| cal-proxy RMSE / corr | 5.37 / 0.875 | 모델의 real 평균 예측이 밝기 휴리스틱(r=0.977)보다 나쁨 |
| 외형 갭 | sim 99.9/57.8 vs real 115.6/65.8; real이 더 부드러움 | 입력 정렬 유효 예상 |
| 라벨 시프트 | sim depth 평균 ~111±7.5 vs real 105–142 | 약지도 교정 필수 |
| 예측 평균 대역 | 102.8–112.5 (real 필요 범위보다 좁음) | 시프트 증상 확인 |

## 실험 로드맵 (각각 가설→구현→게이트→제출 규칙)

### E0. 추론 시 전역 입력 정렬 (재학습 없음 — 즉시 프로브)
- 가설: real 입력을 전역 아핀 x′=(x−μ_r)/σ_r·σ_s+μ_s 으로 sim 강도 분포에 맞추면, 기존 체크포인트로도 real 평균 예측력이 오른다. (전역 변환이라 영상 간 밝기 순서 보존 — 밝기가 깊이 신호이므로 per-image 정규화는 금지)
- 구현: `scripts/e0_probe.py` — 변환 유/무로 cal-proxy·corr 비교. 코드 변경 없음.
- 게이트: corr > 0.90 또는 cal-proxy < 4.5 → 제출 1회로 LB 반응 측정.

### E1. 학습 시 외형 랜덤화 (sim → real 스타일 증강)
- 가설: sim 입력에 밝기/대비 지터(real 통계 대역을 포괄) + 가우시안 블러(real의 부드러움 모사) + 약한 노이즈를 주면 모델이 외형 갭에 강건해진다. 타깃은 불변(입력만 증강).
- 구현: `SimDataset`에 config 주도 `appearance` 증강 추가(+단위 테스트: 타깃 불변·범위 클립), `configs/e1_appearance.yaml` (r34, 40ep).
- 게이트: cal-proxy < 5.37 그리고 corr > 0.875 → 제출.

### E2. `average_depth` 약지도 파인튜닝 (EM-아핀 평균 제약) — 본명
- 가설: real 6만 장의 site 평균 깊이를 "예측 맵 평균의 아핀 사상"에 대한 제약으로 쓰면 라벨 시프트(좁은 예측 대역)가 교정된다. 인코딩 미지수(α,β)는 매 epoch 폐형 최소제곱으로 재적합(EM 스타일 — 퇴화 방지).
- 구현: `src/semdepth/finetune.py` — sim L1 배치 + real 평균 제약 배치 혼합, loss = L1_sim + λ·|α+β·mean(pred_real)·255 − avg| (α,β는 epoch마다 고정 재적합). E1 체크포인트에서 계속. 단위 테스트: 아핀 재적합 폐형해, 혼합 배치 스케줄.
- 게이트: cal-proxy가 유의미하게(>30%) 하락 → 제출. LB가 프록시와 역행하면 인코딩 가설 재검토(계획 중단·분석).

### E3. Self-training (pseudo-label)
- 가설: E2 모델의 real 예측(평균 교정 완료)을 pseudo-label로 sim과 혼합 재학습하면 픽셀 수준 real 적응이 추가된다.
- 구현: E2 모델로 real train 예측 저장 → `SimDataset` 호환 pseudo 쌍 폴더 생성 → 혼합 비율 config로 재학습.
- 게이트: sim-val 유지(±0.2) + cal-proxy 개선 → 제출. sim-val 붕괴 시 pseudo 비중 축소.

### E4. 마무리 (용량·앙상블·TTA)
- convnext_tiny/effnet 인코더 1종 추가 학습(주의: 비-resnet 채택 시 UnetTimm fallback-interpolate 브랜치 테스트 추가 — Stage 0-1 이월 조건), 시드 2개, flip TTA, 스냅샷 평균. 최종 2개 제출 선택.

## 운영 규칙

- 로컬 게이트 통과 실험만 제출 (하루 10회 제한, 오늘 1회 사용).
- 모든 실험: `experiments/results.csv` 행 + 제출 시 `experiments/submissions.csv` 행. (sim-val, cal-proxy, corr, LB) 4중 기록으로 로컬↔LB 상관을 계속 검증.
- config에 `device`를 넣을 경우 `cuda`만 사용(`cuda:0` 금지 — autocast 제약, Stage 0-1 이월 조건).
- 각 실험 완료 시 커밋: `exp(e<N>): <결과 요약>`.
- 중단 규칙: 연속 2개 실험에서 로컬 게이트 미달이면 계획을 멈추고 원인 분석(리포트 소재로 기록).
