# 데이터 실측 노트 (Task 9: 인테이크 & 가정 검증)

- 검증일: 2026-08-24, `scripts/inspect_data.py` 실행 결과 기준 (전 체크 OK)
- 원본: `data/raw/open.zip` (1.2GB) → `data/raw/` 압축 해제

## 확정된 레이아웃

```
data/raw/
  simulation_data/SEM/Case_{1..4}/{80..84}/<base>_itr{0,1}.png   # 173,304장
  simulation_data/Depth/Case_{1..4}/{80..84}/<base>.png          # 86,652장
  train/SEM/Depth_{110,120,130,140}/site_XXXXX/SEM_XXXXXX.png    # 60,665장(아래 참조)
  train/average_depth.csv                                        # 2,059행, 헤더 "0,1"
  test/SEM/XXXXXX.png                                            # 25,988장 (평면)
```

## 검증된 사실

| 항목 | 실측 | 비고 |
|---|---|---|
| 시뮬레이션 페어링 | 86,652 pair, 전부 SEM 2장(itr0/itr1) | `list_sim_pairs` 예외 없음 |
| 구조 반복 | 고유 구조(base name) 21,663개, **각각 정확히 4개 Case에 존재** | 스플릿은 반드시 group_id(base name) 단위 — case 단위는 누수 |
| 버킷 간 충돌 | 케이스당 고유 stem 21,663 == depth 수 | group_id가 bucket을 버려도 충돌 없음 |
| 대회 페이지 "시뮬레이션 259,956" | SEM 173,304 + Depth 86,652의 합 | 이미지 쌍 수가 아님 |
| train 라벨 | **site 단위** 2,059개 (bucket별 516/515/514/514), 이미지 단위 아님 | csv 키 `depth_<bucket>_site_<id>` ↔ 폴더 `Depth_<bucket>/site_<id>` |
| train 이미지 수 | 라벨 커버 60,664장 + 스트레이 1장 = 60,665 | 스트레이: `Depth_140/site_00000/.ipynb_checkpoints/SEM_000165-checkpoint.png` — 주최측이 실수로 포함한 Jupyter 체크포인트. `RealDataset`은 site 폴더 non-recursive glob이라 자연 제외. 대회 페이지의 "60,664"와 일치 |
| avg_depth 범위 | 105.334 ~ 142.156 | 폴더 명목값(110/120/130/140)과 정합 |
| 이미지 크기 | 전부 48×72 (폭×높이) = 배열 (72,48) | 4개 루트 샘플 300장씩 |
| 이미지 모드 | sim: 전부 L(8-bit). real(train/test): **RGB ~2/3 + L ~1/3 혼재** | RGB는 3채널 완전 동일(샘플 200장 중 불일치 0) → `convert("L")` 무손실. train/test의 RGB 비율 유사 — 동일 취득 파이프라인 |
| 픽셀 통계(샘플) | sim SEM mean 100.2/std 57.9 · real train 115.5/65.8 · real test 115.6/65.9 | **train↔test 실도메인 분포 매우 유사** (real 프록시 검증 전략에 유리), sim↔real 갭 존재 |
| sim Depth 픽셀 | mean 111.4, 범위 0~170 | 버킷 번호(80~84)와 픽셀 평균의 관계는 EDA(Task 10)에서 정량화 |
| 제출 형식 | sample_submission **없음** | Depth Map PNG들의 zip으로 추정(평면), Task 12 첫 제출로 검증 |

## 코드에 반영된 결정

- `configs/baseline.yaml`의 데이터 경로는 실측 레이아웃과 일치 (수정 불필요)
- `split_pairs`: group_id(구조 이름) 단위 그룹 스플릿 (Task 5에서 구현·테스트 완료)
- `RealDataset`: site 단위 라벨 fan-out, csv 위치 기반 파싱 (Task 5)
- `load_image01`: `convert("L")`로 RGB/L 혼재 흡수 (무손실 확인됨)

## 미해결 (후속 태스크로)

- 시뮬레이션 버킷(80~84)의 의미와 real(110~140)과의 라벨 분포 관계 → Task 10 EDA
- 제출 zip 내부 구조(평면 가정) → Task 12 첫 제출로 확정
