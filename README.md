# Battery Life Lab

[![tests](https://github.com/ChoJiwon00/battery-life-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/ChoJiwon00/battery-life-lab/actions/workflows/tests.yml)

**"충방전 초반 100번만 보고, 이 배터리가 몇 번까지 버틸지 알 수 있을까? 그리고 왜 늙는 걸까?"**

시판 LFP/흑연 셀 124개의 충방전 데이터로 **배터리 수명을 일찍 예측**하고, 방전 곡선과 dQ/dV로
**열화 원인(리튬 손실 vs 활물질 손실)** 을 분석하는 개인 프로젝트입니다.
기준 논문(Severson 외, *Nature Energy* 2019)의 결과를 직접 구현해 재현하고, 그 위에 한 걸음 더 나아가는 것이 목표예요.

> 처음이라면 [배경 지식 노트](docs/background.md)부터 읽어 보세요. 용량·C-rate·SOH 같은 기본 용어부터
> 열화 메커니즘, 논문 요약, 면접 문답까지 정리했어요.

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 배경 지식 정리 ([docs/background.md](docs/background.md)) | ✅ |
| 1 | 데이터 받기·정리: 논문 규칙대로 124개 셀 데이터셋 만들기 | 🔧 코드 완료, 데이터 받기 전 |
| 2 | 그래프로 살펴보기: 용량 곡선, 수명 분포, 충전 조건별 비교, 무릎(knee) | ⏳ |
| 3 | 열화 원인 분석: ΔQ(V), dQ/dV (저율 데이터 포함) | ⏳ |
| 4 | 수명 예측: 논문 재현 → 더 적은 사이클·새 특징으로 개선 | ⏳ |
| 5 | AI 자동화: 새 셀 데이터 → 분석·한국어 리포트 자동 생성 (Claude Code 스킬) | ⏳ |

## 데이터

Toyota Research Institute가 공개한 [Data-driven prediction of battery cycle life before capacity degradation](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)
데이터를 씁니다. 라이선스는 **CC BY 4.0**이에요.

| 배치 (시험 시작일) | 크기 | 논문에서 쓴 셀 |
|---|---|---|
| 2017-05-12 | 2.82 GB | 41 |
| 2017-06-30 | 1.87 GB | 43 |
| 2018-04-12 | 3.01 GB | 40 |

- A123 Systems APR18650M1A (LFP/흑연, 1.1Ah), 30°C, 여러 가지 급속충전 조건 · 방전은 모두 4C
- 시험이 일찍 끝났거나 잡음이 큰 셀은 빼고, 다음 배치에서 이어 시험한 셀 5개는 이어 붙여요.
  규칙은 저자가 공개한 [데이터 정리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)와 같아요.
- 원본 데이터(약 7.7GB)는 용량이 커서 이 저장소에 올리지 않아요. 아래 명령으로 직접 받아요.

## 실행 방법

Python 3.13 기준이에요.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe -m battery_lab info              # 데이터 상태 보기
.venv\Scripts\python.exe -m battery_lab download          # 받을 양 확인
.venv\Scripts\python.exe -m battery_lab download --yes    # 실제로 받기 (끊기면 다시 실행하면 이어받아요)
.venv\Scripts\python.exe -m battery_lab extract           # .mat에서 셀마다 값 추리기
.venv\Scripts\python.exe -m battery_lab assemble          # 124개 셀 데이터셋 만들기
.venv\Scripts\python.exe -m pytest -q                     # 테스트
```

`assemble`이 만드는 것 (`data/processed/`):

| 파일 | 내용 |
|---|---|
| `cells.csv` | 셀마다 한 줄: 배치, 충전 조건, 수명, 사이클 수, 학습/테스트 구분 |
| `summary.csv` | 사이클마다 한 줄: 방전·충전 용량, 내부저항, 온도, 충전 시간 |
| `curves/<셀>.npz` | 사이클별 방전 곡선 Qdlin(전압 구간별 용량), dQ/dV, 온도 |
| `raw/<셀>.npz` | 몇몇 사이클의 원시 데이터(시간·전압·전류·용량·온도) |

## 폴더 구조

```
battery-life-lab/
├─ battery_lab/
│  ├─ sources.py      데이터 출처, 논문의 셀 고르기 규칙, 학습/테스트 나누기
│  ├─ download.py     내려받기 (이어받기, 크기 확인)
│  ├─ extract.py      .mat(HDF5)에서 셀마다 값 추리기
│  ├─ assemble.py     셀 이어 붙이기·빼기 → 데이터셋 파일
│  └─ __main__.py     명령줄 진입점
├─ docs/background.md 0단계 배경 지식
├─ tests/             자동 테스트 (진짜와 같은 구조의 작은 가짜 .mat으로 확인)
└─ data/              내려받은 데이터 (GitHub에 올리지 않음)
```

## 참고

K. A. Severson, P. M. Attia et al., "Data-driven prediction of battery cycle life before capacity degradation",
*Nature Energy* 4, 383–391 (2019). https://www.nature.com/articles/s41560-019-0356-8
