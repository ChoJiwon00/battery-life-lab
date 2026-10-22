# 관련 연구: Severson 논문 말고 무엇이 더 있을까?

Claude가 참고 문헌을 찾아 정리하고, 배터리 시뮬레이션을 이 PC에서 직접 돌려 본 결과를 함께 적은 문서예요 (만든 날 2026-10-22).
숫자는 모두 각 항목의 링크(초록·보도자료·데이터 페이지·공식 문서)나 이 PC의 실행 기록에서 확인한 것만 옮겼어요.

이 프로젝트는 Severson 외(2019, *Nature Energy*) "Data-driven prediction of battery cycle life before capacity degradation"을 재현했어요.
LFP/흑연 셀 124개의 초반 ΔQ₁₀₀₋₁₀(V) 분산 하나로 수명을 맞히는 모델이고, 테스트 / 2차 테스트 RMSE(평균 제곱근 오차)는 논문과 같은 **138 / 196사이클**이 나왔어요 ([데이터 크기 실험](data_size.md)).
이 문서는 다음 걸음을 위해 세 가지 질문에 답해요.

1. 이 논문 말고 **같은 줄기의 연구와 공개 데이터**는 무엇이 있을까?
2. **시뮬레이션**은 우리가 돌릴 수 있을까? → 실제로 돌려 봤어요 ([5장](#5-시뮬레이션-우리가-돌릴-수-있나))
3. **DFT**는 셀 개발·공정 개발 지원자의 영역일까?

> **읽는 법:** "초록 기준", "보도자료 기준"이라고 적은 숫자는 논문 본문이 아니라 그 페이지에서 확인한 숫자예요. 보도자료는 그 자리에 링크를 함께 달았어요.
> 데이터와 오차 정의가 논문마다 달라서, 다른 논문의 오차 %를 우리 숫자와 바로 비교하면 안 돼요.

### 자주 나오는 용어

| 용어 | 뜻 |
|---|---|
| LLI | 리튬 재고 손실. 충방전에 쓸 수 있는 리튬이 줄어드는 것 |
| LAM | 활물질 손실. 리튬을 담는 전극 재료가 제 역할을 못 하게 되는 것 |
| SEI | 음극 표면에 생기는 얇은 막 |
| dQ/dV · dV/dQ | 충방전 곡선을 미분한 그래프. 봉우리 모양 변화로 열화 원인을 추정해요 |
| 화성(formation) | 셀을 만든 뒤 처음 충방전하는 공정 |
| 리튬 석출(plating) | 음극 표면에 금속 리튬이 쌓이는 것. 큰 전류로 충전할 때 생기는 대표 열화 |
| 무릎(knee) | 용량이 천천히 줄다가 어느 순간 급격히 꺾이는 지점 |
| C-rate | 충방전 속도. 1C는 1시간에 완전 충전(방전)하는 속도, 숫자가 클수록 빨라요 |
| 파라미터 세트 | 시뮬레이션에 넣는 셀의 재료·크기·반응 속도 같은 숫자 묶음 |

## 1. 한눈에 보기

- **수명 예측:** Severson처럼 초반 데이터로 수명 맞히기
- **데이터·도구:** 공개 데이터와 그걸 한꺼번에 불러오는 도구
- **셀 개발:** 열화 원인 진단, 급속충전, 수명 시험 조건
- **공정 개발:** 화성 공정, 출하 검사, 셀 간 편차
- **시뮬레이션:** 배터리 물리 모델과 프로그램
- **DFT:** 원자·전자 수준의 소재 계산

| 분류 | 제목(링크) | 연도 | 우리 프로젝트와의 연결 |
|---|---|---|---|
| 수명 예측 | [Closed-loop optimization of fast-charging protocols for batteries with machine learning](https://doi.org/10.1038/s41586-020-1994-5) (Attia 외, *Nature*) | 2020 | 같은 그룹의 후속. 초기 예측으로 급속충전 방식 탐색을 16일로 단축 → 아이디어 4·2 |
| 수명 예측 | [Statistical Learning for Accurate and Interpretable Battery Lifetime Prediction](https://doi.org/10.1149/1945-7111/ac2704) (Attia·Severson·Witmer, *JES*) | 2021 | 같은 데이터 재분석. "데이터가 적으면 단순 선형 모델로 충분" → 아이디어 9, 데이터 크기 실험 |
| 수명 예측 | [Feature engineering for machine learning enabled early prediction of battery lifetime](https://doi.org/10.1016/j.jpowsour.2022.231127) (Paulson 외, *J. Power Sources*) | 2022 | LFP 하나에서 여러 화학계로 넓히기 → 아이디어 7·2 |
| 수명 예측 | [Predicting battery lifetime under varying usage conditions from early aging data](https://doi.org/10.1016/j.xcrp.2024.101891) (Li 외, *Cell Rep. Phys. Sci.*) | 2024 | NMC, 다양한 사용 조건, 공개 데이터(CC BY 4.0) → 아이디어 7·6·3 |
| 수명 예측 | [Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning](https://doi.org/10.1038/s42256-024-00972-x) (Zhang 외, *Nat. Mach. Intell.*) | 2025 | 소재·조건이 제각각인 셀 401개를 한 모델로 → 아이디어 7 |
| 수명 예측 | [Interpretable Battery Cycle Life Range Prediction Using Early Cell Degradation Data](https://doi.org/10.1109/TTE.2022.3226683) (Zhang 외, *IEEE TTE*) | 2023 | 수명을 범위(예측 구간)로 → 아이디어 3·9 |
| 수명 예측 | [Identification and machine learning prediction of knee-point and knee-onset in capacity degradation curves of lithium-ion cells](https://doi.org/10.1016/j.egyai.2020.100006) (Fermín-Cueto 외, *Energy and AI*) | 2020 | 무릎 예측 + 3\~5사이클 분류 + 예측 구간 → 아이디어 8·2·3 |
| 데이터·도구 | [BatteryML: An Open-source platform for Machine Learning on Battery Degradation](https://arxiv.org/abs/2310.14714) (Zhang 외, ICLR) | 2024 | 우리 RMSE 138 / 196을 남이 만든 구현과 맞대 보기 → 아이디어 7·3 |
| 데이터·도구 | [BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction](https://arxiv.org/abs/2502.18807) (Tan 외, KDD) | 2025 | 공개 데이터 16개를 한 형식으로 → 아이디어 7 |
| 데이터·도구 | [Dataset for: Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning](https://doi.org/10.17632/nsc7hnsg4s.2) (HUST 데이터셋, Mendeley Data; 논문 Ma 외, *EES*) | 2022 | 우리와 같은 LFP/흑연 1.1 Ah 셀 77개 → 아이디어 7의 첫 시험장 |
| 데이터·도구 | [Degradation of Commercial Lithium-Ion Cells as a Function of Chemistry and Cycling Conditions](https://doi.org/10.1149/1945-7111/abae37) (Preger 외, *JES*; Battery Archive) | 2020 | LFP·NCA·NMC를 온도·방전 깊이별로 → 아이디어 7 |
| 데이터·도구 | [Graphite//LFP synthetic V vs. Q dataset (>700,000 unique curves)](https://data.mendeley.com/datasets/bs2j56pn7y/3) (Dubarry, Mendeley Data) | 2021 | 정답(LLI·LAM 값)이 붙은 LFP 가상 곡선 → 아이디어 6·9 |
| 셀 개발 | [Review—"Knees" in Lithium-Ion Battery Aging Trajectories](https://doi.org/10.1149/1945-7111/ac6d13) (Attia 외, *JES*) | 2022 | 용량 급락(무릎)의 원인 정리 → 아이디어 8·6 |
| 셀 개발 | [Lithium-ion battery fast charging: A review](https://doi.org/10.1016/j.etran.2019.100011) (Tomaszewska 외, *eTransportation*) | 2019 | 급속충전 열화(리튬 석출)의 배경 → 아이디어 4 |
| 셀 개발 | [Detecting the Onset of Lithium Plating and Monitoring Fast Charging Performance with Voltage Relaxation](https://doi.org/10.1021/acsenergylett.0c00831) (Konz 외, *ACS Energy Lett.*) | 2020 | 전압 이완으로 리튬 석출 감지 → 아이디어 4 |
| 셀 개발 | [Best practices for incremental capacity analysis](https://doi.org/10.3389/fenrg.2022.1023555) (Dubarry·Anseán, *Front. Energy Res.*) | 2022 | dQ/dV 실무 지침, 4C 데이터의 한계 → 아이디어 6 |
| 셀 개발 | [Degradation diagnostics for lithium ion cells](https://doi.org/10.1016/j.jpowsour.2016.12.011) (Birkl 외, *J. Power Sources*) | 2017 | LLI·LAM 개념의 실험 근거 → 아이디어 6 |
| 셀 개발 | [Dynamic cycling enhances battery lifetime](https://doi.org/10.1038/s41560-024-01675-8) (Geslin 외, *Nature Energy*) | 2024 | 시험 방식이 수명 숫자를 바꿈 → 아이디어 4, 우리 결과의 한계 |
| 공정 개발 | [Predicting the impact of formation protocols on battery lifetime immediately after manufacturing](https://doi.org/10.1016/j.joule.2021.09.015) (Weng 외, *Joule*) | 2021 | 출하 직전 몇 초 측정으로 수명 가늠 → 아이디어 2·6 |
| 공정 개발 | [Data-driven analysis of battery formation reveals the role of electrode utilization in extending cycle life](https://doi.org/10.1016/j.joule.2024.07.024) (Cui 외, *Joule*) | 2024 | 빠른 첫 충전이 수명을 늘린 화성 연구 → 아이디어 6 |
| 공정 개발 | [Differential voltage analysis for battery manufacturing process control](https://doi.org/10.3389/fenrg.2023.1087269) (Weng 외, *Front. Energy Res.*) | 2023 | dV/dQ로 출하 검사, 코드·데이터 공개 → 아이디어 6 |
| 공정 개발 | [Intrinsic Variability in the Degradation of a Batch of Commercial 18650 Lithium-Ion Cells](https://doi.org/10.3390/en11051031) (Devie 외, *Energies*) | 2018 | 같은 배치 셀도 늙으면서 벌어짐 → 아이디어 5·9 |
| 공정 개발 | [Production caused variation in capacity aging trend and correlation to initial cell performance](https://doi.org/10.1016/j.jpowsour.2013.08.108) (Baumhöfer 외, *J. Power Sources*) | 2014 | 생산 편차 → 수명 편차를 다룬 고전 → 아이디어 5 |
| 공정 개발 | [Challenges and opportunities for high-quality battery production at scale](https://doi.org/10.1038/s41467-025-55861-7) (Attia 외, *Nat. Commun.*) | 2025 | 대량 생산 품질의 큰 그림 → 아이디어 2 |
| 시뮬레이션 | [Python Battery Mathematical Modelling (PyBaMM)](https://doi.org/10.5334/jors.309) (Sulzer 외, *JORS*) | 2021 | 이번에 이 PC에서 설치하고 돌린 도구 (인용용) |
| 시뮬레이션 | [PyBaMM 공식 문서: 결합 열화 예제 · 파라미터 세트 · 설치](https://docs.pybamm.org/en/latest/source/examples/notebooks/models/coupled-degradation.html) (PyBaMM Team) | 2026 확인 | 실습 (b)의 출처. LFP 세트는 하나뿐 |
| 시뮬레이션 | [Lithium-ion battery degradation: how to model it](https://doi.org/10.1039/D2CP00417H) (O'Kane 외, *PCCP*) | 2022 | 실습 (b)에서 쓴 OKane2022 세트의 출처 → 아이디어 4·8 |
| 시뮬레이션 | [Lithium ion battery degradation: what you need to know](https://doi.org/10.1039/d1cp00359c) (Edge 외, *PCCP*) | 2021 | 열화 메커니즘 입문 리뷰 (면접 공부) |
| 시뮬레이션 | [A continuum of physics-based lithium-ion battery models reviewed](https://doi.org/10.1088/2516-1083/ac7d31) (Brosa Planella 외, *Prog. Energy*) | 2022 | SPM·SPMe·DFN의 관계, 실습 (a) 해석 |
| 시뮬레이션 | [A Simplified Electrochemical and Thermal Aging Model of LiFePO4-Graphite Li-ion Batteries: Power and Capacity Fade Simulations](https://doi.org/10.1149/2.053304jes) (Prada 외, *JES*) | 2013 | PyBaMM 유일한 LFP 세트의 출처, 실습 (a)·(c)에서 사용 (Severson 셀과는 다름) |
| 시뮬레이션 | [Synthesize battery degradation modes via a diagnostic and prognostic model](https://doi.org/10.1016/j.jpowsour.2012.07.016) (Dubarry 외, *J. Power Sources*) | 2012 | 가벼운 LLI·LAM 합성 ('alawa 툴박스의 바탕) → 아이디어 6 |
| 시뮬레이션 | [Analysis of Synthetic Voltage vs. Capacity Datasets for Big Data Li-ion Diagnosis and Prognosis](https://doi.org/10.3390/en14092371) (Dubarry·Beck, *Energies*) | 2021 | 시뮬레이션을 돌리지 않고 결과 데이터를 내려받아 쓰기 → 아이디어 6 |
| 시뮬레이션 | [Perspective—Combining Physics and Machine Learning to Predict Battery Lifetime](https://doi.org/10.1149/1945-7111/abec55) (Aykol 외, *JES*) | 2021 | 물리 모델 + 머신러닝을 섞는 설계도 |
| 시뮬레이션 | [PyBOP: A Python package for battery model optimisation and parameterisation](https://doi.org/10.21105/joss.07874) (Planden 외, *JOSS*) | 2025 | 모델을 우리 데이터에 맞추는 도구 (나중 단계) |
| 시뮬레이션 | [BattMo -- Battery Modelling Toolbox](https://arxiv.org/abs/2512.17933) (Raynaud 외, arXiv) | 2025 | PyBaMM의 대안 (MATLAB·Julia) |
| 시뮬레이션 | [COMSOL Battery Design Module](https://www.comsol.com/battery-design-module) (COMSOL, 상용) | 2026 확인 | 회사 셀 설계·CAE 도구, 유료 |
| DFT | [Computational understanding of Li-ion batteries](https://doi.org/10.1038/npjcompumats.2016.2) (Urban·Seo·Ceder, *npj Comput. Mater.*) | 2016 | DFT는 '셀'이 아니라 '소재' 계산이라는 근거 |
| DFT | [QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials](https://doi.org/10.1088/0953-8984/21/39/395502) (Giannozzi 외, *J. Phys.: Condens. Matter*) | 2009 | 무료 DFT 프로그램 (이름만 알아 두기) |
| DFT | [VASP - Vienna Ab initio Simulation Package](https://www.vasp.at/) (VASP Software GmbH, 상용) | 2026 확인 | 유료 DFT 프로그램 (이름만 알아 두기) |
| DFT | [Commentary: The Materials Project: A materials genome approach to accelerating materials innovation](https://doi.org/10.1063/1.4812323) (Jain 외, *APL Materials*) | 2013 | DFT 결과를 '찾아보기'만 하는 길 |

아이디어 번호는 [응용 아이디어](ideas.md)의 번호예요.

## 2. 같은 줄기의 연구: 수명 조기 예측

Severson 이후 연구는 크게 세 방향으로 이어졌어요.

| 방향 | 질문 | 대표 연구 |
|---|---|---|
| 시험 시간 줄이기 | 예측을 실제 개발 일정 단축에 쓸 수 있나? | Attia 2020 |
| 더 단순하게 · 더 넓게 | 단순 모델로 충분한가? 다른 셀에도 통하나? | Attia 2021, Paulson 2022, Li 2024, Zhang 2025 |
| 숫자 하나 대신 | 범위·무릎·분류로 답할 수 있나? | Zhang 2023, Fermín-Cueto 2020 |

### 2-1. 같은 연구 그룹의 후속

**[Closed-loop optimization of fast-charging protocols for batteries with machine learning](https://doi.org/10.1038/s41586-020-1994-5)** — Attia, Grover, Jin, Severson 외 (교신: Chueh), *Nature* 578, 397–402 (2020)

- **무엇:** Severson 논문을 낸 그룹(MIT·스탠퍼드·TRI)의 후속이에요. 10분 급속충전 방식 후보 224개 중 오래 가는 방식을 찾는데, 셀을 끝까지 돌리지 않고 초기 사이클로 수명을 예측한 뒤, 다음에 시험할 충전 방식을 알고리즘이 골라 주는 '닫힌 고리(closed-loop)' 실험을 했어요.
- **핵심 결과 (초록 기준):** 6단계 10분 급속충전 방식 224개를 탐색해서, 초기 예측 없이 전부 시험하면 500일 넘게 걸릴 일을 16일 만에 해냈어요.
- **우리와 연결:** [아이디어 4 급속충전 조건과 수명](ideas.md#4-급속충전-조건과-수명)의 배경이에요. 아이디어 2(5사이클 선별)도 결국 시험 기간을 줄이는 게 목적이라 같은 이야기로 묶여요. 면접에서는 "수명 시험을 몇 달에서 며칠로 줄인다"로 쓸 수 있어요.
- **공개 코드:** [chueh-ermon/battery-fast-charging-optimization](https://github.com/chueh-ermon/battery-fast-charging-optimization) (코드·그림 스크립트, 저장소 페이지에 라이선스 표기 없음)

**[Statistical Learning for Accurate and Interpretable Battery Lifetime Prediction](https://doi.org/10.1149/1945-7111/ac2704)** — Attia, Severson, Witmer, *JES* 168(9), 090547 (2021, [arXiv:2101.01885](https://arxiv.org/abs/2101.01885))

- **무엇:** Severson 데이터를 통계학적으로 다시 분석했어요. 사이클마다의 용량 곡선을 행렬 하나로 묶은 'capacity matrix'를 제안하고, 여기서 만든 여러 특징으로 단순한 선형 모델을 만들었어요.
- **핵심 결과 (초록 기준):** 공학적으로 설계한 특징을 쓴 단순 선형 모델이, 특히 학습 데이터가 적을 때 고급 머신러닝·딥러닝과 비슷한 성능을 내면서 해석도 쉬워요. 여러 단변량·다변량 모델이 같은 데이터에서 기존 최고 결과와 맞먹었어요.
- **우리와 연결:** 우리 [데이터 크기 실험](data_size.md)이 논문 variance 모델 숫자(138 / 196사이클)를 가져온 출처예요. 그 실험의 결론("복잡한 모델은 데이터가 많을 때만 확실히 이겨요")과 같은 방향이라, 면접에서 "왜 딥러닝 대신 선형 모델을 썼나요?"에 답할 근거가 돼요. [아이디어 9 특징 해석](ideas.md#9-특징이-왜-통하나-해석)과도 이어져요.
- **데이터:** Severson 데이터와 같아요 (data.matr.io). arXiv 페이지에 코드 링크는 없었어요.

### 2-2. 다른 셀 · 다른 조건으로 넓히기

**[Feature engineering for machine learning enabled early prediction of battery lifetime](https://doi.org/10.1016/j.jpowsour.2022.231127)** — Paulson, Kubal, Ward, Saxena, Lu, Babinec, *J. Power Sources* 527, 231127 (2022)

- **무엇:** 미국 아르곤 국립연구소가 '초기 데이터로 수명 예측'을 LFP 한 종류에서 여러 양극 화학계로 넓혔어요. 화학계가 달라도 잘 맞는 특징(feature)을 찾는 데 집중했어요.
- **핵심 결과 ([Argonne 보도자료](https://www.newswise.com/doescience/https-www-anl-govarticleresearchers-now-able-to-predict-battery-lifetimes-with-machine-learning) 기준, 2022-05-05, Newswise 게재본):** 6가지 화학계, 셀 300개의 데이터를 썼고, 경우에 따라 사이클 1개 분량의 데이터만으로 수명을 추정했어요. 정확도 숫자는 보도자료에 없었고, 논문 초록 페이지는 열리지 않아 확인하지 못했어요.
- **우리와 연결:** [아이디어 7 일반화](ideas.md#7-다른-셀에도-통할까-일반화)의 대표 선행 연구예요. 우리 결과를 "LFP에서만 확인했다"는 한계로 정리할 때 함께 인용하기 좋아요. 공개 데이터 여부는 확인하지 못했어요.

**[Predicting battery lifetime under varying usage conditions from early aging data](https://doi.org/10.1016/j.xcrp.2024.101891)** — Li, Zhou, Thelen, Howey, Hu, *Cell Reports Physical Science* 5(4), 101891 (2024, [arXiv:2307.08382](https://arxiv.org/abs/2307.08382))

- **무엇:** NMC/흑연 파우치 셀을 여러 충전 속도·방전 속도·방전 깊이(DOD, 한 번에 얼마나 깊게 쓰는지) 조건에서 노화시키고, 중간중간 천천히(저율로) 잰 용량-전압 곡선에서 특징을 뽑아 수명을 예측했어요. 학습 때 보지 못한 조건(분포 밖)의 셀은 계층적 베이지안 모델로 예측했어요.
- **핵심 결과 (arXiv 초록 기준):** 셀 225개. 대부분의 셀에서 처음 15% 이하의 데이터만으로, 학습 분포 안의 셀은 MAPE(평균 절대 백분율 오차) 15.1%, 분포 밖 셀은 21.8%였어요.
- **우리와 연결:** 아이디어 세 개에 걸려요. ① 일반화(NMC, 다양한 조건) ② LLI·LAM 분석(저율 기준 성능 시험, 즉 RPT 데이터가 들어 있음) ③ 예측 구간(베이지안 모델은 불확실성을 함께 내요).
- **데이터:** ISU-ILCC Battery Aging Dataset v2, CC BY 4.0 → [3. 공개 데이터셋](#3-공개-데이터셋)

**[Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning](https://doi.org/10.1038/s42256-024-00972-x)** — Zhang, Li, Zheng, Lu, Gui, Xu, Bian, *Nature Machine Intelligence* 7(2), 270–277 (2025, [arXiv:2310.05052](https://arxiv.org/abs/2310.05052))

- **무엇:** 마이크로소프트 연구팀이 셀 하나만 보지 않고 '셀과 셀 사이의 차이'를 함께 학습하는 딥러닝(inter-cell learning)으로, 전극 소재와 시험 조건이 제각각인 셀들의 수명을 한 모델로 예측했어요. (arXiv 제목은 "Accurate battery lifetime prediction across diverse aging conditions with deep learning")
- **핵심 결과 (arXiv 초록 기준):** 전극 소재 5종, 시험 조건 168개, 셀 401개에서 처음 100사이클만으로 약 10% 예측 오차를 냈고, 여러 경우에 셀 하나씩 모델링하는 방식보다 오차를 거의 절반으로 줄였어요.
- **우리와 연결:** 일반화의 최신 대표 결과예요. "다음 단계는 이런 방향"이라고 말할 때 인용하면 좋아요. 같은 팀이 아래 BatteryML을 만들었어요. arXiv 페이지에 코드 링크는 없었어요.

### 2-3. 숫자 하나 대신 범위 · 무릎 · 분류로

**[Interpretable Battery Cycle Life Range Prediction Using Early Cell Degradation Data](https://doi.org/10.1109/TTE.2022.3226683)** — Zhang, Su, Altaf, Wik, Gros, *IEEE Trans. Transportation Electrification* 9(2), 2669–2682 (2023, [arXiv:2204.12420](https://arxiv.org/abs/2204.12420))

- **무엇:** 수명을 "800\~1000사이클 사이"처럼 범위(예측 구간)로 내놓는 방법이에요. 분위수 회귀 숲(Quantile Regression Forest, 여러 결정 트리로 상·하한을 함께 추정하는 모델)을 썼어요.
- **핵심 결과 (arXiv 초록 기준):** 분포를 가정하지 않고 수명 범위와 점 예측을 함께 내고, 'alpha-logistic-weighted' 기준으로 하이퍼파라미터(모델 설정값)를 골라 범위가 실제로 맞는 비율을 보정했어요. 초록에 데이터셋 이름과 구체 수치는 없었어요.
- **우리와 연결:** [아이디어 3 예측 구간](ideas.md#3-예측에-오차-범위-붙이기-불확실성)의 직접 참고 자료예요. 순열 중요도·부분 의존도 그림으로 모델을 해석해서 아이디어 9에도 쓸 수 있어요. 공정 관점에서는 "범위가 넓은 셀은 재시험" 같은 판정 규칙으로 이어져요. 코드 공개 여부는 확인하지 못했어요.

**[Identification and machine learning prediction of knee-point and knee-onset in capacity degradation curves of lithium-ion cells](https://doi.org/10.1016/j.egyai.2020.100006)** — Fermín-Cueto, McTurk, Allerhand, Medina-Lopez, Anjos, Sylvester, dos Reis, *Energy and AI* 1, 100006 (2020)

- **무엇:** 용량 곡선에서 '무릎 지점(knee-point)'과 그보다 앞서 비선형 열화가 시작되는 '무릎 시작점(knee-onset)'을 찾는 방법을 제안하고, 초기 데이터만으로 머신러닝으로 예측해요. 예측에 신뢰 구간도 붙여요.
- **핵심 결과:** 무릎 지점은 평균 용량 약 95%, 무릎 시작점은 약 97.1%에서 나타났고 둘 사이는 약 108사이클이었어요. 처음 50사이클로 무릎 지점·시작점을 오차 9.4%로 예측했고, **처음 3\~5사이클만으로 셀을 단수명·중간·장수명으로 88\~90% 정확도로 분류**했어요. 셀 제조사가 생산 기법을 더 빨리 검증하는 데 쓸 수 있다고 제안해요.
- **우리와 연결:** 우리 계획 중 세 가지(아이디어 8 무릎, 2 5사이클 선별, 3 예측 구간)를 한 논문이 다뤄요. 1C\~8C 고속 사이클 데이터를 써서 Severson 데이터와 조건이 비슷해요 (데이터 출처를 명시했는지까지는 확인하지 못함). 아이디어 2를 할 때 우리 결과를 이 숫자와 나란히 놓으면 설득력이 커져요.
- **공개:** 오픈 액세스(CC BY-NC-ND). 코드 공개 여부는 확인한 페이지에 없었어요.

## 3. 공개 데이터셋

> 데이터를 내려받기 전에는 [응용 아이디어](ideas.md)에 적은 대로 **크기를 먼저 확인하고 내려받을지 정해요.** 아래 데이터의 파일 크기는 아직 확인하지 않았어요.

### 셀 시험 데이터

| 데이터 | 셀 | 규모 | 라이선스 | 우리에게 |
|---|---|---|---|---|
| Severson 외 2019 (MATR), data.matr.io | LFP/흑연 | 셀 124개 (논문 사용) | [데이터 카드](dataset_card.md) 참고 | 지금 쓰는 데이터 |
| [HUST](https://data.mendeley.com/datasets/nsc7hnsg4s/2) (중국 화중과기대) | LFP/흑연, 정격 1.1 Ah · 3.3 V | 셀 77개. 충전 방식은 모두 같고 다단계 방전 방식을 바꿈, 30°C 일정 | CC BY 4.0 | **일반화의 첫 시험장.** 화학계·용량이 같고 '충전' 대신 '방전'이 달라요 |
| [ISU-ILCC Battery Aging Dataset v2](https://doi.org/10.25380/iastate.22582234) (Li 외 2024) | NMC/흑연 파우치 | 시험 251개 중 238개 공개, 시험 조건 63개, JSON·CSV | CC BY 4.0 | NMC로 넓히기 + 저율 RPT로 LLI·LAM 분석. 논문의 225개와 숫자가 다른 건 공개 버전 차이로 보여요 |
| [Sandia](https://www.batteryarchive.org/study_summaries.html) (Preger 외 2020, Battery Archive) | 시판 18650 LFP · NCA · NMC | 방전 속도·방전 깊이·온도를 바꿔 몇 년 동안 시험 (셀 수는 확인 안 함) | 사이트 차원의 명시적 오픈 라이선스 없음. 원 논문과 BatteryArchive.org 인용 요구, 일부 기관은 별도 재사용 규칙 링크 | 화학계·온도 바꿔 보기 |
| [MICH (formation)](https://www.batteryarchive.org/study_summaries.html) (Weng 외 2021, Battery Archive) | NMC/흑연 | 셀 40개 | 위 Battery Archive 조건과 같음 | 화성 공정 → 수명 (공정 개발) |
| Weng 외 2023 논문의 데이터: [Mohtat 외 2021](https://doi.org/10.7302/7tw1-kc35), [Weng 외 2021](https://doi.org/10.7302/pa3f-4w30) | 확인 안 함 | 확인 안 함 | 확인 안 함 | dV/dQ 분석 연습 (논문 데이터 공개 문구 기준) |

- [**Battery Archive**](https://www.batteryarchive.org/study_summaries.html)는 공개 데이터를 한곳에서 찾는 '지도'예요. Sandia 말고도 CALCE, HNEI, Oxford, 미시간, UL-Purdue, ISU-ILCC 데이터가 모여 있어요.
- Sandia 논문의 핵심(초록 기준): 제조사 사양 안에서 돌려도 시험 조건이 열화에 큰 영향을 주어, 같은 화학계 안에서도 용량 80% 도달 시점이 수천 시간·수천 사이클 차이 났어요. 셀 개발의 수명 평가 조건 설계와 바로 이어지는 결과예요.

### 시뮬레이션으로 만든 데이터 (정답이 붙은 연습용)

| 데이터 | 내용 | 규모 | 라이선스 | 우리에게 |
|---|---|---|---|---|
| [Graphite//LFP synthetic V vs. Q dataset](https://data.mendeley.com/datasets/bs2j56pn7y/3) (Dubarry, V3) | 흑연/LFP 셀의 가상 전압-용량 곡선. 열화 모드(LLI, LAM 등)를 조금씩 바꿔 가며 계산 | 고유한 곡선 70만 개 이상, C/25 충전 기준, 열화 경로 5,000개 이상 (경로마다 열화 모드당 최대 0.85%씩 증가) | CC BY 4.0 | Severson 셀과 같은 계열. 곡선마다 LLI·LAM 정답이 있어 dQ/dV 분석 연습에 좋아요 |
| Dubarry·Anseán 2022의 열화 지도(degradation map), Mendeley Data [10.17632/4k8f4t352y.1](https://doi.org/10.17632/4k8f4t352y.1) | dQ/dV 봉우리 변화와 열화 모드의 대응 | 확인 안 함 | 확인 안 함 | 봉우리 해석의 참고표 |

관련 논문 [Dubarry·Beck(2021)](https://doi.org/10.3390/en14092371)은 LFP·NCA·NMC811 세 화학계의 합성 데이터셋을 소개하고, 세 개의 학습 가능한 파라미터 조합으로 진단과 조기 예측 성능이 최신 방법과 비슷하면서 물리적으로 해석 가능하다고 적었어요 (Crossref 초록 기준). 합성 데이터 페이지는 관련 논문으로 Dubarry & Beck, *Journal of Power Sources* 479, 228806 (2020)도 함께 들어요.

### 여러 데이터를 한 번에 불러오는 도구

| 도구 | 지원 데이터 | 라이선스 | 우리에게 |
|---|---|---|---|
| [BatteryML](https://github.com/microsoft/BatteryML) (마이크로소프트, [arXiv](https://arxiv.org/abs/2310.14714)) | CALCE, MATR(= Severson), HUST, HNEI, RWTH, SNL, UL_PUR | MIT | 명령 몇 줄로 다운로드 → 전처리 → 학습·평가. **Severson용 variance 모델 설정 파일**(`configs/baselines/sklearn/variance_model/matr_1.yaml`)이 있어 우리 RMSE 138 / 196과 맞대 볼 수 있어요. Variance·Discharge·Full 모델, 가우시안 프로세스, XGBoost, 랜덤포레스트, MLP·CNN·LSTM·Transformer 포함 |
| [BatteryLife](https://github.com/Ruifeng-Tan/BatteryLife) ([arXiv](https://arxiv.org/abs/2502.18807)) | 공개 데이터 16개 통합 (MATR, HUST, ISU_ILCC, MICH, SNL 등). 기존 최대 데이터셋의 약 2.5배, 셀 형태 8종·화학계 59종·온도 9가지·충방전 프로토콜 421개, 리튬이온 외에 아연이온·나트륨이온·산업용 대용량 셀 포함 | 코드 MIT | 이미 전처리된 파일: [Hugging Face](https://huggingface.co/datasets/Hongwxx/BatteryLife_processed), [Zenodo](https://zenodo.org/records/19688272). 18개 방법을 비교했고, 다른 시계열 분야에서 인기 있는 일부 모델은 배터리 수명 예측에 맞지 않을 수 있다고 보고 (arXiv 초록 기준) |

## 4. 셀 개발 · 공정 개발과 연결되는 연구

수명 예측 자체는 데이터 분석이지만, 지원하는 직무의 언어로 바꾸려면 "**왜 늙었나**"(셀 개발)와 "**어떻게 빨리 걸러내나**"(공정 개발)를 함께 말할 수 있어야 해요. 그 다리가 되는 연구들이에요.

### 4-1. 공정 개발: 화성 공정과 출하 직후 판정

**[Predicting the impact of formation protocols on battery lifetime immediately after manufacturing](https://doi.org/10.1016/j.joule.2021.09.015)** — Weng, Mohtat, Attia, Sulzer, Lee, Less, Stefanopoulou, *Joule* 5(11), 2971–2992 (2021, [arXiv:2203.14158](https://arxiv.org/abs/2203.14158))

- **무엇:** 화성 공정을 빠르게 바꾸면 수명이 어떻게 되는지 보고, 공장 출하 직전에 몇 초 만에 잴 수 있는 신호로 수명을 미리 가늠하는 방법을 제안했어요. 그 신호는 **낮은 충전 상태(SOC)에서 잰 셀 저항**이에요.
- **핵심 결과 (arXiv 초록 기준):** 낮은 SOC 저항이 사이클 수명과 상관이 있고 데이터 기반 수명 예측의 정확도를 높여요. 생산 라인 끝에서 특수 장비 없이 몇 초 만에 얻을 수 있고, 화성 중 소모된 리튬량과 물리적으로 연결돼요.
- **직무 연결:** 공정 개발과 가장 직접 맞닿아요. [아이디어 2](ideas.md#2-5사이클로-수명이-짧을-셀-골라내기-분류)를 "생산 라인 끝에서 단수명 셀 걸러내기"로 넓히는 근거이고, 소모된 리튬(LLI)과 이어져 아이디어 6과도 연결돼요. 데이터는 Battery Archive의 MICH (formation), 셀 40개예요.

**[Data-driven analysis of battery formation reveals the role of electrode utilization in extending cycle life](https://doi.org/10.1016/j.joule.2024.07.024)** — Cui 외 (… Bazant, Sun, Chueh), *Joule* 8(11), 3072–3087 (2024)

- **무엇:** SLAC-스탠퍼드·토요타연구소(TRI)·MIT 연구팀이 데이터 분석과 머신러닝으로 화성 공정의 여러 조건을 훑어, 무엇이 수명을 좌우하는지 찾았어요. 결론은 상식과 반대로 "첫 충전을 빠르게(높은 전류로) 하면 수명이 늘 수 있다"예요.
- **핵심 결과 ([SLAC 보도자료](https://www6.slac.stanford.edu/news/2024-08-29-researchers-discover-surprising-way-jump-start-battery-performance) 기준, 2024-08-29):** 첫 충전을 높은 전류로 한 셀은 평균 수명이 50% 길었고, 첫 충전 시간은 10시간에서 20분으로 줄었어요. 고전류 화성은 초기 리튬의 약 30%를 비활성화했는데(기존 방식은 9%), 이렇게 더 소모된 리튬이 양쪽 전극에 여유 공간을 만들어 양극이 더 효율적으로 사이클한다고 설명해요. 머신러닝 분석에서는 충전 온도와 전류가 가장 큰 영향을 줬어요.
- **직무 연결:** "화성 시간을 줄이면서 수명은 늘린다"는 면접 소재로 좋아요. "처음에 리튬을 더 잃었는데 오히려 오래 갔다"는 설명은 LLI를 깊이 이해하게 해 줘서 아이디어 6과 이어져요. 데이터·코드 공개 정보는 확인한 페이지에 없었어요.

**[Differential voltage analysis for battery manufacturing process control](https://doi.org/10.3389/fenrg.2023.1087269)** — Weng, Siegel, Stefanopoulou 외, *Frontiers in Energy Research* 11, 1087269 (2023)

- **무엇:** 화성 공정에서 이미 얻는 저속 전압 곡선을 dV/dQ(차분 전압)로 분석해 출하 전 검사(라인 끝 공정 관리)에 쓰는 방법이에요. 화성 중 소모된 리튬량, 실제 음극/양극 용량비(N/P 비) 같은 제조 지표를 뽑아요.
- **핵심 결과:** NMC/흑연 셀 두 배치를 비교한 사례에서 화성 중 소모된 리튬이 한 배치에서 약 3배 많았어요 (0.35 vs 0.12 mAh/cm²). 실제 N/P 비는 1.24 vs 1.14였어요. 추정한 양극 용량이 이론값의 약 54\~55%에 그쳐, 저자들은 이를 '접근 불가능한 리튬 문제'로 설명하고 모델을 고쳤어요. 측정 C-rate와 충전/방전 방향이 재현성에 영향을 준다고 지적해요.
- **직무 연결:** 공정 개발과 dQ/dV 아이디어를 한 번에 잇는 논문이에요. 두 생산 배치를 숫자로 비교하는 모습이 "공정 차이를 전압 곡선으로 잡아낸다"는 공정 엔지니어의 실제 업무 그대로예요.
- **공개:** 코드 [wengandrew/dvdq](https://github.com/wengandrew/dvdq) (MATLAB로 C/20 전압 곡선을 전극별 지표로 바꾸고, Python 노트북으로 그림 재현. 라이선스는 확인 안 됨). 데이터는 [3장](#3-공개-데이터셋) 표에 있어요.

### 4-2. 공정 개발 · 품질: 셀 간 편차

**[Intrinsic Variability in the Degradation of a Batch of Commercial 18650 Lithium-Ion Cells](https://doi.org/10.3390/en11051031)** — Devie, Baure, Dubarry, *Energies* 11(5), 1031 (2018)

- **무엇:** 같은 배치(같은 생산 묶음)의 상용 18650 셀들을 같은 조건으로 노화시키며 용량·저항·출력이 셀마다 얼마나 달라지는지 추적했어요. 셀을 뜯지 않는 방법으로 LLI, LAM, 반응 속도 저하를 전극별로 정량했어요.
- **핵심 결과 (초록 기준, 구체 수치 없음):** 초기 셀 간 차이는 작았지만 열화 속도가 셀마다 달라서 노화하면서 상당한 차이가 생겼어요. 이 차이는 **초기 차이와 상관이 없었어요.**
- **직무 연결:** Severson 모델이 초기 용량 대신 '초기 사이클 동안의 변화(ΔQ)'를 쓰는 이유와 맥이 닿아요. 셀 매칭(비슷한 셀끼리 묶기)이 출하 시 측정만으로는 부족한 이유를 설명할 때 쓸 수 있어요.

**[Production caused variation in capacity aging trend and correlation to initial cell performance](https://doi.org/10.1016/j.jpowsour.2013.08.108)** — Baumhöfer, Brühl, Rothgang, Sauer, *J. Power Sources* 247, 332–338 (2014)

- **무엇:** 생산 과정의 편차 때문에 같은 종류 셀들의 용량 노화 추세가 어떻게 달라지는지, 그것이 초기 셀 성능과 어떤 상관이 있는지를 다룬 셀 간 편차 분야의 초기 대표 연구예요.
- **핵심 결과:** 출판사 초록 페이지가 로봇 확인 화면 때문에 열리지 않아 제목·메타데이터만 확인했어요. 그래서 결과는 적지 않았어요.
- **직무 연결:** Devie 2018과 나란히 읽으면 "출하 시 측정값으로 수명을 어디까지 예측할 수 있나"라는 질문을 정리할 수 있어요. [아이디어 5 셀 수 계산기](ideas.md#5-셀-몇-개가-필요한가-계산기)와 데이터 크기 실험의 배경 설명으로도 써요.

**[Challenges and opportunities for high-quality battery production at scale](https://doi.org/10.1038/s41467-025-55861-7)** — Attia, Moch, Herring 외, *Nature Communications* 16 (2025, 무료 사전 공개본 [arXiv:2403.01065](https://arxiv.org/abs/2403.01065))

- **무엇:** 배터리를 기가와트시(GWh) 규모로 대량 생산할 때의 '품질' 문제를 다룬 관점(perspective) 논문이에요. 고장 모드가 어떤 근본 원인에서 오는지, 생산 중 품질을 관리할 때 어떤 맞교환(트레이드오프)이 있는지 설명해요.
- **핵심:** 배터리는 대량 생산이 어렵고 작은 제조 편차에도 민감해서, 업계가 이미 눈에 띄는 안전 사고와 잘 드러나지 않는 신뢰성 문제를 겪었다고 말해요.
- **직무 연결:** Severson 연구의 공저자(Attia)가 산업 쪽에서 쓴 글이라, 우리 수명 예측을 '품질 관리·불량 조기 선별'이라는 산업 언어로 바꿔 말할 때 도움이 돼요.

### 4-3. 셀 개발: 열화 원인 진단 (dQ/dV · dV/dQ)

**[Degradation diagnostics for lithium ion cells](https://doi.org/10.1016/j.jpowsour.2016.12.011)** — Birkl, Roberts, McTurk, Bruce, Howey, *J. Power Sources* 341, 373–386 (2017)

- **무엇:** 복잡한 열화 원인들을 세 가지 '열화 모드'(LLI, 양극 활물질 손실, 음극 활물질 손실)로 묶고, 각각이 개방전압(OCV, 전류가 흐르지 않을 때의 전압) 곡선을 어떻게 바꾸는지 실험으로 처음 보인 논문이에요.
- **핵심 결과:** 리튬·활물질 양을 알고 만든 코인셀로 "세 모드가 OCV에 서로 다른 흔적을 남긴다"는 가정을 처음 실험으로 확인했어요. 이를 바탕으로 셀의 OCV를 맞춰(fitting) 수명 중 어느 시점에서든 각 모드의 양을 계산하는 진단 알고리즘을 만들었어요.
- **직무 연결:** [아이디어 6 LLI vs LAM](ideas.md#6-저율-데이터로-열화-원인-가르기-lli-vs-lam)의 이론적 근거예요. 셀 개발에서 열화 원인 분석 보고서를 쓸 때의 개념 틀 그대로예요.

**[Best practices for incremental capacity analysis](https://doi.org/10.3389/fenrg.2022.1023555)** — Dubarry, Anseán, *Frontiers in Energy Research* 10 (2022)

- **무엇:** dQ/dV(증분 용량 분석, ICA) 곡선을 제대로 얻고 해석하는 실무 지침이에요. Birkl(개념)과 짝을 이루는 '방법' 쪽 자료예요.
- **핵심 내용:**
  - 전압 분해능 약 1 mV 이하의 장비, **저속 정전류 데이터**(C/25 권장, 최소 C/10 이하)를 권해요.
  - 2 mV 간격으로 다시 샘플링하고, 미분 **전에** 전압을 다듬으라고 해요 (미분 후에 다듬으면 봉우리가 왜곡돼요).
  - 용량 손실과 열화는 따로 봐야 해요. 초기에는 용량이 거의 안 줄어도 열화가 '숨어' 있을 수 있어요.
  - 봉우리 면적 분석은 LFP처럼 봉우리 사이가 0 근처로 떨어질 때만 믿을 만해요.
  - 리튬 석출은 충전 곡선의 높은 전압 쪽에 추가 봉우리로 나타나는 경우가 많고, 음극 활물질 손실은 dV/dQ 분석이 가장 좋은 방법일 수 있다고 제안해요.
- **중요한 경고:** Severson 데이터의 방전은 4C로 매우 빨라서, 이 데이터만으로 한 dQ/dV 분석은 왜곡될 수 있어요. 그래서 아이디어 6처럼 **저율 데이터를 따로** 구해야 해요.
- **공개:** 열화 지도 데이터(Mendeley Data), 합성 데이터 생성 도구 'alawa 툴박스(학술용 무료)

### 4-4. 셀 개발: 급속충전 · 리튬 석출 · 무릎

**[Lithium-ion battery fast charging: A review](https://doi.org/10.1016/j.etran.2019.100011)** — Tomaszewska 외 (저자 20명), *eTransportation* 1, 100011 (2019, 오픈 액세스 CC BY 4.0)

- **무엇:** 급속충전을 원자 수준부터 시스템 수준까지 정리한 리뷰예요. 충전 속도를 막는 물리적 한계, 큰 전류로 충전할 때 생기는 열화(대표적으로 리튬 석출), 이를 줄이는 방법, 특히 저온 충전 문제를 다뤄요.
- **핵심:** 큰 충전 전류는 충전을 빠르게 하지만 에너지 효율을 낮추고 용량·출력 감소를 가속할 수 있어요. 핵심 연구 공백으로 "차량 안에서 리튬 석출과 기계적 열화를 믿을 만하게 감지하는 방법"을 꼽아요. 모델 기반 충전 최적화와 열관리(충전 중 냉각, 추운 날 예열, 온도를 고르게 유지하기)가 중요하다고 정리해요.
- **직무 연결:** Severson 셀들은 서로 다른 급속충전 조건으로 시험했으니, 왜 어떤 조건이 수명을 더 깎는지 설명할 때 이 리뷰를 인용할 수 있어요. 셀 개발에서 급속충전 사양(최대 충전 전류, 저온 충전 제한)을 정하는 일의 기본기예요.

**[Detecting the Onset of Lithium Plating and Monitoring Fast Charging Performance with Voltage Relaxation](https://doi.org/10.1021/acsenergylett.0c00831)** — Konz, McShane, McCloskey 외, *ACS Energy Letters* 5(6), 1750–1757 (2020)

- **무엇:** 충전을 멈춘 뒤 전압이 서서히 안정되는 과정(전압 이완)을 미분해서 보는 'dOCV 분석'으로, 셀을 뜯지 않고 리튬 석출이 언제 시작되는지 알아내요. 흔히 측정하는 전압 데이터만 쓰는 게 장점이에요.
- **핵심 결과:** 상온에서 급속충전 한 번만으로 석출 시작을 감지할 수 있고, 사이클을 반복하는 중에도 찾을 수 있어요. 감도는 흑연 1 g당 석출된 리튬 4 mAh 수준이에요.
- **직무 연결:** 아이디어 4에서 "수명이 짧은 충전 조건 = 석출이 일어난 조건일 가능성"이라는 가설을 세울 수 있어요. 다만 **Severson 데이터에 충전 직후 휴지 구간 전압이 충분히 촘촘하게 기록돼 있는지 먼저 확인**해야 해요.

**[Review—"Knees" in Lithium-Ion Battery Aging Trajectories](https://doi.org/10.1149/1945-7111/ac6d13)** — Attia, Bills, Brosa Planella, Dechent, dos Reis, Dubarry 외, *JES* 169(6), 060517 (2022)

- **무엇:** 용량이 천천히 줄다가 어느 순간 급격히 꺾이는 '무릎' 현상을 정리한 리뷰예요. 무릎이 왜 생기는지, 설계·사용 조건이 무릎 시점에 어떤 영향을 주는지, 예측이 왜 어려운지를 다뤄요.
- **핵심 (초록 기준):** 무릎을 만드는 내부 변화를 snowball · hidden · threshold 세 유형으로 나누고, 리튬 석출, 전극 포화, 저항 증가, 전해질·첨가제 고갈, 퍼콜레이션 제한 연결성(전극 안 연결 통로가 끊기는 문제), 기계적 변형의 6가지 경로를 논의해요. 일부는 전기화학 측정으로 감지되지 않고, 무릎 예측은 아직 어렵다고 결론지어요.
- **직무 연결:** [아이디어 8 무릎 예측](ideas.md#8-무릎knee-지점-예측)의 출발점이에요. 아이디어 6과 함께 보면 "왜 갑자기 꺾이나"를 설명할 수 있어요. 셀 개발 면접에서 자주 나오는 주제예요.

### 4-5. 셀 개발: 수명 시험 조건 자체가 숫자를 바꾼다

**[Dynamic cycling enhances battery lifetime](https://doi.org/10.1038/s41560-024-01675-8)** — Geslin, Xu, Ganapathi, Moy, Chueh, Onori, *Nature Energy* (2024년 12월 온라인)

- **무엇:** Severson 논문 공저자인 Chueh 교수가 참여한 스탠퍼드 연구예요. 실험실에서 흔히 쓰는 '일정 전류 방전' 대신 실제 전기차 주행처럼 전류가 오르내리는 방전으로 시험했더니 수명이 더 길게 나왔어요.
- **핵심 결과 (초록 기준):** 평균 전류와 전압 범위를 같게 두었을 때, 주행을 흉내 낸 동적 방전이 일정 전류 방전보다 수명 종료 시 등가 완전 사이클 기준으로 **최대 38%** 수명을 늘렸어요.
- **직무 연결:** "시험 프로토콜 자체가 수명 숫자를 바꾼다"는 결과라, 우리 결과(일정 조건의 실험실 데이터)의 한계를 설명할 때 써요. 셀 개발의 수명 평가 조건 설계 이야기로도 이어져요. 데이터 공개 여부는 확인하지 못했어요.

## 5. 시뮬레이션: 우리가 돌릴 수 있나

> 실행한 스크립트 3개와 결과 파일(JSON·CSV), 설치 명령, 전체 결과 표는 [**simulations/pybamm/README.md**](../simulations/pybamm/README.md)에 있어요.
> 여기서는 결론과 해석 위주로 적고, 숫자는 핵심만 옮겼어요.

### 짧은 답: 네, 돌릴 수 있어요

- **돌리는 것 자체는 쉬웠어요.** 무료 오픈소스 [PyBaMM](https://doi.org/10.5334/jors.309)(파이썬 배터리 시뮬레이터) 26.8.0.0을 이 PC(AMD Ryzen 5 5600XT 6코어/12스레드, RAM 31.9GB, Windows 11 Home)에 설치해서 실습 세 가지를 돌렸어요.
  - LFP 셀 방전 한 번: **약 0.06\~0.21초**
  - 가장 자세한 모델(DFN)에 열화 4종(SEI 성장, 리튬 석출, 입자 균열, 활물질 손실)을 모두 켠 50사이클: **약 20초** (사이클당 약 0.4초)
  - 슈퍼컴퓨터는 필요 없어요.
- **어려웠던 건 두 가지예요.**
  1. **윈도우 설치 함정** — 세 번 만에 성공했어요 (아래 표).
  2. **우리 셀에 맞는 파라미터(모델에 넣는 재료·셀 특성 숫자) 구하기** — 남의 셀 값을 빌려 쓰면 숫자가 의미 없어요 (실습 c).
- **그래서:** 포트폴리오에서는 시뮬레이션을 "우리 셀의 수명을 맞히는 도구"가 아니라 "**데이터 분석 결과를 물리로 설명하는 보조 그림**"으로 쓰는 게 가장 현실적이고 효과도 커요.

### 설치: 세 번 만에 성공

프로젝트의 `.venv`에는 아무것도 설치하지 않고, **따로 만든 가상환경**(프로젝트마다 따로 쓰는 파이썬 설치 공간)에서 시험했어요.

| 시도 | 버전 | 결과 | 걸린 시간 | 원인 → 해결 |
|---|---|---|---|---|
| 1차 | Python 3.13.15, pybamm 26.10.1.1 시도 | ❌ 설치 중 실패: `[WinError 206] 파일 이름이나 확장명이 너무 깁니다` | 34초 | 가상환경 폴더 경로가 136자(임시 폴더 124자 + `\pybamm-venv`)여서, `\Lib\site-packages\`(19자)와 casadi 안의 가장 긴 파일 경로(108자)를 합치면 263자로 윈도우의 기본 경로 길이 제한(약 260자)을 넘었어요. → 가상환경을 **짧은 경로**에 다시 만들었어요 |
| 2차 | pybamm 26.10.1.1, pybammsolvers 0.11.1, casadi 3.8.1 | ❌ 설치는 됐지만 `import pybamm`에서 실패: `DLL load failed while importing _casadi: 애플리케이션 제어 정책에서 이 파일을 차단했습니다.` | 34초 | 이 PC는 윈도우 보안 기능 '스마트 앱 컨트롤'이 켜져 있어서, casadi 3.8.1의 서명 없는 DLL(프로그램 부품 파일)을 막았어요. pybammsolvers 0.11.1이 casadi==3.8.1을 고정으로 요구해서 casadi만 내릴 수도 없었어요 |
| 3차 | **pybamm 26.8.0.0**, pybammsolvers 0.9.1, **casadi 3.7.2**, numpy 2.5.4, scipy 1.18.1 (Python 3.13.15) | ✅ 설치·import 모두 성공 | 약 38초 | `pip install pybamm "casadi==3.7.2"`로 casadi 버전을 고정하니 pip가 맞는 PyBaMM(26.8.0.0)을 자동으로 골랐어요 |

- **casadi 버전 시험:** 3.7.2와 3.7.0은 정상으로 불러와졌고, 3.6.7은 차단됐어요. numpy, scipy, pandas, pybammsolvers는 처음부터 정상이었어요.
- **보안 설정은 건드리지 않았어요.** 스마트 앱 컨트롤 상태는 읽기만 했고, 긴 경로 허용(LongPathsEnabled)을 포함해 시스템·보안 설정은 하나도 바꾸지 않았어요. PyBaMM 때문에 이 기능을 끄는 것도 권하지 않아요.
- **사용 통계 끄기:** PyBaMM은 사용 통계 패키지(posthog)를 함께 설치해서, 실행할 때 환경 변수 `PYBAMM_DISABLE_TELEMETRY=true`로 껐어요.
- **나중에 이 프로젝트에 PyBaMM을 넣는다면:** 이 PC에서는 requirements에 `casadi==3.7.2`를 고정하는 게 안전해요. 이후 버전이 허용될지는 그때 다시 확인해야 해요.

### 실습 세 가지

모델 이름: SPM(단일 입자 모델, 가장 단순) · SPMe(SPM에 전해질을 더한 모델) · DFN(가장 자세한 표준 모델)

| 실습 | 셀 · 모델 | 조건 | 걸린 시간 | 결과 |
|---|---|---|---|---|
| (a) 모델 비교 | Prada2013 (PyBaMM 내장 LFP/흑연, 정격 2.3 Ah) · SPM · SPMe · DFN | 0.5C · 1C · 2C 정전류 방전, 2.0 V까지 | 9번 합계 3.7초 (프로세스 시작·import 포함). 한 번에 SPM 약 0.06초 · SPMe 약 0.15초 · DFN 약 0.21초 | 방전 용량 0.5C 2.119 Ah(정격의 92%) · 1C 1.939 Ah(84%) · 2C 1.599 Ah(70%). 세 모델 차이 0.001 Ah 이하. 중간 전압 약 3.14\~3.23 V로 LFP 특유의 평평한 구간이 나와요 |
| (b) 열화 | OKane2022 (LG M50, 정격 5 Ah, NMC811/흑연+SiOx) · DFN + SEI 성장 · 부분 가역 리튬 석출 · 입자 팽창과 균열 · 균열면 SEI · 응력에 의한 활물질 손실 ([공식 결합 열화 예제](https://docs.pybamm.org/en/latest/source/examples/notebooks/models/coupled-degradation.html)와 같은 옵션) | 1사이클 = 1C 방전 → 휴지 10분 → 충전(4.2 V까지) → 4.2 V 유지(C/20까지) → 휴지 10분. **충전 속도만** 0.5C · 1C · 2C로 바꿔 각각 50사이클 | 18.6 · 19.6 · 21.1초 (사이클당 약 0.4초) | 아래 표 |
| (c) 남의 파라미터 빌리기 (시연용) | Prada2013에는 SEI 파라미터가 없어서 OKane2022의 값 78개를 빌려 넣음 · SPMe + SEI 성장 | Severson과 비슷한 고속 조건: 4C 충전(3.6 V까지, C/20까지 유지), 4C 방전(2.0 V까지), 100사이클 | 계산 2.9초 (기록된 전체 실행 4.8초) | 1사이클 1.127 Ah(초기 상태 영향), 5사이클 1.1772 Ah가 최고, 100사이클 1.1766 Ah. 감소 약 0.05%, LLI 0.033% |

**실습 (b) 핵심 숫자** (음극·양극 활물질 손실까지 담은 전체 표는 [README](../simulations/pybamm/README.md)에 있어요)

| 충전 속도 | 1C 방전 용량 변화 (1 → 50사이클) | LLI (리튬 재고 손실) | 리튬 석출로 잃은 용량 |
|---|---|---|---|
| 0.5C | 4.9357 → 4.9062 Ah (−0.60%) | 0.59% | 0.0216 Ah |
| 1C | 4.9357 → 4.9038 Ah (−0.65%) | 0.62% | 0.0259 Ah |
| 2C | 4.9357 → 4.8912 Ah (−0.90%) | 0.76% | 0.0364 Ah |

→ **충전이 빠를수록 리튬 석출이 늘고 용량이 더 빨리 줄어드는 경향**이 물리 모델에서도 재현됐어요. 음극 활물질 손실은 세 조건 모두 약 0.075%로 비슷했고, 양극 활물질 손실은 충전이 빠를수록 컸어요.

### 결과를 읽을 때 주의할 점

- **(b)는 우리와 다른 셀이에요.** OKane2022는 NMC811/흑연+SiOx 5 Ah 원통형 셀(LG M50) 파라미터라 LFP인 Severson 셀과 화학계가 달라요. 또 50사이클이라 감소량이 1% 미만이고, 실제 셀 수명과 비교할 수 있는 값이 아니에요. **경향**으로만 말해요.
- **(c)의 숫자는 그대로 인용하면 안 돼요.** 실제 Severson 셀은 수백\~수천 사이클 만에 용량이 20% 줄어드는데, 이 계산은 100사이클 동안 거의 줄지 않았어요. 다른 셀의 열화 파라미터를 빌려 쓰면 숫자가 의미 없다는 것, 즉 **'돌리기'보다 '파라미터를 실제 데이터에 맞추기'가 진짜 어려운 일**이라는 걸 보여 주는 결과예요. [O'Kane 외(2022)](https://doi.org/10.1039/D2CP00417H)도 열화 모델의 파라미터화가 여전히 큰 과제라고 적었어요.
- **더 긴 계산은 아직 안 해 봤어요.** 1,000사이클 정도도 같은 속도라면 몇 분 수준으로 예상되지만, 해 보지 않은 추정이에요. 열화가 심해지는 구간에서는 계산이 느려지거나 멈출 수 있어요.

### PyBaMM 문서에서 확인한 것: LFP 세트는 하나뿐

- [파라미터 세트 목록](https://docs.pybamm.org/en/latest/source/api/parameters/parameter_sets.html) 중 리튬이온 LFP는 **Prada2013 하나뿐**이에요. 이 세트는 정격 2.3 Ah이고 SEI 파라미터가 없어서, Severson 셀(LFP/흑연 1.1 Ah)과 같은 셀이 아니에요.
- 리튬 석출 파라미터가 있는 세트는 Mohtat2020, OKane2022_graphite_SiOx_halfcell 등이에요.
- 여러 SEI 값에는 "예시용으로 맞춘 값이라 실제 셀을 대표하지 않는다"는 주석이 달려 있어요.
- [설치 문서](https://docs.pybamm.org/en/latest/source/user_guide/installation/index.html): Python 3.10\~3.15, 윈도우·맥·리눅스 지원. 기본 설치 명령은 `pip install pybamm`이지만, 이 PC에서는 위처럼 casadi 버전 고정이 필요했어요.

### 시뮬레이션에서 꺼낼 수 있는 값

- **방전 곡선과 방전 용량** (실습 a): 1C 방전 곡선 50점을 모델별 CSV로 저장했어요.
- **사이클별 열화 원인 숫자** (실습 b): 사이클마다 방전 용량과 LLI, 음극·양극 LAM, SEI·리튬 석출로 잃은 용량을 JSON으로 저장했어요. 결합 열화 예제가 그리는 `Loss of lithium inventory [%]`, `Loss of active material in negative/positive electrode [%]`와 같은 값이에요. 실제 셀에서는 dQ/dV·OCV 분석으로 간접 추정해야 하는 값([4-3](#4-3-셀-개발-열화-원인-진단-dqdv--dvdq))을 모델 안에서 바로 볼 수 있다는 게 장점이에요.
- **수명이 끝나는 경로:** [O'Kane 외(2022)](https://doi.org/10.1039/D2CP00417H)는 음극의 열화 메커니즘 네 가지를 PyBaMM에서 결합했고 (셋 이상을 직접 결합한 첫 시도라고 밝혀요), 한 셀이라도 쓰는 방식에 따라 수명 종료 경로가 다섯 가지로 나뉜다고 보고해요. Severson 데이터에서 충전 조건별로 수명이 크게 갈리는 현상을 설명하는 물리적 배경이 돼요.

### 현실적인 것 vs 아닌 것

| 구분 | 할 일 | 연결 아이디어 |
|---|---|---|
| ✅ 현실적 | PyBaMM으로 LLI만, LAM만, 리튬 석출만 켠 '가상 셀'을 만들고 10사이클과 100사이클의 Q(V) 차이 곡선이 각각 어떻게 생기는지 그리기. Severson 특징이 왜 수명과 연결되는지 물리로 설명하는 그림이 돼요 | [9 특징 해석](ideas.md#9-특징이-왜-통하나-해석) |
| ✅ 현실적 | 충전 속도를 바꿨을 때 리튬 석출이 늘어나는 경향을 정성적으로 보여 주기 (실습 b) | [4 급속충전](ideas.md#4-급속충전-조건과-수명) |
| ✅ 현실적 | 시뮬레이션을 직접 돌리지 않고, Dubarry의 LFP 가상 곡선 70만 개(CC BY 4.0)로 LLI·LAM 판별 연습 | [6 LLI vs LAM](ideas.md#6-저율-데이터로-열화-원인-가르기-lli-vs-lam) |
| ❌ 비현실적 | 물리 모델만으로 Severson 124개 셀의 수명을 그대로 맞히기. LFP 세트가 하나뿐이고 열화 파라미터도 없어요 (실습 c) | — |
| ⏸ 나중에 | 파라미터를 실험 데이터에 맞추기 ([PyBOP](https://doi.org/10.21105/joss.07874)). 가능하지만 난이도가 높아요 | — |
| ⏸ 후순위 | COMSOL(유료), BattMo(MATLAB 유료 또는 Julia). 초보 포트폴리오에는 PyBaMM보다 불리해요 | — |

> **정리:** "시뮬레이션은 우리가 돌리기 힘들다"는 반만 맞아요. 돌리는 것은 일반 PC에서 몇 초\~몇 분이면 돼요. 하지만 실제 셀과 숫자까지 맞추는 것은 연구 수준의 일이에요.

### 도구와 자료 비교

| 도구 · 자료 | 비용 | 언어 | 무엇 | 우리에게 |
|---|---|---|---|---|
| [PyBaMM](https://doi.org/10.5334/jors.309) | 무료 (오픈소스) | Python | SPM · SPMe · DFN, 열화 결합 예제. 모델을 '수식 트리'로 표현하고 모듈식 단계로 풀어요 | **이 PC에서 설치·실행 확인 (26.8.0.0).** 지금 쓰는 파이썬 흐름과 바로 이어져요 |
| [Dubarry LFP 합성 데이터](https://data.mendeley.com/datasets/bs2j56pn7y/3) | 무료 (CC BY 4.0) | (데이터) | 흑연/LFP 가상 곡선 70만 개 이상 | **추천.** 시뮬레이션을 직접 돌리지 않고 결과를 내려받아 써요 |
| 'alawa 툴박스 ([Dubarry 외 2012](https://doi.org/10.1016/j.jpowsour.2012.07.016)가 바탕) | 학술용 무료 | — | 양극·음극 전압 곡선을 이동·축소해 LLI · LAM을 '합성' | 미분방정식을 푸는 무거운 계산이 아니라 개념을 따라가기 쉬워요. HNEI 공식 페이지는 이번 확인 때 인증서 오류로 열리지 않았어요 |
| [PyBOP](https://doi.org/10.21105/joss.07874) | 무료 (오픈소스) | Python | 모델 파라미터를 실험 데이터에 맞추고 설계를 최적화 | 실습 (c)의 다음 단계 후보. 초보에게는 어려워 우선순위 낮음 |
| [BattMo](https://arxiv.org/abs/2512.17933) (노르웨이 SINTEF) | 오픈소스 (단, MATLAB은 유료) | MATLAB · Julia | DFN 기반, 열 해석, 3D 형상(원통형·각형), SEI, 실리콘/흑연 전극. 입력은 JSON, 기울기 기반 파라미터 보정 지원 | 3D·열이 궁금할 때 후보. 파이썬만 쓰는 우리에게는 진입 장벽이 높아요 |
| [COMSOL Battery Design Module](https://www.comsol.com/battery-design-module) | 유료 (가격 비공개, 영업 문의) | 상용 프로그램 | 1D·2D·3D Newman 모델, SEI, 리튬 석출, 전기화학-열 결합, 팩 열폭주 전파 | 배터리 회사 셀 설계·CAE(컴퓨터 해석) 부서에서 많이 써서 이름은 알아 두기. 개인 포트폴리오에는 비현실적이에요 |

### 공부용 자료

- **[Brosa Planella 외 2022](https://doi.org/10.1088/2516-1083/ac7d31) (모델 지도):** 가장 자세한 모델에서 DFN을 거쳐 SPM까지 체계적으로 유도하고 장단점을 비교해요. 응용에 따라 필요한 정밀도가 다르다고 말해요 (설계에는 복잡한 모델, 전기차 실시간 제어에는 빠르고 단순한 모델). 실습 (a)에서 세 모델의 방전 용량이 0.001 Ah 이하로 거의 같았는데, 어떤 조건에서 단순 모델이 틀리는지 공부할 때 봐요.
- **[Edge 외 2021](https://doi.org/10.1039/d1cp00359c) (열화 입문):** O'Kane 논문과 같은 그룹(Imperial College, Offer 교수팀)의 리뷰예요. SEI, 리튬 석출, 활물질 손실이 무엇이고 어떻게 이어지는지, 시뮬레이션 옵션 이름을 이해하는 바탕이 돼요.
- **[Prada 외 2013](https://doi.org/10.1149/2.053304jes) (LFP 노화 모델):** PyBaMM Prada2013 세트의 출처예요. 음극 SEI 막이 자라 기공(공극률)이 줄어드는 효과로 용량 감소와 출력 감소를 연결했고, 캘린더·사이클 노화를 모두 보정·검증했어요. 사이클 중 SEI 균열이 노화를 가속할 수 있다고 제안해요.
- **[Aykol 외 2021](https://doi.org/10.1149/1945-7111/abec55) (물리 + 머신러닝):** TRI·스탠퍼드·MIT 연구자들이 순수 물리 모델과 순수 데이터 모델의 장단점을 비교하고, 둘을 섞는 여러 구조의 구현 난이도·장점·한계·실현 가능성을 평가했어요. "시뮬레이션으로 만든 특징을 머신러닝 입력에 더하기" 같은 아이디어를 정당화할 때 인용해요.

## 6. DFT는 우리 영역인가

### 짧은 답: 직접 돌리는 건 우리 영역이 아니에요

DFT(밀도범함수이론)는 양자역학으로 원자와 전자의 에너지를 계산하는 방법이에요. 배터리에서는 **소재** 수준의 질문에 써요.
예를 들면 어떤 양극재의 이론 전압은 얼마인지, 리튬이 결정 속에서 얼마나 쉽게 움직이는지, 구조가 안정한지, 전해질이나 첨가제가 어떤 전압에서 분해되는지 같은 질문이에요.
보통 원자 수십\~수백 개 크기만 다루고, 대학·기업의 계산 서버나 슈퍼컴퓨터에서 돌려요.

[Urban · Seo · Ceder(2016)](https://doi.org/10.1038/npjcompumats.2016.2) 리뷰를 보면 DFT가 다루는 대상은 전극·전해질 **소재**(결정, 분자)이지 **셀**이 아니에요.

- **DFT로 계산하는 것:** 평형 셀 전압, 0 K 및 유한 온도의 전압 곡선, 이온 이동도, 열·전해질 안정성
- **잘 되는 것:** 정돈된 결정 고체(대부분의 전극 소재, 고체 전해질). 이제 신뢰할 만하고 정량적이에요.
- **아직 어려운 것:** 분자 · 무질서 · 비정질 상 (SEI 같은 것)
- 리뷰는 DFT+U, 하이브리드 범함수 같은 계산 방식도 비교하지만, 셀·공정 직무에서는 이름만 알아도 충분해요.

### 크기별로 보면

| 크기 | 다루는 것 | 대표 도구 | 주로 쓰는 곳 | 우리 프로젝트 |
|---|---|---|---|---|
| 원자 · 소재 | 양극재 조성, 코팅·도핑, 고체 전해질, 전해액 첨가제 | DFT ([Quantum ESPRESSO](https://www.quantum-espresso.org/), [VASP](https://www.vasp.at/)), [Materials Project](https://doi.org/10.1063/1.4812323) | 배터리·소재 회사의 소재 선행개발·계산재료 조직, 대학 계산재료 연구실 | 결과를 **찾아보기만** |
| 전극 · 셀 내부 | 전극 두께, 공극률, 로딩, N/P 비, 전해액 양 | 연속체 모델 (PyBaMM, BattMo, COMSOL) | 셀 설계 · CAE 부서 | 현상 설명용으로 조금 (5장) |
| 셀 시험 데이터 | 화성 조건, 충전 프로토콜, 수명 시험 | dQ/dV · EIS(교류 임피던스 측정) · 수명 데이터 분석 | 셀 개발 · 공정 개발 | **주력** (Severson 재현) |

셀 개발·공정 개발 직무가 다루는 건 표의 아래 두 줄(셀과 공정 수준)이에요. 이 수준의 계산 도구는 DFT가 아니라 PyBaMM·COMSOL 같은 연속체 모델과 데이터 분석이에요.
DFT가 중요한 때는 소재 쪽에서 새 소재 후보를 실험 전에 걸러 낼 때예요.

### 그래도 알아 두면 좋은 때

1. **새 소재가 셀에 들어올 때:** 실리콘 음극, 하이니켈 양극, 새 첨가제 같은 소재 논문에 나오는 DFT 결과를 읽고 "왜 이 소재가 이런 전압이나 안정성을 보이는지" 이해할 수 있으면 좋아요.
2. **다단계(멀티스케일) 흐름을 말로 설명할 때:** "DFT(원자) → 소재 물성 → 셀 모델(PyBaMM) → 수명 데이터"로 이어지는 흐름을 설명할 수 있으면 면접에서 플러스가 돼요.
3. **직접 계산하지 않고 찾아볼 때:** Materials Project(무료 웹 DB)에서 이미 DFT로 계산된 소재 값을 볼 수 있어요.

### 도구 이름

- **Quantum ESPRESSO:** 대표적인 무료 오픈소스 DFT 프로그램이에요 (공식 사이트 소개: 밀도범함수이론, 평면파, 의사퍼텐셜에 기반한 코드 모음). 인용은 2009년 논문(*J. Phys.: Condens. Matter* 21, 395502)과 2017년 논문을 하라고 안내해요. 셀·공정 직무에서 직접 쓸 일은 드물어요.
- **VASP:** 라이선스를 신청해야 쓰는 유료(독점) 프로그램이에요. 소재 개발 부서나 계산재료 연구실 채용 공고에서 자주 보이는 이름이라 "이런 게 있다" 정도면 충분해요.
- **Materials Project:** DFT로 이미 계산해 둔 수많은 무기 소재 데이터를 웹에서 무료로 볼 수 있어요. 배터리 전극 소재용 [Battery Explorer 앱](https://docs.materialsproject.org/apps/explorer-apps/battery-explorer)도 있어요. LFP나 NMC 소재의 계산 전압을 확인해 보는 정도는 셀 직무 지원자도 할 수 있어서, 면접에서 "소재 → 셀" 연결을 이해한다는 걸 보여 줄 때 쓸 수 있어요.

> **면접 한 줄 예시:** "DFT는 원자 수준에서 소재 물성을 예측하는 계산이라 소재 개발 쪽 도구이고, 저는 셀 수준에서 데이터 분석과 물리 기반 셀 모델(PyBaMM)로 열화를 해석하는 쪽에 집중했습니다."

## 7. 그래서 다음에 할 만한 것

### 추천 순서 (작은 것부터)

| 순서 | 할 일 | 연결 아이디어 | 왜 | 필요한 것 |
|---|---|---|---|---|
| 1 | PyBaMM으로 LLI만 · LAM만 · 리튬 석출만 켠 가상 셀을 만들어 ΔQ(V) 모양 비교 | [9 특징 해석](ideas.md#9-특징이-왜-통하나-해석) | "ΔQ 분산이 왜 수명을 알려 주나"를 물리 그림으로 설명할 수 있어요. 설치는 이미 끝났고 실습 (b) 스크립트를 고쳐 쓰면 돼요 | [simulations/pybamm](../simulations/pybamm/README.md). 우리 셀과 같은 셀이 아니니 **경향**으로만 말해요 |
| 2 | Dubarry LFP 합성 데이터로 dQ/dV → LLI · LAM 맞히기 연습 | [6 LLI vs LAM](ideas.md#6-저율-데이터로-열화-원인-가르기-lli-vs-lam), [9 해석](ideas.md#9-특징이-왜-통하나-해석) | 정답이 붙은 데이터로 방법을 먼저 검증한 뒤 실제 저율 데이터로 갈 수 있어요 | 합성 데이터(CC BY 4.0). 크기 확인 후 진행 |
| 3 | BatteryML의 variance 모델 설정(`matr_1.yaml`)으로 우리 RMSE 138 / 196사이클을 교차 검증 | 재현 검증 ([데이터 크기 실험](data_size.md)) | 남이 만든 구현과 숫자가 맞으면 재현을 더 믿을 수 있어요 | BatteryML 설치(MIT), Severson 데이터 |
| 4 | HUST 77셀에 우리 단순 모델을 그대로 적용 | [7 일반화](ideas.md#7-다른-셀에도-통할까-일반화) | 같은 LFP/흑연 1.1 Ah라 "조건만 바뀌어도 통하나"를 깨끗하게 볼 수 있어요 | HUST 데이터(CC BY 4.0). 크기 확인 후 진행 |
| 5 | (여유가 있으면) Aykol 2021의 '물리 + 머신러닝 결합' 구조 중 하나를 적용 | [9 특징 해석](ideas.md#9-특징이-왜-통하나-해석) | 1번에서 만든 시뮬레이션 특징을 머신러닝 입력에 더해 보는 식으로 이어 갈 수 있어요 | 1·2번 결과 |

### 아이디어별 읽을거리

| 아이디어 | 먼저 읽을 것 | 쓸 데이터 · 도구 |
|---|---|---|
| [1 몇 사이클이면 충분할까](ideas.md#1-몇-사이클이면-충분할까-시간-축-실험) (결과: [cycles.md](cycles.md)) | Attia 2020 (초기 예측으로 시험 기간 단축) | — |
| [2 5사이클 선별](ideas.md#2-5사이클로-수명이-짧을-셀-골라내기-분류) | Fermín-Cueto 2020 (3\~5사이클 분류 88\~90%), Weng 2021 (출하 직후 낮은 SOC 저항), Attia 2025 (품질 관점) | MICH (formation) |
| [3 예측 구간](ideas.md#3-예측에-오차-범위-붙이기-불확실성) | Zhang 2023 (분위수 회귀 숲), Li 2024 (계층적 베이지안), Fermín-Cueto 2020 | BatteryML의 가우시안 프로세스 |
| [4 급속충전](ideas.md#4-급속충전-조건과-수명) | Attia 2020, Tomaszewska 2019, Konz 2020, O'Kane 2022, Geslin 2024 | PyBaMM 실습 (b) |
| [5 셀 수 계산기](ideas.md#5-셀-몇-개가-필요한가-계산기) | Devie 2018, Baumhöfer 2014 (셀 간 편차), Attia 2021 (적은 데이터와 단순 모델) | [데이터 크기 실험](data_size.md) |
| [6 LLI vs LAM](ideas.md#6-저율-데이터로-열화-원인-가르기-lli-vs-lam) | Birkl 2017 (개념) → Dubarry · Anseán 2022 (실무, **4C 데이터 경고**) → Weng 2023 (dV/dQ 코드) | Dubarry 합성 데이터, ISU-ILCC 저율 RPT, Severson 저율 데이터, wengandrew/dvdq |
| [7 일반화](ideas.md#7-다른-셀에도-통할까-일반화) | Paulson 2022, Li 2024, Zhang 2025 | HUST → ISU-ILCC → Sandia, BatteryML · BatteryLife |
| [8 무릎 예측](ideas.md#8-무릎knee-지점-예측) | Knees 리뷰 2022, Fermín-Cueto 2020, O'Kane 2022 (수명 종료 경로) | Severson 데이터 |
| [9 특징 해석](ideas.md#9-특징이-왜-통하나-해석) | Attia 2021 (capacity matrix), Zhang 2023 (순열 중요도), Aykol 2021 | Dubarry 합성 데이터, PyBaMM |

- **dQ/dV 주의:** Severson 데이터의 4C 방전 곡선으로 dQ/dV를 그리면 Dubarry · Anseán(2022)의 권장(C/25, 최소 C/10 이하)보다 훨씬 빨라요. 봉우리 해석을 할 때는 이 한계를 함께 적어 두는 게 좋아요.
- **내려받기:** HUST, ISU-ILCC, 합성 데이터, Severson 저율 데이터 모두 내려받기 전에 크기와 라이선스를 먼저 확인하고 내려받을지 정해요.

## 8. 확인 방법

이 문서 [1장](#1-한눈에-보기) 표의 참고 자료 40개(논문 35편 · 데이터셋 2개 · 도구·공식 문서 페이지 3개)는 자료를 찾은 에이전트와 다른 별도의 에이전트가 링크를 하나씩 열어, 서지 정보(제목·저자·연도·학술지)와 내용을 Crossref · arXiv · 출판사 · 데이터 페이지와 교차 확인했어요. 그중 2개는 출판사 초록 페이지가 열리지 않아, Paulson 2022는 서지 정보와 보도자료로, Baumhöfer 2014는 서지 정보만 확인했어요. Quantum ESPRESSO 논문은 링크가 프로그램 홈페이지로 걸려 있던 것을 나중에 논문 DOI로 바꿨고, Crossref에서 서지 정보가 같은지 다시 확인했어요. 확인되지 않는 항목은 빼기로 했고, 이번에 빠진 항목은 **0개**예요. 핵심 결과는 대부분 초록 · 보도자료 · 데이터 페이지 기준이라 본문에는 더 자세한 조건이 있을 수 있고, 확인하지 못한 부분(라이선스, 코드 공개 여부, 파일 크기 등)은 "확인 안 함"으로 적었어요. 시뮬레이션 숫자는 이 PC에서 실제로 돌린 실행 기록에서 그대로 옮겼고, 스크립트와 결과 파일은 [simulations/pybamm/](../simulations/pybamm/README.md)에 있어요.
