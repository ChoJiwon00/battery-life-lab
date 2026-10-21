# Battery Life Lab

시판 LFP/흑연 셀 124개의 충방전 데이터(Severson 외, 2019)로 배터리 수명을 일찍 예측하고 열화 원인을 분석하는 개인 프로젝트.
이차전지 셀 개발·공정 개발 직무 포트폴리오용이다.

## 대화 규칙

- 사용자에게 보내는 모든 답변과 보고, 작업 목록, 버튼은 **한국어**로 쓴다.
- 사용자는 개발 초보다. 전문 용어는 짧게 풀어서 설명한다.

## 작업 방법

- Python 명령은 프로젝트 루트에서 `.venv\Scripts\python.exe`로 실행한다. 예: `.venv\Scripts\python.exe -m battery_lab info`
- 테스트: `.venv\Scripts\python.exe -m pytest -q`
- `data/`(내려받은 데이터, 약 7.7GB)는 GitHub에 올리지 않는다.
- 큰 데이터를 내려받기 전에는 파일 이름·출처·크기를 알려 주고 사용자에게 먼저 묻는다.
- 진행 단계와 상태는 README의 "진행 상황" 표에 맞춰 고친다.
