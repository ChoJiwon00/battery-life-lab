"""데이터 출처와 논문(Severson 외, 2019)이 쓴 셀 고르기 규칙.

규칙은 저자가 공개한 데이터 정리 코드(Load Data.ipynb)를 그대로 옮겼다.
https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # 내려받은 .mat 원본
INTERIM = DATA / "interim"    # 배치별로 셀마다 추린 값 (extract)
PROCESSED = DATA / "processed"  # 논문 규칙으로 고른 124개 셀 (assemble)

PROJECT_URL = "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"
LICENSE = "CC BY 4.0"


@dataclass(frozen=True)
class Batch:
    key: str        # b1, b2, b3 (셀 이름 앞부분: b1c0, b1c1, ...)
    date: str       # 시험 시작일
    file_id: str
    filename: str
    size: int       # 바이트

    @property
    def url(self) -> str:
        return f"https://data.matr.io/1/api/v1/file/{self.file_id}/download"


# 크기는 2026-10-22에 서버가 알려 준 값이다. 내려받은 뒤 이 크기와 같은지 확인한다.
BATCHES = (
    Batch("b1", "2017-05-12", "5c86c0b5fa2ede00015ddf66",
          "2017-05-12_batchdata_updated_struct_errorcorrect.mat", 3_025_320_241),
    Batch("b2", "2017-06-30", "5c86bf13fa2ede00015ddd82",
          "2017-06-30_batchdata_updated_struct_errorcorrect.mat", 2_007_331_155),
    Batch("b3", "2018-04-12", "5c86bd64fa2ede00015ddbb2",
          "2018-04-12_batchdata_updated_struct_errorcorrect.mat", 3_236_690_412),
)
BATCH_BY_KEY = {b.key: b for b in BATCHES}


@dataclass(frozen=True)
class Rules:
    removed: tuple[str, ...]
    # (다음 배치에서 이어 시험한 셀, 원래 셀, 원래 셀의 수명에 더할 사이클 수)
    carryover: tuple[tuple[str, str, int], ...]


PAPER_RULES = Rules(
    removed=(
        "b1c8", "b1c10", "b1c12", "b1c13", "b1c22",            # 80%에 닿기 전에 시험이 끝남
        "b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43",   # 측정 잡음이 큰 채널
    ),
    carryover=(
        ("b2c7", "b1c0", 662), ("b2c8", "b1c1", 981), ("b2c9", "b1c2", 1060),
        ("b2c15", "b1c3", 208), ("b2c16", "b1c4", 482),
    ),
)


def cell_order(cell_id: str) -> tuple[int, int]:
    """b1c10이 b1c9 뒤에 오도록 숫자로 정렬한다."""
    batch, cell = cell_id[1:].split("c")
    return int(batch), int(cell)


def select(cell_ids, rules: Rules = PAPER_RULES) -> tuple[list[str], dict[str, str]]:
    """논문 규칙으로 셀을 고르고 학습·테스트로 나눈다.

    돌려주는 값: (최종 셀 순서, {셀: "train" | "test" | "secondary_test"})
    나누기도 저자 코드와 같다: 1·2배치를 합친 순서에서 짝수 번째와 마지막 셀은 테스트, 홀수 번째는 학습,
    3배치는 전부 2차 테스트. 논문 기준으로 학습 41 · 테스트 43 · 2차 테스트 40개가 된다.
    """
    merged_away = {src for src, _, _ in rules.carryover}
    kept = sorted((c for c in set(cell_ids) if c not in rules.removed and c not in merged_away), key=cell_order)
    first_two = [c for c in kept if not c.startswith("b3")]
    n12 = len(first_two)
    test = set(range(0, n12, 2)) | ({n12 - 1} if n12 else set())  # 논문에서 n12 - 1 = 83
    split = {}
    for n, c in enumerate(kept):
        if n >= n12:
            split[c] = "secondary_test"
        else:
            split[c] = "test" if n in test else "train"
    return kept, split
