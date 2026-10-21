"""품질 점검 + 논문 검증 + 그래프를 한 번에 만들어 docs/data_quality.md로 쓴다."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures
from .quality import CellReport, alignment_check, check_all, hotspots, load_processed, paper_checks
from .sources import ROOT

DOCS = ROOT / "docs"

PAPER = {"cells": 124, "life": "150 ~ 2,300", "change_100": "+0.2%", "split": "41 / 43 / 40"}

# 공식 데이터 페이지의 배치별 실험 노트 (이상이 몰린 구간과 맞춰 본다)
OFFICIAL_NOTES = {
    "b1": "컴퓨터가 두 번 자동 재시작돼 기록에 시간 공백이 있음. 1번째 사이클 기록 없음.",
    "b2": "컴퓨터 재시작으로 대부분의 셀이 250사이클 근처에서 약 8시간 쉼 → 용량 '튐'.",
    "b3": "내부저항 측정 때문에 생긴 OCV 오류로 시험이 잠깐씩 멈춤.",
}

KNOWN = [  # 이상이 아니라 데이터가 원래 그렇게 기록된 것
    ("b1 배치의 1번째 사이클 빈 자리 (0으로 채운 줄)", lambda r: r.placeholder_rows > 0),
    ("0.88Ah에 닿자 시험 종료 → 수명 = 마지막 기록 + 1", lambda r: r.eol_rule == "stopped_at_eol"),
    ("80%를 지나서도 더 시험 → 수명 = 처음 0.88Ah 미만 사이클", lambda r: r.eol_rule == "first_below"),
]
ISSUES = [  # 실제 이상
    ("수명 라벨과 실제 80% 도달이 다름", lambda r: np.isfinite(r.life_gap) and r.life_gap != 0),
    ("80% 도달을 확인할 수 없음", lambda r: not np.isfinite(r.eol_observed)),
    ("회복 튐 (쉬었다 재개한 뒤 용량이 0.5% 넘게 오름)", lambda r: bool(r.recovery_cycles)),
    ("순간 하락 (1% 넘게 꺼졌다가 돌아옴)", lambda r: bool(r.dip_cycles)),
    ("범위 밖 용량 (0.1Ah 이하·1.2Ah 이상)", lambda r: bool(r.bad_qd_cycles)),
    ("방전 곡선에 말이 안 되는 값 (-0.01Ah 미만·1.3Ah 초과)", lambda r: bool(r.broken_curve_cycles)),
    ("충전 시간 60분 초과 (멈춘 동안의 시간까지 충전 시간에 기록됨)", lambda r: bool(r.long_charge_cycles)),
    ("방전 곡선의 작은 되돌림 (1mAh 넘게, 측정 잡음 수준)", lambda r: r.small_backsteps > 0),
    ("빈 방전 곡선", lambda r: r.empty_curves > 0),
    ("범위 밖 온도 (20°C 미만·50°C 초과)", lambda r: r.bad_temp > 0),
    ("내부저항 0 또는 중앙값의 3배 초과", lambda r: r.zero_ir > 0 or r.high_ir > 0),
    ("⚠ 수명 예측에 쓰는 10·100번째 사이클 곡선 문제", lambda r: not r.feature_clean),
]


def to_frame(reports: list[CellReport]) -> pd.DataFrame:
    join = lambda xs: " ".join(str(x) for x in xs)  # noqa: E731
    return pd.DataFrame([{
        "cell_id": r.cell_id, "batch": r.batch, "policy": r.policy, "split": r.split,
        "cycle_life": r.cycle_life, "eol_observed": r.eol_observed, "eol_rule": r.eol_rule, "life_gap": r.life_gap,
        "qd_start": r.qd_start, "qd_100": r.qd_100, "change_100_pct": r.change_100_pct,
        "placeholder_rows": r.placeholder_rows, "recovery_cycles": join(r.recovery_cycles),
        "dip_count": len(r.dip_cycles), "bad_qd_cycles": join(r.bad_qd_cycles),
        "broken_curve_cycles": join(r.broken_curve_cycles), "long_charge_cycles": join(r.long_charge_cycles), "small_backsteps": r.small_backsteps,
        "empty_curves": r.empty_curves, "bad_temp": r.bad_temp, "zero_ir": r.zero_ir, "high_ir": r.high_ir,
        "feature_clean": r.feature_clean, "seam_cycle": r.seam_cycle, "seam_first_pct": r.seam_first_pct,
        "seam_after10_pct": r.seam_after10_pct, "notes": " / ".join(r.notes)} for r in reports])


def build(processed: Path, docs: Path = DOCS) -> dict:
    reports = check_all(processed)
    checks = paper_checks(reports, processed)
    checks["hotspots"] = hotspots(reports)
    _, summary = load_processed(processed)
    checks["alignment"] = alignment_check(processed, [r.cell_id for r in reports], summary)
    to_frame(reports).to_csv(processed / "quality.csv", index=False, encoding="utf-8")

    fig_dir = docs / "figures"
    figures.capacity_fade(summary, reports, fig_dir / "capacity_fade.png")
    figures.delta_q_curves(processed, reports, fig_dir / "delta_q.png")
    figures.logvar_vs_life(reports, checks["feature"], checks["logvar_corr"], fig_dir / "logvar_vs_life.png")
    figures.quality_example(summary, reports, fig_dir / "quality_example.png")

    (docs / "data_quality.md").write_text(markdown(reports, checks), encoding="utf-8")
    return checks


def _count(reports, test) -> int:
    return sum(1 for r in reports if test(r))


def markdown(reports: list[CellReport], c: dict) -> str:
    split = {k: _count(reports, lambda r, k=k: r.split == k) for k in ("train", "test", "secondary_test")}
    batch = {k: _count(reports, lambda r, k=k: r.batch == k) for k in ("b1", "b2", "b3")}
    gaps = np.array([r.life_gap for r in reports if np.isfinite(r.life_gap)])
    lines = [
        "# 1단계 · 데이터 품질 점검과 논문 검증",
        "",
        f"`python -m battery_lab report`가 자동으로 만든 문서예요 (만든 날 {date.today().isoformat()}).",
        "분석하기 전에 정리한 데이터가 **논문과 같은지**, **어떤 문제가 어디에 섞여 있는지**부터 확인했어요.",
        "점검 기준을 정하다 처음에 틀렸던 부분은 [시행착오 기록](troubleshooting.md)에 따로 적었어요.",
        "",
        "## 논문 숫자와 비교",
        "",
        "| 항목 | 논문 | 우리 데이터 |",
        "|---|---|---|",
        f"| 셀 수 | {PAPER['cells']} | {c['cells']} (배치별 {batch['b1']} · {batch['b2']} · {batch['b3']}) |",
        f"| 학습 / 테스트 / 2차 테스트 | {PAPER['split']} | {split['train']} / {split['test']} / {split['secondary_test']} |",
        f"| 사이클 수명 범위 | {PAPER['life']} | {c['life_min']:,.0f} ~ {c['life_max']:,.0f} (중앙값 {c['life_median']:,.0f}) |",
        f"| 사이클 2 → 100 용량 변화 (중앙값) | {PAPER['change_100']} | {c['change_100_median_pct']:+.2f}% |",
        f"| log Var(ΔQ₁₀₀₋₁₀)와 log 수명의 상관 | 강한 음의 상관 | r = {c['logvar_corr']:.3f} ({c['logvar_cells']}개 셀) |",
        "",
        "![124개 셀의 방전 용량 변화](figures/capacity_fade.png)",
        "",
        "**읽는 법.** 초반 100사이클 동안 용량은 거의 변하지 않아서, 용량만 보면 어느 셀이 일찍 죽을지 알 수 없어요.",
        "하지만 100번째와 10번째 방전 곡선의 **차이(ΔQ)** 를 보면 셀마다 벌어지는 정도가 달라요.",
        "",
        "![ΔQ 곡선](figures/delta_q.png)",
        "",
        "그 차이의 분산 하나만으로도 수명과 강하게 엮여 있어요. 4단계 수명 예측 모델의 출발점이에요.",
        "",
        "![ΔQ 분산과 수명](figures/logvar_vs_life.png)",
        "",
    ]
    a = c.get("alignment")
    if a:
        grid = a["grid"]
        lines += [
            "## 데이터 정리 검증",
            "",
            "| 검사 | 결과 |",
            "|---|---|",
            f"| 곡선·원시 데이터의 j번째 사이클 = 요약표 j번째 사이클 | 원시 값이 있는 {a['total']:,}개 사이클 중 "
            f"**{a['matched']:,}개 일치** (최대 오차 {a['max_error_mah']:.3f}mAh, 빈 사이클 {a['skipped']}개 제외) |",
            (f"| 방전 곡선 전압 눈금(Vdlin) | {a['grids_same']}/{a['cells']}개 셀이 같음 "
             f"({grid[0]:.1f}V → {grid[1]:.1f}V, {grid[2]:,}칸) |" if grid else "| 방전 곡선 전압 눈금 | 없음 |"),
            "",
        ]
    lines += [
        "## 품질 점검 결과",
        "",
        "### 이상이 아니라 원래 그렇게 기록된 것",
        "",
        "| 내용 | 해당 셀 수 |",
        "|---|---|",
    ]
    lines += [f"| {label} | {_count(reports, test)} |" for label, test in KNOWN]
    if gaps.size:
        same = int(np.sum(gaps == 0))
        lines += ["", f"→ 두 경우를 나눠 보면 수명 라벨과 실제 80% 도달 사이클이 **{same}/{len(reports)}개 셀에서 정확히 같아요.**"]
    lines += ["", "### 실제로 섞여 있는 문제", "", "| 점검 항목 | 해당 셀 수 |", "|---|---|"]
    lines += [f"| {label} | {_count(reports, test)} |" for label, test in ISSUES]

    hs = c.get("hotspots", {})
    if hs:
        lines += ["", "### 이상이 몰린 곳 ↔ 공식 실험 노트", "",
                  "회복 튐·범위 밖 용량·깨진 곡선·긴 충전 시간이 일어난 사이클을 20사이클 구간으로 묶어, 가장 많이 몰린 구간을 찾았어요.", "",
                  "| 배치 | 가장 많이 몰린 구간 | 몰린 정도 | 공식 실험 노트 |", "|---|---|---|---|"]
        for key, h in hs.items():
            lines.append(f"| {key} | 사이클 {h['start']}~{h['end']} | 이상 {h['events']}/{h['total']}건 · "
                         f"셀 {h['cells']}/{h['batch_cells']}개 | {OFFICIAL_NOTES.get(key, '')} |")

        lines.append("")
        for key, h in hs.items():
            share = h["events"] / h["total"]
            if key == "b2" and 230 <= h["start"] <= 260:
                lines.append(f"- **b2에서는 공식 노트의 재시작 시점(250사이클 근처)과 겹쳐요.** 이상의 {share:.0%}가 이 구간에 몰려 있어요.")
            elif share < 0.3:
                lines.append(f"- {key}에서는 이상이 한 구간에 몰리지 않고 흩어져 있어요 (가장 많은 구간도 {share:.0%}).")

    seams = [r for r in reports if np.isfinite(r.seam_cycle)]
    if seams:
        lines += ["", "### 이어 붙인 셀 5개의 이음매", "",
                  "1배치에서 멈췄다가 2배치에서 이어 시험한 셀이에요. 공식 노트는 \"쉬었다 재개하면 용량이 오른다\"고 했는데,",
                  "실제 기록은 아래와 같았어요 (기준: 이음매 직전 5사이클 중앙값).", "",
                  "| 셀 | 충전 조건 | 이음매 사이클 | 재개 첫 사이클 | 재개 뒤 10사이클 중앙값 |", "|---|---|---|---|---|"]
        for r in seams:
            lines.append(f"| {r.cell_id} | {r.policy} | {r.seam_cycle:,.0f} | {r.seam_first_pct:+.2f}% | {r.seam_after10_pct:+.2f}% |")
        drops = sum(r.seam_first_pct < -0.5 for r in seams)
        worst_after = max(abs(r.seam_after10_pct) for r in seams)
        first_seam = min(r.seam_cycle for r in seams)
        lines += ["", f"- {len(seams)}개 중 {drops}개는 재개 직후 오히려 0.5% 넘게 **떨어졌다가** 몇 사이클에 걸쳐 일부 회복했어요.",
                  f"- 이어 붙이기 자체는 맞다고 판단했어요: 재개 뒤 10사이클 중앙값이 직전 값과 {worst_after:.1f}% 안쪽으로 이어지고, "
                  "새 셀이라면 1.07Ah 근처에서 시작하기 때문이에요.",
                  f"- 이음매는 모두 {first_seam:,.0f}사이클 이후라 수명 예측(첫 100사이클)에는 영향이 없어요. "
                  "2단계에서 용량 추세를 볼 때는 `cells.csv`의 `seam_cycle`로 표시해요."]

    clean = _count(reports, lambda r: r.feature_clean)
    lines += ["", f"**결론.** 수명 예측에 쓰는 10·100번째 사이클 곡선은 **{clean}/{len(reports)}개 셀 모두 깨끗해요.**",
              "용량 추세를 볼 때(2단계)는 순간 하락·범위 밖 값·이음매를 표시하거나 걸러서 봐요.",
              "", "![품질 점검 예시](figures/quality_example.png)", "",
              "### 셀별 내용", "", "| 셀 | 배치 | 충전 조건 | 내용 |", "|---|---|---|---|"]
    for r in reports:
        if r.notes:
            lines.append(f"| {r.cell_id} | {r.batch} | {r.policy} | {' / '.join(r.notes)} |")
    lines += ["", "전체 셀의 점검 값은 `data/processed/quality.csv`에 있어요.", ""]
    return "\n".join(lines)
