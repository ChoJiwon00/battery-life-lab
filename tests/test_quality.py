"""데이터 품질 점검·논문 검증·리포트 테스트."""

import numpy as np
import pandas as pd
import pytest

from battery_lab.assemble import assemble
from battery_lab.extract import extract_batch
from battery_lab.quality import check_cell, delta_q, find_eol, hotspots, log_var_delta_q
from battery_lab.report import build
from battery_lab.sources import Rules

META = {"cell_id": "b1c0", "batch": "b1", "policy": "6C(40%)-3C", "split": "train", "cycle_life": "273"}


def summary(n=300) -> pd.DataFrame:
    """용량이 사이클마다 0.0007Ah씩 고르게 줄어드는 깨끗한 셀."""
    c = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({"cycle": c, "QD": 1.07 - 0.0007 * (c - 1), "QC": 1.07, "IR": 0.016,
                         "Tavg": 31.0, "Tmin": 30.0, "Tmax": 34.0, "chargetime": 10.0})


def test_clean_cell_matches_its_life_label():
    r = check_cell(META, summary(), None)

    assert (r.eol_observed, r.eol_rule) == (273, "first_below")  # 1.07 − 0.0007·(c−1) < 0.88 이 처음 되는 사이클
    assert r.change_100_pct == pytest.approx((-0.0007 * 98) / (1.07 - 0.0007) * 100)
    assert r.notes == []


def test_test_stopped_at_80_percent_means_life_is_last_cycle_plus_one():
    s = summary(272)  # 마지막 기록 0.8803Ah — 0.88 미만인 사이클은 기록에 없다
    r = check_cell(META, s, None)
    assert (r.eol_observed, r.eol_rule, r.life_gap) == (273, "stopped_at_eol", 0)


def test_eol_is_not_fooled_by_a_dip_below_80_percent():
    qd = np.array([0.95, 0.94, 0.85, 0.94, 0.93, 0.92, 0.90, 0.879, 0.878, 0.877, 0.876])
    assert find_eol(qd, np.arange(1, 12)) == (8, "first_below")


def test_b1_placeholder_row_is_counted_separately_not_as_errors():
    s = summary()
    s.loc[s["cycle"] == 1, ["QD", "QC", "IR", "Tavg", "Tmin", "Tmax", "chargetime"]] = 0.0
    q = np.tile(np.linspace(0, 1, 10), (300, 1))
    q[0] = np.nan
    r = check_cell(META, s, {"Qdlin": q, "cycle": np.arange(1, 301)})

    assert r.placeholder_rows == 1
    assert (r.bad_qd_cycles, r.bad_temp, r.zero_ir, r.empty_curves) == ([], 0, 0, 0)


def test_recovery_jump_and_dip_are_told_apart():
    s = summary()
    s.loc[s["cycle"] >= 150, "QD"] += 0.01                # 쉬었다 재개: 한 번 오르고 유지
    s.loc[s["cycle"].between(50, 52), "QD"] -= 0.03       # 측정 이상: 3사이클 꺼졌다 돌아옴
    s.loc[s["cycle"] == 60, "QD"] = 1.5                   # 범위 밖

    r = check_cell(META, s, None)

    assert r.recovery_cycles == [150]
    assert r.dip_cycles == [50]          # 3사이클이어도 한 번
    assert r.bad_qd_cycles == [60]


def test_frequent_dips_do_not_make_normal_values_look_like_jumps():
    s = summary()
    for start in range(40, 120, 3):          # 잦은 순간 하락 (b1c0의 400~850사이클처럼)
        s.loc[s["cycle"].between(start, start + 1), "QD"] -= 0.012
    for start in range(150, 230, 2):         # 1%보다 얕은 하락이 연달아
        s.loc[s["cycle"] == start, "QD"] -= 0.007
    s.loc[s["cycle"].between(250, 257), "QD"] -= 0.008  # 8사이클 처졌다가 원래 수준으로 (b1c0의 409사이클처럼)
    assert check_cell(META, s, None).recovery_cycles == []


def test_seam_change_is_measured_against_the_cycles_before_it():
    s = summary()
    s.loc[s["cycle"] >= 200, "QD"] -= 0.013   # 재개 직후 떨어짐
    r = check_cell({**META, "seam_cycle": "200"}, s, None)
    base = 1.07 - 0.0007 * 196                # 직전 5사이클(195~199)의 중앙값 = 197번째
    first = 1.07 - 0.0007 * 199 - 0.013
    assert r.seam_cycle == 200
    assert r.seam_first_pct == pytest.approx((first / base - 1) * 100)


def test_curve_checks_split_broken_values_from_small_noise():
    q = np.tile(np.linspace(0, 1, 10), (120, 1))
    q[20, 5] = 500.0                       # 말이 안 되는 값
    q[30, 5] = q[30, 4] - 0.002            # 2mAh 되돌림 (잡음 수준)
    r = check_cell(META, summary(120), {"Qdlin": q, "cycle": np.arange(1, 121)})

    assert r.broken_curve_cycles == [21]
    assert r.small_backsteps == 1
    assert r.feature_clean  # 10·100번째 곡선은 멀쩡하다

    q[99, 3] = -5.0
    assert not check_cell(META, summary(120), {"Qdlin": q, "cycle": np.arange(1, 121)}).feature_clean


def test_long_charge_time_is_flagged():
    s = summary()
    s.loc[s["cycle"] == 248, "chargetime"] = 500.0   # 8시간 쉰 시간이 충전 시간에 들어감
    s.loc[s["cycle"] == 249, "chargetime"] = 22.0    # 늙으면서 길어진 정전압 충전은 정상
    assert check_cell(META, s, None).long_charge_cycles == [248]


def test_hotspots_find_where_problems_cluster():
    a = check_cell(META, summary(), None)
    a.recovery_cycles, a.broken_curve_cycles = [248], [251]
    b = check_cell({**META, "cell_id": "b1c1"}, summary(), None)
    b.bad_qd_cycles = [253]

    h = hotspots([a, b])["b1"]

    assert (h["start"], h["end"], h["events"], h["cells"]) == (240, 259, 3, 2)


def test_delta_q_looks_up_rows_by_cycle_number():
    cycles = np.arange(2, 121)  # 1번째 사이클이 없는 배치처럼
    q = np.outer(cycles, np.ones(5)).astype(np.float32)
    curves = {"cycle": cycles, "Qdlin": q}

    assert np.allclose(delta_q(curves), 90)  # 사이클 100 − 사이클 10
    assert np.isnan(log_var_delta_q({"cycle": np.arange(1, 50), "Qdlin": q[:49]}))


def test_report_builds_markdown_and_figures(fake_mat, tmp_path):
    interim = tmp_path / "interim"
    mat = fake_mat("b1.mat", [{"life": life, "policy": f"p{i}", "n": 110} for i, life in enumerate((150, 400, 900, 2000))])
    extract_batch(mat, "b1", interim, progress=lambda _: None)
    assemble(interim, tmp_path / "processed", Rules(removed=(), carryover=()))

    checks = build(tmp_path / "processed", tmp_path / "docs")

    assert checks["cells"] == 4
    assert checks["logvar_corr"] < -0.9  # 빨리 늙는 셀일수록 ΔQ 분산이 크다
    a = checks["alignment"]
    assert a["total"] > 0 and a["matched"] == a["total"]  # 원시 최대 용량 = 같은 행의 요약 용량
    text = (tmp_path / "docs" / "data_quality.md").read_text(encoding="utf-8")
    assert "## 논문 숫자와 비교" in text and "| 셀 수 | 124 | 4" in text
    for name in ("capacity_fade", "delta_q", "logvar_vs_life", "quality_example"):
        assert (tmp_path / "docs" / "figures" / f"{name}.png").stat().st_size > 10_000
    assert (tmp_path / "processed" / "quality.csv").exists()
