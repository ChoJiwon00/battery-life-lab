"""급속충전 조건과 수명 분석 테스트 (작은 가짜 데이터만 쓴다)."""

import numpy as np
import pandas as pd
import pytest

from battery_lab import policies as pol
from battery_lab import policies_report as report

NOMINAL = 1.1


def raw_cycle(parts, dt=0.01, nominal=NOMINAL, volt=3.4):
    """[(분, C-rate), ...]로 원시 기록 한 사이클을 만든다. q는 그 시각까지 넣은 용량, i는 그 시각부터 흐르는 전류."""
    t, i, q = [], [], []
    now, charge = 0.0, 0.0
    for minutes, c in parts:
        for _ in range(int(round(minutes / dt))):
            t.append(now)
            i.append(c)
            q.append(charge)
            now += dt
            charge += max(c, 0.0) * dt / 60 * nominal
    t.append(now)
    i.append(0.0)
    q.append(charge)
    t, i, q = np.array(t), np.array(i), np.array(q)
    return t, np.full(len(t), volt), i, q


def policy_parts(c1, q1, c2, rest=5.0):
    """표기대로 0→80% 충전 → 쉬기 → 1C로 95%까지 → 4C 방전 조금."""
    first = min(q1, 80) / 100 / c1 * 60
    second = max(0.0, 80 - q1) / 100 / c2 * 60
    return [(first, c1), (second, c2), (rest, 0.0), (15 / 100 / 1 * 60, 1.0), (1.0, -4.0)]


# ---------------------------------------------------------------- 표기 읽기

@pytest.mark.parametrize("text, expect", [
    ("5.4C(40%)-3.6C", {"c1": 5.4, "q1": 40, "c2": 3.6, "newstructure": False, "unit_missing": False,
                        "single_step": False}),
    ("4C(31%)-5", {"c1": 4, "q1": 31, "c2": 5, "newstructure": False, "unit_missing": True, "single_step": False}),
    ("5.3C(54%)-4C-newstructure", {"c1": 5.3, "q1": 54, "c2": 4, "newstructure": True, "unit_missing": False,
                                   "single_step": False}),
    ("4.8C(80%)-4.8C", {"c1": 4.8, "q1": 80, "c2": 4.8, "newstructure": False, "unit_missing": False,
                        "single_step": True}),
])
def test_parse_policy_reads_every_format_in_the_data(text, expect):
    assert pol.parse_policy(text) == expect


@pytest.mark.parametrize("text", ["", "6C-3C", "abc", "0C(40%)-3C", "6C(0%)-3C", "6C(40%)-3C-oldstructure"])
def test_parse_policy_returns_none_for_unreadable(text):
    assert pol.parse_policy(text) is None


def test_base_policy_drops_newstructure():
    assert pol.base_policy("4.8C(80%)-4.8C-newstructure") == "4.8C(80%)-4.8C"
    assert pol.base_policy("6C(40%)-3C") == "6C(40%)-3C"


# ---------------------------------------------------------------- 이상적인 시간과 평균 전류

def test_window_minutes_matches_hand_calculation():
    assert pol.window_minutes(1, 4, 6) == pytest.approx(10.0)          # 4%/1C = 2.4분, 76%/6C = 7.6분
    assert pol.window_minutes(6, 40, 3) == pytest.approx(12.0)         # 4분 + 8분
    assert pol.window_minutes(3.6, 80, 3.6) == pytest.approx(80 / 3.6 * 0.6)
    assert pol.window_minutes(6, 40, 3, 40, 80) == pytest.approx(8.0)  # C1 구간과 겹치지 않음


def test_window_rate_is_time_average_over_the_window():
    assert pol.window_rate(6, 40, 3, 0, 40) == pytest.approx(6.0)
    assert pol.window_rate(6, 40, 3, 40, 80) == pytest.approx(3.0)
    assert pol.window_rate(1, 4, 6, 0, 40) == pytest.approx(0.4 / (0.04 / 1 + 0.36 / 6))


# ---------------------------------------------------------------- 원시 기록에서 실제 충전 재기

def test_charge_profile_recovers_currents_and_times():
    t, v, i, q = raw_cycle(policy_parts(6, 40, 3))
    p = pol.charge_profile(t, v, i, q, q1=40)
    assert p["t40"] == pytest.approx(4.0, abs=0.02)
    assert p["t80"] == pytest.approx(4.0 + (pol.END_SOC - 40) / 100 / 3 * 60, abs=0.02)
    assert p["low_rate"] == pytest.approx(6.0, rel=0.01)
    assert p["high_rate"] == pytest.approx(3.0, rel=0.01)
    assert p["c1_real"] == pytest.approx(6.0) and p["c2_real"] == pytest.approx(3.0)


def test_rest_after_fast_charge_is_not_counted_when_it_stops_just_below_80():
    """급속충전이 80%에 살짝 못 미쳐 끝나도(79.95%) 쉬는 시간을 급속충전 시간에 넣지 않는다 (시행착오 4)."""
    parts = [(4.0, 6.0), (7.99, 3.0), (5.0, 0.0), (6.0, 1.0)]  # 40% + 39.95% = 79.95%, 그다음 5분 쉼
    t, v, i, q = raw_cycle(parts)
    p = pol.charge_profile(t, v, i, q, q1=40)
    assert p["t80"] < 12.5                      # 80.00%로 잡으면 쉬고 난 뒤인 약 17분이 나온다
    assert p["high_rate"] == pytest.approx(3.0, rel=0.01)


def test_single_step_policy_has_no_second_current():
    t, v, i, q = raw_cycle(policy_parts(4.8, 80, 4.8))
    p = pol.charge_profile(t, v, i, q, q1=80)
    assert np.isnan(p["c2_real"])
    assert p["low_rate"] == pytest.approx(4.8, rel=0.01) and p["high_rate"] == pytest.approx(4.8, rel=0.01)


# ---------------------------------------------------------------- 표기 vs 측정

def test_time_check_names_the_reason():
    rows = [
        # 정상
        dict(cell_id="a", c1=6, q1=40, c2=3, c1_real=6.0, c2_real=3.0, v_hit=False, ideal_min=12.0, measured_min=12.1),
        # 표기보다 약한 전류 (4.8C 표기인데 4.36C)
        dict(cell_id="b", c1=4.8, q1=80, c2=4.8, c1_real=4.36, c2_real=np.nan, v_hit=False, ideal_min=10.0,
             measured_min=11.0),
        # 높은 SOC에서 3.6V 상한에 닿아 전류가 줄어듦
        dict(cell_id="c", c1=1, q1=4, c2=6, c1_real=1.0, c2_real=5.5, v_hit=True, ideal_min=10.0, measured_min=10.4),
        # 표기보다 센 전류
        dict(cell_id="d", c1=4, q1=31, c2=5, c1_real=4.0, c2_real=5.5, v_hit=True, ideal_min=10.53, measured_min=10.2),
    ]
    table = pd.DataFrame(rows).assign(batch="b2", policy="p", t80=lambda d: d["measured_min"])
    check = pol.time_check(table).set_index("cell_id")
    assert not check.loc["a", "flagged"]
    assert check.loc["b", "reason"] == "표기보다 약한 전류"
    assert check.loc["c", "reason"].startswith("3.6V")
    assert check.loc["d", "reason"] == "표기보다 센 전류"
    assert check.loc["b", "real_ideal_min"] == pytest.approx(0.8 / 4.36 * 60)


# ---------------------------------------------------------------- 수명과의 관계

def test_spearman_uses_ranks():
    assert pol.spearman([1, 2, 3, 4], [1, 10, 100, 1000]) == pytest.approx(1.0)
    assert pol.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(pol.spearman([1, 1, 1], [1, 2, 3]))


def fake_table(seed=0, noise=0.01, per_group=2, groups=8):
    """log10 수명 = 3.2 − 0.12·낮은 SOC 전류 − 0.18·높은 SOC 전류 + 배치 출발점 + 잡음."""
    rng = np.random.default_rng(seed)
    offset = {"b1": 0.0, "b2": -0.2, "b3": 0.1}
    rows = []
    for b, off in offset.items():
        for g in range(groups):
            low, high = rng.uniform(3, 7), rng.uniform(3, 6)
            for k in range(per_group):
                log_life = 3.2 - 0.12 * low - 0.18 * high + off + rng.normal(0, noise)
                rows.append({"cell_id": f"{b}c{g}_{k}", "batch": b, "group": f"{b} · p{g}", "low_rate": low,
                             "high_rate": high, "shift": high - low, "log_life": log_life,
                             "cycle_life": 10 ** log_life, "measured_min": 24 * (1 / low + 1 / high)})
    return pd.DataFrame(rows)


def test_effects_recover_slopes_with_batch_offsets():
    fit = pol.effects(fake_table(), ("low_rate", "high_rate"), boot=200)
    assert fit["terms"]["low_rate"]["coef"] == pytest.approx(-0.12, abs=0.01)
    assert fit["terms"]["high_rate"]["coef"] == pytest.approx(-0.18, abs=0.01)
    assert fit["terms"]["high_rate"]["pct"] == pytest.approx((10 ** -0.18 - 1) * 100, abs=2)
    assert fit["second_worse_share"] > 0.9   # 높은 SOC 쪽 기울기가 더 음수


def test_cluster_bootstrap_works_when_batches_are_mixed():
    """여러 배치를 섞어도 다시 뽑기가 실제로 쓰여야 한다 (시행착오 3: 줄 번호 버그로 0번이었음)."""
    fit = pol.effects(fake_table(), ("low_rate", "high_rate"), boot=200)
    assert fit["boot_used"] > 150
    assert fit["terms"]["low_rate"]["lo"] < fit["terms"]["low_rate"]["coef"] < fit["terms"]["low_rate"]["hi"]


def test_effects_are_reproducible_with_the_same_seed():
    a = pol.effects(fake_table(), ("low_rate", "high_rate"), boot=100, seed=7)
    b = pol.effects(fake_table(), ("low_rate", "high_rate"), boot=100, seed=7)
    assert a == b


def test_leave_out_drops_the_shortest_cells_of_one_batch():
    """민감도 점검: b2에서 수명이 가장 짧은 2셀만 빼고, 다른 배치 셀은 그대로 둔다."""
    table = fake_table().assign(policy="p")
    shortest = table[table["batch"] == "b2"].nsmallest(2, "cycle_life")["cell_id"].tolist()
    out = pol.leave_out_shortest(table, "b2", 2, boot=100)
    assert out["cells"] == shortest
    assert out["pooled"]["cells"] == len(table) - 2
    assert out["within"]["cells"] == (table["batch"] == "b2").sum() - 2
    assert pol.leave_out_shortest(table, "b9", 2, boot=10) == {}


def test_eta2_and_permutation_baseline():
    y, labels = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]), np.array(list("aabbcc"))
    assert pol.eta2(y, labels) == pytest.approx(1.0)
    share = pol.variance_share(y, labels, perm=200)
    assert share["within_sd"] == pytest.approx(0.0)
    assert share["null_mean"] < share["eta2"]


def test_replicates_only_use_conditions_with_two_or_more_cells():
    table = fake_table(per_group=2)
    single = table.iloc[[0]].assign(group="b1 · 혼자", cell_id="lonely")
    rep = pol.replicates(pd.concat([table, single]).assign(policy="p"), perm=50)
    assert "b1 · 혼자" not in set(rep["groups"]["group"])
    assert rep["all"]["groups"] == table["group"].nunique()


def test_frontier_picks_best_median_per_bin_and_running_best():
    groups = pd.DataFrame({
        "group": ["g1", "g2", "g3", "g4"], "batch": ["b1", "b1", "b2", "b1"], "policy": ["p1", "p2", "p3", "p4"],
        "cells": [2, 1, 3, 2], "minutes": [9.0, 10.0, 10.2, 12.0], "life_median": [500, 800, 900, 700],
        "life_max": [520, 800, 1000, 750]})
    front = pol.frontier(groups, bins=(8.5, 9.5, 10.5, 11.5, 12.5)).set_index("lo")
    assert front.loc[9.5, "best_group"] == "g3" and front.loc[9.5, "best_cell_life"] == 1000
    assert front.loc[10.5, "groups"] == 0
    assert front.loc[11.5, "best_life"] == 700 and front.loc[11.5, "so_far"] == 900   # 더 빠른 쪽에서 본 최고


# ---------------------------------------------------------------- 전체 흐름 (가짜 processed 폴더)

FAKE = {  # 셀: (배치, 표기, 실제 C1, 실제 C2, 수명)
    "b1c0": ("b1", "6C(40%)-3C", 6, 3, 900), "b1c1": ("b1", "6C(40%)-3C", 6, 3, 950),
    "b1c2": ("b1", "4C(80%)-4C", 4, 4, 1500), "b1c3": ("b1", "4C(80%)-4C", 4, 4, 1650),
    "b1c4": ("b1", "8C(15%)-3.6C", 8, 3.6, 980), "b1c5": ("b1", "5.4C(80%)-5.4C", 5.4, 5.4, 540),
    "b1c6": ("b1", "5.4C(80%)-5.4C", 5.4, 5.4, 560),
    "b2c0": ("b2", "4.8C(80%)-4.8C", 4.8, 4.8, 470), "b2c1": ("b2", "4.8C(80%)-4.8C", 4.8, 4.8, 490),
    "b2c2": ("b2", "1C(4%)-6C", 1, 6, 300), "b2c3": ("b2", "6C(60%)-3C", 6, 3, 700),
    "b2c4": ("b2", "4C(31%)-5", 4, 5.5, 460), "b2c5": ("b2", "3.6C(30%)-6C", 3.6, 6, 510),
    "b3c0": ("b3", "5C(67%)-4C-newstructure", 5, 4, 900), "b3c1": ("b3", "5C(67%)-4C-newstructure", 5, 4, 1300),
    "b3c2": ("b3", "4.8C(80%)-4.8C-newstructure", 4.36, 4.36, 1700),
    "b3c3": ("b3", "4.8C(80%)-4.8C-newstructure", 4.36, 4.36, 1800),
    "b3c4": ("b3", "3.7C(31%)-5.9C-newstructure", 3.7, 5.9, 600),
    "b3c5": ("b3", "3.7C(31%)-5.9C-newstructure", 3.7, 5.9, 700),
}


def write_fake_processed(folder):
    (folder / "raw").mkdir(parents=True)
    cells, summary = [], []
    for cid, (batch, policy, r1, r2, life) in FAKE.items():
        p = pol.parse_policy(policy)
        cells.append({"cell_id": cid, "batch": batch, "policy": policy, "cycle_life": life, "n_cycles": 120,
                      "split": "train", "merged_from": "", "seam_cycle": ""})
        minutes = pol.window_minutes(r1, p["q1"], r2) + 0.1
        for cyc in range(1, 121):
            placeholder = batch == "b1" and cyc == 1
            summary.append({"cell_id": cid, "cycle": cyc, "QD": 0.0 if placeholder else 1.07, "QC": 0.0 if placeholder
                            else 1.07, "IR": 0.0 if placeholder else 0.016, "Tavg": 0.0 if placeholder else 31.0,
                            "Tmin": 30.0, "Tmax": 34.0, "chargetime": 0.0 if placeholder else minutes})
        pieces = [raw_cycle(policy_parts(r1, p["q1"], r2)) for _ in range(3)]
        offsets = np.cumsum([0] + [len(x[0]) for x in pieces])
        stack = [np.concatenate([x[k] for x in pieces]) for k in range(4)]
        np.savez(folder / "raw" / f"{cid}.npz", raw_index=np.array([0, 9, 99]), raw_offsets=offsets,
                 raw_t=stack[0], raw_V=stack[1], raw_I=stack[2], raw_Qc=stack[3],
                 raw_Qd=np.zeros(len(stack[0])), raw_T=np.full(len(stack[0]), 31.0))
    pd.DataFrame(cells).to_csv(folder / "cells.csv", index=False)
    pd.DataFrame(summary).to_csv(folder / "summary.csv", index=False)
    return folder


def test_cell_table_on_fake_folder(tmp_path):
    folder = write_fake_processed(tmp_path / "processed")
    table, unparsed = pol.cell_table(folder)
    assert unparsed == [] and len(table) == len(FAKE)
    t = table.set_index("cell_id")
    assert t.loc["b3c2", "c1_real"] == pytest.approx(4.36, rel=0.01)       # 표기 4.8C지만 실제 4.36C
    assert t.loc["b2c4", "unit_missing"] and t.loc["b2c4", "c2_real"] == pytest.approx(5.5, rel=0.01)
    assert t.loc["b1c0", "measured_min"] == pytest.approx(12.1)            # b1 1번째 사이클 빈 자리는 빠짐
    assert t.loc["b1c0", "raw_cycles"] == 2                                 # 줄 0은 범위(사이클 2~100) 밖
    reasons = set(pol.time_check(table).query("flagged")["reason"])
    assert {"표기보다 약한 전류", "표기보다 센 전류"} <= reasons


def test_build_writes_doc_figures_and_csvs(tmp_path):
    folder = write_fake_processed(tmp_path / "processed")
    docs = tmp_path / "docs"
    out = report.build(processed=folder, docs=docs, progress=lambda *_: None)

    assert out["cells"] == len(FAKE) and out["unparsed"] == []
    md = (docs / "policies.md").read_text(encoding="utf-8")
    for header in ("## 질문", "## 왜 중요한가", "## 방법", "## 결과", "## 읽는 법", "## 한계", "## 시행착오"):
        assert header in md
    assert "nan" not in md.lower()
    for name in ("time_check", "map", "effects", "replicates", "frontier"):
        assert (docs / "figures" / f"policies_{name}.png").stat().st_size > 0
    for name in ("cells", "groups", "frontier", "effects"):
        assert (folder / f"policies_{name}.csv").exists()
