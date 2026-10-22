"""ΔQ 해석(interpret) 테스트. 진짜 데이터 없이 작은 가짜 셀로만 확인한다."""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import cumulative_trapezoid

from battery_lab import interpret as it
from battery_lab.model import SimpleModel, errors

POINTS = 120
V = np.linspace(3.5, 2.0, POINTS)  # 진짜 데이터처럼 높은 전압 → 낮은 전압


def base_curve() -> np.ndarray:
    """4C 방전처럼 3.15V 근처에 큰 평탄 구간이 있는 누적 용량 곡선 Q(V) (Ah). 전압이 내려갈수록 커진다."""
    return 1.0 / (1 + np.exp((V - 3.15) / 0.02)) + 0.05 * (3.5 - V) / 1.5


def loss(depth: float) -> np.ndarray:
    """진짜 ΔQ처럼 음수이고 2.9V 근처가 가장 깊은 모양."""
    return -depth * np.exp(-((V - 2.9) / 0.2) ** 2)


def fake_curves(cycles: np.ndarray, depth: float, late: int = 100, rng=None) -> dict[str, np.ndarray]:
    q10 = base_curve()
    rows = np.tile(q10, (len(cycles), 1))
    rows[cycles == late] = q10 + loss(depth)
    if rng is not None:
        rows = rows + rng.normal(0, 2e-5, rows.shape)
    return {"cycle": cycles.astype(float), "Qdlin": rows.astype(np.float32),
            "dQdV": np.gradient(rows, V, axis=1).astype(np.float32), "Vdlin": V}


def fake_population(n=30, seed=0):
    """log10|ΔQ|가 log10(수명)과 직선으로 엮인 가짜 셀들."""
    rng = np.random.default_rng(seed)
    life = 10 ** rng.uniform(2.2, 3.3, n)
    depth = 0.5 * life ** -0.7 * 10 ** rng.normal(0, 0.03, n)
    split = np.array(["train" if i % 3 == 0 else ("test" if i % 3 == 1 else "secondary_test") for i in range(n)])
    return life, depth, split, rng


# ---------------------------------------------------------------- 곡선 찾기

def test_cell_curves_look_rows_up_by_cycle_number_not_row_index():
    cycles = np.arange(2, 122)  # 1번째 사이클이 없는 셀: 행 j = 사이클 j + 2
    c = fake_curves(cycles, depth=0.03)
    got = it.cell_curves(c)
    assert (got["late_used"], got["early_used"]) == (100, 10)
    assert np.allclose(got["dq"], c["Qdlin"][98].astype(float) - c["Qdlin"][8].astype(float))
    assert got["dq"].min() == pytest.approx(-0.03, rel=0.01)


def test_cell_curves_use_a_nearby_clean_cycle_when_the_curve_is_broken():
    cycles = np.arange(1, 121)
    c = fake_curves(cycles, depth=0.03)
    c["Qdlin"][99, 5] = np.nan  # 100번째 사이클 곡선이 깨짐
    got = it.cell_curves(c)
    assert got["late_used"] in (99, 101)
    assert it.cell_curves({**c, "cycle": c["cycle"] + 500}) is None  # 근처에 쓸 곡선이 없으면 None


# ---------------------------------------------------------------- 상관 스펙트럼

def test_spectrum_finds_the_voltage_where_the_signal_was_planted():
    rng = np.random.default_rng(1)
    life = 10 ** rng.uniform(2.2, 3.3, 40)
    dq = -np.abs(rng.normal(0.01, 0.003, (40, POINTS)))
    k = 70
    dq[:, k] = -0.5 * life ** -0.7  # 이 칸만 수명과 정확히 엮임
    spec = it.correlation_spectrum(V, dq, life)

    assert it.strongest(spec, "r_log")["index"] == k
    assert spec["r_log"].iloc[k] == pytest.approx(-1.0)
    assert spec["r_raw"].iloc[k] > 0.9  # ΔQ가 음수라 그대로 쓰면 양의 상관
    assert spec["dq_mean_mah"].iloc[k] == pytest.approx(dq[:, k].mean() * 1000)


def test_column_corr_matches_numpy_and_leaves_flat_columns_empty():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(25, 4))
    x[:, 3] = 1.0
    y = rng.normal(size=25)
    r = it.column_corr(x, y)
    assert r[0] == pytest.approx(np.corrcoef(x[:, 0], y)[0, 1])
    assert np.isnan(r[3])


def test_log_transform_does_not_turn_zero_into_minus_infinity():
    out = it.transform(np.array([[-0.01, 0.0]]), "log")
    assert out[0, 0] == pytest.approx(-2.0) and np.isnan(out[0, 1])


def test_band_grows_only_through_neighbours_above_the_threshold():
    r = np.array([0.1, 0.5, 0.96, -1.0, 0.97, 0.2, 0.99])  # 마지막 0.99는 떨어져 있어서 구간에 안 들어간다
    v = np.linspace(3.0, 2.4, 7)
    assert it.band(v, r, 0.95) == (pytest.approx(v[2]), pytest.approx(v[4]))


def test_bootstrap_is_reproducible_with_the_fixed_seed():
    life, depth, _, rng = fake_population()
    dq = np.array([loss(d) for d in depth]) + rng.normal(0, 1e-4, (len(depth), POINTS))
    a = it.boot_peak_voltage(V, it.transform(dq, "log"), life, repeats=20)
    b = it.boot_peak_voltage(V, it.transform(dq, "log"), life, repeats=20)
    assert np.array_equal(a, b)
    assert np.all((a > 2.6) & (a < 3.2))  # 신호를 넣은 2.9V 근처


# ---------------------------------------------------------------- 전압별 단일 모델

def test_single_voltage_model_is_fitted_on_training_cells_only():
    life, depth, split, _ = fake_population(seed=3)
    dq = np.array([loss(d) for d in depth])
    k = 40
    dq[:, k] = -0.5 * life ** -0.7  # 이 칸은 log|ΔQ|와 log 수명이 정확히 직선
    t = it.univariate_errors(V, dq, life, split, "log")
    best = it.pick(t, "train")
    assert best["index"] == k
    assert best["rmse_test"] == pytest.approx(0.0, abs=1e-6)

    wrong = life.copy()
    wrong[split != "train"] *= 3  # 평가 셀의 답을 바꿔도 맞춘 직선은 그대로여야 한다
    t2 = it.univariate_errors(V, dq, wrong, split, "log")
    assert t2["slope"].iloc[k] == pytest.approx(t["slope"].iloc[k])
    assert t2["rmse_test"].iloc[k] > 100


def test_pick_by_voltage_takes_the_nearest_grid_point():
    t = pd.DataFrame({"voltage": V, "rmse_train": np.arange(POINTS, dtype=float)})
    j = int(np.argmin(np.abs(V - 2.959)))
    assert it.pick(t, voltage=2.959)["index"] == j
    assert it.pick(t, "train")["index"] == 0


def test_variance_model_is_the_simple_model_on_log_variance():
    life, depth, split, rng = fake_population(seed=4)
    dq = np.array([loss(d) for d in depth]) + rng.normal(0, 1e-4, (len(depth), POINTS))
    out = it.variance_model(dq, life, split)
    X = pd.DataFrame({"logvar": np.log10(np.var(dq, axis=1))})
    train = split == "train"
    pred = SimpleModel().fit(X[train], life[train]).predict(X)
    test = split == "test"
    assert out["rmse_test"] == pytest.approx(errors(life[test], pred[test])["rmse"])
    assert out["r"] < -0.9


# ---------------------------------------------------------------- 무리·누적·봉우리

def test_life_groups_split_cells_into_thirds_by_life():
    assert it.life_groups(np.array([5, 1, 3, 2, 6, 4])).tolist() == [2, 0, 1, 0, 2, 1]
    g = it.life_groups(np.arange(10))
    assert np.bincount(g).tolist() == [4, 3, 3]


def test_group_curves_mean_and_quartiles():
    values = np.arange(12, dtype=float).reshape(4, 3)
    out = it.group_curves(values, np.array([0, 0, 1, 1]))
    assert np.allclose(out[0]["mean"], [1.5, 2.5, 3.5])
    assert np.allclose(out[1]["p25"], np.percentile(values[2:], 25, axis=0))
    assert out[1]["n"] == 2


def test_moved_adds_back_up_to_delta_q():
    """ΔQ(V) = 3.5V부터 V까지 moved를 더한 값이어야 한다 (다듬지 않을 때)."""
    dq = loss(0.04) * (1 - 0.7 / (1 + np.exp((V - 2.9) / 0.05)))  # 2.9V 아래에서 일부를 되찾는 모양
    m = it.moved(V, dq, smooth=1)
    rebuilt = -cumulative_trapezoid(m, V, initial=0) + dq[0]
    assert np.allclose(rebuilt, dq, atol=5e-4)
    assert m[np.argmin(np.abs(V - 3.0))] < 0  # 2.9V 위쪽: 100번째 사이클에서 덜 나옴


def test_depth_and_recovery():
    dq = np.interp(V, [2.0, 2.9, 3.5], [-0.01, -0.04, 0.0])
    out = it.depth_and_recovery(V, dq)
    assert out["voltage"] == pytest.approx(V[np.argmin(dq)])
    assert out["deepest_mah"] == pytest.approx(dq.min() * 1000)
    assert out["end_mah"] == pytest.approx(-10.0)
    assert out["recovered"] == pytest.approx(1 - (-0.01) / dq.min())  # 가장 깊은 곳에서 끝까지 되찾은 비율


def test_peak_metrics_ignore_the_spike_at_2V_and_measure_shift():
    v = np.linspace(3.5, 2.0, 1501)  # 1mV 간격 (3.150·3.135V가 칸 위에 오도록)
    early = -6.0 * np.exp(-((v - 3.150) / 0.02) ** 2)
    late = -5.7 * np.exp(-((v - 3.135) / 0.02) ** 2)
    early[-1] = late[-1] = -20.0  # 방전 끝 튐 (진짜 데이터의 b3 셀처럼)
    out = it.peak_metrics(v, early[None, :], late[None, :]).iloc[0]
    assert out["peak_v_early"] == pytest.approx(3.150)
    assert out["shift_mv"] == pytest.approx(-15.0, abs=1e-6)
    assert out["height_change_pct"] == pytest.approx(-5.0, abs=0.01)
    full = it.peak_metrics(v, early[None, :], late[None, :], (2.0, 3.5)).iloc[0]
    assert full["peak_v_early"] == pytest.approx(2.0)  # 범위를 안 좁히면 튐을 봉우리로 잡는다


def test_count_peaks():
    one = -6.0 * np.exp(-((V - 3.15) / 0.03) ** 2)
    two = one - 2.0 * np.exp(-((V - 2.8) / 0.03) ** 2)
    assert it.count_peaks(V, one) == 1
    assert it.count_peaks(V, two) == 2


def test_ir_change_ignores_placeholder_rows_and_zero_resistance():
    c = np.arange(1, 121, dtype=float)
    ir = np.where(c >= 50, 0.0170, 0.0160)
    s = pd.DataFrame({"cell_id": "b1c0", "cycle": c, "QD": 1.05, "IR": ir})
    s.loc[s["cycle"] == 1, ["QD", "IR"]] = 0.0     # b1의 빈 1번째 줄
    s.loc[s["cycle"] == 100, "IR"] = 0.0           # 측정 실패
    s.loc[s["cycle"] == 99, "IR"] = 0.0170 + 0.5   # 튀는 값 하나 (중앙값이 막아 준다)
    out = it.ir_change(s, ["b1c0", "b9c9"])
    assert out[0] == pytest.approx(1.0)  # mΩ
    assert np.isnan(out[1])


def test_ir_at_is_the_median_around_one_cycle_in_milliohm():
    c = np.arange(1, 21, dtype=float)
    s = pd.DataFrame({"cell_id": "b1c0", "cycle": c, "QD": 1.05, "IR": 0.016})
    s.loc[s["cycle"] == 11, "IR"] = 0.5  # 튀는 값 하나
    out = it.ir_at(s, ["b1c0", "b9c9"], 10)
    assert out[0] == pytest.approx(16.0)
    assert np.isnan(out[1])


# ---------------------------------------------------------------- 문서 전체

def write_processed(folder, n=24, seed=5):
    life, depth, _, rng = fake_population(n, seed)
    cycles = np.arange(1, 121)
    cells, summary = [], []
    (folder / "curves").mkdir(parents=True)
    for i in range(n):
        cid = f"b1c{i}"
        split = "train" if i % 3 == 0 else ("test" if i % 3 == 1 else "secondary_test")
        cells.append({"cell_id": cid, "batch": "b1", "policy": "4C(80%)-4C", "cycle_life": round(life[i]),
                      "n_cycles": 120, "split": split, "merged_from": "", "seam_cycle": ""})
        c = fake_curves(cycles, depth[i], rng=rng)
        np.savez(folder / "curves" / f"{cid}.npz", Tdlin=np.zeros_like(c["Qdlin"]), **c)
        summary.append(pd.DataFrame({"cell_id": cid, "cycle": cycles, "QD": 1.05, "QC": 1.05,
                                     "IR": 0.016 + 1e-5 * cycles * depth[i], "Tavg": 31.0, "Tmin": 30.0,
                                     "Tmax": 34.0, "chargetime": 10.0}))
    pd.DataFrame(cells).to_csv(folder / "cells.csv", index=False)
    pd.concat(summary).to_csv(folder / "summary.csv", index=False)


def test_build_writes_the_document_figures_and_tables(tmp_path):
    from battery_lab.interpret_report import build

    processed, docs = tmp_path / "processed", tmp_path / "docs"
    write_processed(processed)
    out = build(processed, docs, progress=lambda *_: None)

    md = (docs / "interpret.md").read_text(encoding="utf-8")
    for section in ("## 질문", "## 왜 중요한가", "## 방법", "## 결과", "## 읽는 법", "## 한계", "## 시행착오"):
        assert section in md
    assert "nan" not in md.lower()
    for name in ("interpret_spectrum.png", "interpret_groups.png"):
        assert (docs / "figures" / name).stat().st_size > 10_000
    for name in ("interpret_spectrum.csv", "interpret_cells.csv", "interpret_groups.csv"):
        assert (processed / name).exists()
    assert out["cells"] == 24 and out["skipped"] == []
    assert 2.6 < out["strongest_v_log"] < 3.2      # 신호를 넣은 2.9V 근처
    assert out["strongest_r_log"] < -0.9
    assert out["groups"]["dq_signal_median_mah"].is_monotonic_increasing  # 짧은 수명일수록 더 깊다
