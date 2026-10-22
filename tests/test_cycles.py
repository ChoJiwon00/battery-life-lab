"""사이클 수 실험 테스트."""

import numpy as np
import pandas as pd
import pytest

from battery_lab import cycles
from battery_lab.features import cell_features, find_row, median_curve

from test_model import fake_features


def curves_with(n=120, points=20):
    cyc = np.arange(1, n + 1)
    q = np.tile(np.linspace(0, 1.0, points), (n, 1))
    return {"cycle": cyc, "Qdlin": q}


def summary(n=120):
    c = np.arange(1, n + 1)
    return pd.DataFrame({"cycle": c, "QD": 1.05 + 0.0001 * c})


def test_broken_curve_is_replaced_by_the_nearest_clean_one():
    c = curves_with()
    c["Qdlin"][69, 5] = 50.0                     # 70번째 곡선이 깨짐
    assert find_row(c, 70) == (68, 69)           # 가까운 쪽부터, 같은 거리면 앞쪽
    c["Qdlin"][68, 5] = 50.0
    assert find_row(c, 70) == (70, 71)           # 양쪽(forward=True)이면 71번째까지 본다
    assert find_row(c, 70, forward=False) == (67, 68)   # 앞쪽으로만: 71번째(미래)는 쓰지 않는다


def aging_curves(n=120, points=20):
    """사이클마다 곡선 끝쪽이 조금씩 줄어드는 가짜 곡선 (ΔQ가 0이 아니게)."""
    cyc = np.arange(1, n + 1)
    shape = np.linspace(0, 1.0, points)
    return {"cycle": cyc, "Qdlin": np.outer(1 - 0.0005 * cyc, shape) - 0.001 * np.outer(cyc, shape ** 3)}


def test_late_curve_never_uses_a_later_cycle_when_substituting():
    def broken():
        c = aging_curves()
        c["Qdlin"][99, 5] = 50.0                 # 100번째(행 99)와 99번째(행 98)가 깨짐
        c["Qdlin"][98, 5] = 50.0
        return c
    a, b = broken(), broken()
    b["Qdlin"][100:] -= 0.3                      # 101번째 이후만 다르게 바꿔도
    fa, fb = cell_features(a, summary(), late=100, early=10), cell_features(b, summary(), late=100, early=10)
    assert fa["late_used"] == fb["late_used"] == 98   # 앞쪽(98번째)을 쓰고
    assert np.isfinite(fa["logvar"]) and fa["logvar"] == fb["logvar"]  # 미래 곡선은 결과에 들어가지 않는다


def test_window_for_late_cycle_never_uses_future_cycles():
    c = curves_with()
    c["Qdlin"][100:] += 0.5                      # 101번째 이후가 바뀌어도
    a = cell_features(c, summary(), late=100, early=10, window=5)
    c2 = curves_with()
    b = cell_features(c2, summary(), late=100, early=10, window=5)
    assert a["late_used"] == b["late_used"] == 98  # 96~100의 가운데
    assert a == b or all(np.isclose(a[k], b[k], equal_nan=True) for k in a)


def test_median_curve_ignores_unclean_rows():
    c = curves_with()
    c["Qdlin"][7, 3] = np.nan
    merged, used = median_curve(c, range(8, 13))
    assert used == [9, 10, 11, 12] and np.allclose(merged, c["Qdlin"][10])


def test_capacity_features_only_use_cycles_up_to_k():
    s = summary()
    f40 = cell_features(curves_with(), s, late=40)
    assert f40["qmax_q2"] == pytest.approx(0.0001 * 40 - 0.0001 * 2)


def test_same_split_is_used_for_every_k():
    base = fake_features(n=45)
    feats = {20: base, 60: base.copy()}        # 특징이 같으면 오차도 반복마다 같아야 한다
    runs = cycles.random_splits(feats, repeats=3)
    a = runs[runs["late"] == 20].set_index(["model", "repeat"])["mape"]
    b = runs[runs["late"] == 60].set_index(["model", "repeat"])["mape"]
    assert np.allclose(a, b)


def test_grid_and_summaries():
    base = fake_features(n=45)
    feats = {20: base, 100: base.copy()}
    g = cycles.summarize_grid(cycles.grid(feats, sizes=(45, 10), repeats=4))
    assert set(g["n"]) == {45, 10} and set(g["late"]) == {20, 100}
    s = cycles.summarize_splits(cycles.random_splits(feats, repeats=3))
    assert {"median", "p10", "p90"} <= set(s.columns)


def test_shortest_enough_and_deepest_dip():
    s = pd.DataFrame({"late": [20, 40, 60, 80, 100], "model": "simple", "median": [30, 20, 15.5, 14.2, 14.0]})
    assert cycles.shortest_enough(s, "simple", margin=1.0) == 80   # 60은 15.5% > 14.0 + 1.0
    assert cycles.shortest_enough(s, "simple", margin=1.5) == 60
    curve = pd.DataFrame({"window": 1, "late": [50, 60, 70, 80, 90], "r": [-0.86, -0.90, -0.87, -0.91, -0.90]})
    dip = cycles.deepest_dip(curve)
    assert dip["late"] == 70 and dip["drop"] == pytest.approx(0.035)


def test_substitutions_lists_only_moved_cycles():
    f = fake_features(n=6).assign(late_used=40, early_used=10)
    f.loc[2, "late_used"] = 39
    out = cycles.substitutions({40: f})
    assert out[["late", "late_used"]].values.tolist() == [[40, 39]]
