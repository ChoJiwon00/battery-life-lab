"""예측 범위(불확실성) 테스트. 진짜 데이터 없이 작은 가짜 셀로 빠르게 돈다."""

import numpy as np
import pandas as pd
import pytest

from battery_lab import uncertainty as unc
from battery_lab.features import FEATURES


def fake_cells(n=90, seed=0, noise=0.06, b3_share=1 / 3) -> pd.DataFrame:
    """log 수명이 logvar에 직선으로 엮이고, 셀마다 log 오차(noise)가 붙은 가짜 셀들. 진짜처럼 logvar는 음수."""
    rng = np.random.default_rng(seed)
    logvar = rng.uniform(-5, -2.8, n)
    life = 10 ** (-0.5 * logvar + 0.95 + rng.normal(0, noise, n))
    n_b3 = int(n * b3_share)
    batch = np.array(["b1"] * ((n - n_b3) // 2) + ["b2"] * (n - n_b3 - (n - n_b3) // 2) + ["b3"] * n_b3)
    df = pd.DataFrame({"cell_id": [f"{b}c{i}" for i, b in enumerate(batch)], "batch": batch,
                       "policy": np.where(batch == "b3", "4C(80%)-4C-newstructure", "4C(80%)-4C"),
                       "cycle_life": life, "logvar": logvar})
    for k, name in enumerate(FEATURES[1:]):
        df[name] = logvar * (0.3 + 0.1 * k) + rng.normal(0, 0.1, n)
    order = np.arange(n)
    df["split"] = np.where(batch == "b3", "secondary_test", np.where(order % 2 == 0, "test", "train"))
    return df


# ---------------------------------------------------------------- 분할 conformal의 순번

def test_conformal_quantile_picks_the_right_rank():
    scores = np.arange(1, 21, dtype=float)[::-1]          # 20개, 순서가 섞여 있어도 정렬해서 쓴다
    assert unc.conformal_quantile(scores, 0.9) == 19        # ⌈21 × 0.9⌉ = 19번째
    assert unc.conformal_quantile(scores, 0.5) == 11        # ⌈21 × 0.5⌉ = 11번째
    assert unc.conformal_quantile(scores, 0.95) == 20       # ⌈21 × 0.95⌉ = 20번째 (가장 큰 값)


def test_conformal_quantile_is_not_pushed_by_float_residue():
    # 20 × 0.95는 소수 계산으로 19.000000000000004가 될 수 있다 → 그대로 올리면 20번째로 잘못 밀린다
    scores = np.arange(1, 20, dtype=float)                 # 19개
    assert unc.conformal_quantile(scores, 0.95) == 19


def test_conformal_quantile_is_infinite_when_too_few_scores():
    assert np.isinf(unc.conformal_quantile(np.arange(1, 6, dtype=float), 0.9))   # ⌈6 × 0.9⌉ = 6 > 5


# ---------------------------------------------------------------- 범위 만들기

def test_with_residuals_adds_each_models_own_oob_errors():
    preds = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    resid = [np.array([0.1]), np.array([-1.0, 2.0])]
    s = unc.with_residuals(preds, resid)
    assert s.shape == (3, 3)
    assert np.allclose(s, [[1.1, 2.1, 3.1], [9.0, 19.0, 29.0], [12.0, 22.0, 32.0]])


def test_bootstrap_oob_residuals_come_only_from_cells_left_out():
    df = fake_cells(n=30, noise=0.0)                       # 잡음이 없으면 OOB 오차도 0이어야 한다
    train = df.iloc[:20]
    preds, resid = unc.bootstrap(train, df.iloc[20:], "simple", n_boot=15, seed=1)
    assert preds.shape == (15, 10)
    assert all(len(r) < 20 for r in resid) and sum(len(r) for r in resid) > 0
    assert np.allclose(np.concatenate(resid), 0, atol=1e-9)


def test_bounds_are_nested_by_level_and_line_only_is_narrowest():
    df = fake_cells(n=120, seed=3)
    train, evaluate = df.iloc[:60], df.iloc[60:]
    b = unc.bounds(train, evaluate, "simple", levels=(50, 80, 90), n_boot=200, seed=0)
    for method, (center, lo, hi) in b.items():
        assert lo.shape == hi.shape == (3, len(evaluate)) and center.shape == (len(evaluate),)
        assert np.all(lo[0] >= lo[1] - 1e-12) and np.all(lo[1] >= lo[2] - 1e-12)   # 90%가 50%를 감싼다
        assert np.all(hi[0] <= hi[1] + 1e-12) and np.all(hi[1] <= hi[2] + 1e-12)
    width = {m: float(np.mean(hi[2] - lo[2])) for m, (_, lo, hi) in b.items()}
    assert width["boot_line"] < width["boot_resid"] / 2
    assert width["boot_line"] < width["conformal"] / 2


def test_honest_methods_cover_about_the_promised_share_and_line_only_does_not():
    # 평가 셀만 많으면 안 된다: 학습 셀이 200개면 학습 셀에 실제로 붙은 잡음 크기가 우연히 5~8% 작게 나와
    # 정직한 범위도 85%만 맞을 수 있었다 (시행착오). 그래서 학습 1,000 · 평가 2,000개로 둘 다 크게 잡는다.
    df = fake_cells(n=3000, seed=5, b3_share=0)
    train, evaluate = df.iloc[:1000], df.iloc[1000:]
    cells = unc.cell_rows(evaluate, "simple", unc.bounds(train, evaluate, "simple", (90,), n_boot=200, seed=0), (90,))
    cov = cells.groupby("method")["covered"].mean()
    assert 0.86 <= cov["boot_resid"] <= 0.94
    assert 0.86 <= cov["conformal"] <= 0.94
    assert cov["boot_line"] < 0.5


def test_conformal_needs_enough_cells_for_the_model():
    df = fake_cells(n=10)
    with pytest.raises(ValueError):
        unc.conformal(df.iloc[:8], df.iloc[8:], "full")    # 모델용 4개 < 6개 특징 모델 최소 6개


# ---------------------------------------------------------------- 평가

def test_cell_rows_marks_misses_on_each_side():
    ev = pd.DataFrame({"cell_id": ["a", "b", "c"], "batch": "b1", "split": "test", "cycle_life": [100.0, 500.0, 2000.0]})
    lo, hi = np.log10([[200.0, 200.0, 200.0]]), np.log10([[1000.0, 1000.0, 1000.0]])
    rows = unc.cell_rows(ev, "simple", {"conformal": (np.log10([400.0, 400.0, 400.0]), lo, hi)}, levels=(90,))
    assert rows["below"].tolist() == [True, False, False]   # 100 < 200: 범위보다 짧음
    assert rows["above"].tolist() == [False, False, True]   # 2000 > 1000: 범위보다 김
    assert rows["covered"].tolist() == [False, True, False]
    assert np.allclose(rows["width"], 800.0) and np.allclose(rows["pred"], 400.0)
    cov = unc.coverage(rows, ["method", "level"]).iloc[0]
    assert cov["coverage"] == pytest.approx(1 / 3) and cov["rel_half"] == pytest.approx(1.0)


def test_chance_band_contains_the_promise_and_narrows_with_more_cells():
    lo40, hi40 = unc.chance_band(40, 0.9)
    lo400, hi400 = unc.chance_band(400, 0.9)
    assert lo40 < 0.9 < hi40 and lo400 < 0.9 < hi400
    assert (hi400 - lo400) < (hi40 - lo40)


def test_conformal_band_is_wider_than_the_binomial_band():
    # 보정 셀 20개면 범위마다 진짜 적중 확률이 Beta(19, 2)로 흔들리므로, 평가 셀만의 우연(이항분포)보다 넓다
    lo_b, hi_b = unc.chance_band(83, 0.9)
    lo_c, hi_c = unc.conformal_band(83, 20, 0.9)
    assert lo_c < lo_b and hi_c >= hi_b
    assert lo_c < 0.9 < hi_c
    assert unc.conformal_band(40, 5, 0.9) == (1.0, 1.0)     # ⌈6 × 0.9⌉ = 6 > 5: 범위가 무한대라 항상 맞음


def test_conformal_band_matches_simulated_split_conformal():
    # 가짜 데이터로 분할 conformal을 여러 번 돌려 본 적중률의 10~90%가 이론 폭과 비슷한지
    rng = np.random.default_rng(0)
    cover = []
    for _ in range(2000):
        cal, ev = np.abs(rng.normal(size=20)), np.abs(rng.normal(size=83))
        cover.append(np.mean(ev <= unc.conformal_quantile(cal, 0.9)))
    lo, hi = unc.conformal_band(83, 20, 0.9, 0.1, 0.9)
    assert np.percentile(cover, 10) == pytest.approx(lo, abs=0.03)
    assert np.percentile(cover, 90) == pytest.approx(hi, abs=0.03)


def test_narrower_share_pairs_the_two_models_within_each_repeat():
    rows = []
    for rep, (w_simple, w_full) in enumerate([(100, 80), (100, 120), (100, 90), (100, 99)]):
        for model, w in (("simple", w_simple), ("full", w_full)):
            rows.append({"scheme": "mixed", "method": "conformal", "level": 90, "group": "all",
                         "repeat": rep, "model": model, "width_mean": w})
    assert unc.narrower_share(pd.DataFrame(rows), 90) == pytest.approx(0.75)


def test_paper_split_rows_cover_every_eval_cell_model_method_and_level():
    df = fake_cells(n=60)
    paper = unc.paper_split(df, levels=(50, 90), boot={"simple": 30, "full": 4}, seed=0)
    n_eval = int((df["split"] != "train").sum())
    assert len(paper) == 2 * len(unc.METHODS) * 2 * n_eval
    assert set(paper["split"]) == {"test", "secondary_test"}


def test_random_splits_keep_b3_out_of_training_in_the_b12_scheme():
    df = fake_cells(n=90)
    runs = unc.random_splits(df, repeats=2, n_boot=20, levels=(90,), methods={"simple": unc.METHODS})
    n_b3 = int((df["batch"] == "b3").sum())
    b12_b3 = runs[(runs["scheme"] == "b12") & (runs["group"] == "b3")]
    assert (b12_b3["cells"] == n_b3).all()                  # b3는 모두 평가에 있다
    mixed_b3 = runs[(runs["scheme"] == "mixed") & (runs["group"] == "b3")]
    assert (mixed_b3["cells"] < n_b3).any()                  # 섞어 나누면 일부는 학습에 들어간다
    all_cells = runs[runs["group"] == "all"]["cells"]
    assert (all_cells == len(df) - unc.n_train(len(df))).all()

    s = unc.summarize_splits(runs)
    assert {"mid", "p10", "p90", "short_by_10"} <= set(s.columns)
    assert s["repeats"].eq(2).all()


def test_random_splits_are_reproducible_with_the_same_seed():
    df = fake_cells(n=60)
    kw = dict(repeats=2, n_boot=10, levels=(90,), methods={"simple": ("boot_resid", "conformal")})
    pd.testing.assert_frame_equal(unc.random_splits(df, **kw), unc.random_splits(df, **kw))


def test_shift_reports_bias_spread_and_typical_error_per_split():
    df = fake_cells(n=60)
    sh = unc.shift(df).set_index("split")
    assert list(sh.index) == ["train", "test", "secondary_test"]
    assert sh.loc["secondary_test", "batches"] == "b3"
    assert sh.loc["train", "bias_simple"] == pytest.approx(0, abs=1e-9)   # 직선은 학습 셀의 log 오차 평균을 0으로 맞춘다
    for col in ("spread_simple", "abs_median_simple", "spread_full", "abs_median_full",
                "mape_simple", "rmse_simple", "mape_full", "rmse_full"):
        assert (sh[col] > 0).all()


def test_feature_shift_measures_distance_in_training_spread_units():
    df = fake_cells(n=60)
    df.loc[df["split"] == "secondary_test", "q2"] += 10.0    # 2차 테스트의 한 특징만 크게 옮긴다
    fs = unc.feature_shift(df).set_index(["feature", "split"])
    assert fs.loc[("q2", "secondary_test"), "z"] > 5
    assert fs.loc[("q2", "secondary_test"), "outside"] == 1.0
    assert abs(fs.loc[("logvar", "secondary_test"), "z"]) < 1


def test_policy_overlap_ignores_the_newstructure_suffix():
    assert unc.policy_overlap(fake_cells(n=60)) == (1, 1)
