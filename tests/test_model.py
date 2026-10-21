"""수명 예측 특징·모델·데이터 크기 실험 테스트."""

import numpy as np
import pandas as pd
import pytest

from battery_lab import sizes
from battery_lab.features import FEATURES, cell_features
from battery_lab.model import SimpleModel, baseline, errors, make


def fake_features(n=60, seed=0, noise=0.05) -> pd.DataFrame:
    """log 수명이 logvar에 직선으로 엮인 가짜 셀들."""
    rng = np.random.default_rng(seed)
    logvar = rng.uniform(-5, -2.8, n)
    life = 10 ** (-0.5 * logvar + 0.95 + rng.normal(0, noise, n))
    df = pd.DataFrame({"cell_id": [f"b1c{i}" for i in range(n)], "cycle_life": life, "logvar": logvar})
    for k, name in enumerate(FEATURES[1:]):
        df[name] = logvar * (0.3 + 0.1 * k) + rng.normal(0, 0.1, n)
    df["split"] = ["train" if i % 3 == 0 else ("test" if i % 3 == 1 else "secondary_test") for i in range(n)]
    return df


def test_features_from_delta_q_and_early_capacity():
    cycles = np.arange(1, 121)
    q = np.tile(np.linspace(0, 1.0, 50), (120, 1))
    q[99] -= np.linspace(0, 0.02, 50) ** 2 * 10       # 100번째 사이클만 용량이 조금 줄어듦 (진짜 ΔQ처럼 음수)
    summary = pd.DataFrame({"cycle": cycles, "QD": 1.05 + 0.0001 * np.minimum(cycles, 60)})

    f = cell_features({"cycle": cycles, "Qdlin": q}, summary)

    d = q[99] - q[9]
    assert f["logvar"] == pytest.approx(np.log10(np.var(d)))
    assert f["logmin"] == pytest.approx(np.log10(abs(d.min())))
    assert f["q2"] == pytest.approx(1.05 + 0.0002)
    assert f["qmax_q2"] == pytest.approx(0.0001 * 60 - 0.0002)
    assert all(np.isfinite(v) for v in f.values())


def test_flat_delta_q_gives_empty_features_not_infinity():
    cycles = np.arange(1, 121)
    q = np.tile(np.linspace(0, 1.0, 50), (120, 1))   # 100번째와 10번째 곡선이 똑같음 → 분산 0
    f = cell_features({"cycle": cycles, "Qdlin": q}, pd.DataFrame({"cycle": cycles, "QD": 1.05}))
    assert np.isnan(f["logvar"]) and np.isnan(f["logmin"])


def test_simple_model_recovers_an_exact_line():
    df = fake_features(noise=0.0)
    model = SimpleModel().fit(df, df["cycle_life"].to_numpy())
    assert (model.slope, model.intercept) == (pytest.approx(-0.5), pytest.approx(0.95))
    assert np.allclose(model.predict(df), df["cycle_life"])


def test_errors_are_percent_and_cycles():
    e = errors(np.array([100, 200]), np.array([110, 180]))
    assert e["mape"] == pytest.approx(10.0)
    assert e["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))


def test_baseline_reports_both_models_on_all_three_splits():
    out = baseline(fake_features())
    assert set(zip(out["model"], out["split"])) == {(m, s) for m in ("simple", "full")
                                                     for s in ("train", "test", "secondary_test")}
    assert (out["mape"] < 20).all()


def test_full_model_fits_with_few_cells():
    df = fake_features(n=8)
    pred = make("full").fit(df, df["cycle_life"].to_numpy()).predict(df)
    assert pred.shape == (8,) and np.all(pred > 0)


def test_size_experiment_skips_full_model_when_too_few_training_cells():
    df = fake_features()
    pred = sizes.prediction_runs(df, sizes=(60, 10), repeats=3)

    assert set(pred.loc[pred["n"] == 10, "model"]) == {"simple"}   # 학습 3개 → 6개 특징 모델 없음
    assert set(pred.loc[pred["n"] == 60, "model"]) == {"simple", "full"}
    big = pred[pred["n"] == 60]
    assert (big["n_train"] == 20).all() and (big["n_test"] == 40).all()


def test_with_all_cells_reported_error_equals_actual_error():
    df = fake_features()
    pred = sizes.prediction_runs(df, sizes=(60,), repeats=2)
    assert np.allclose(pred["reported_mape"], pred["actual_mape"])


def test_correlation_runs_and_summary():
    df = fake_features()
    corr = sizes.correlation_runs(df, sizes=(60, 10), repeats=50)
    pred = sizes.prediction_runs(df, sizes=(60, 10), repeats=3)
    s = sizes.summarize(corr, pred)

    assert s["corr"][60]["runs"] == 1 and s["corr"][10]["runs"] == 50
    assert s["corr"][60]["median"] < -0.9
    assert (10, "full") not in s["pred"] and (10, "simple") in s["pred"]


def test_experiment_is_reproducible_with_the_same_seed():
    df = fake_features()
    a = sizes.prediction_runs(df, sizes=(30,), repeats=3)
    b = sizes.prediction_runs(df, sizes=(30,), repeats=3)
    pd.testing.assert_frame_equal(a, b)
