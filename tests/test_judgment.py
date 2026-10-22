"""판단 실험(놓침 손해 비율) 테스트. 모델은 가짜로 바꿔 '손해 비율만 바꿨을 때'의 동작만 본다."""

import numpy as np
import pandas as pd
import pytest

from battery_lab import judgment
from battery_lab import screening as sc


@pytest.fixture
def fake(monkeypatch):
    rng = np.random.default_rng(0)
    life = np.r_[rng.uniform(150, 540, 40), rng.uniform(560, 2000, 80)]
    f = pd.DataFrame({"cycle_life": life, "score": np.log10(life) + rng.normal(0, 0.15, len(life))})

    def proba(rows):  # 수명이 짧을수록 단수명 확률이 높게
        return 1 / (1 + np.exp((rows["score"].to_numpy() - np.log10(550)) * 12))

    monkeypatch.setattr(sc, "fit_predict", lambda kind, tr, y, te, seed=0: proba(te))
    monkeypatch.setattr(sc, "oof_proba", lambda kind, tr, y, seed=0: proba(tr))
    return f


def test_higher_miss_cost_lowers_the_cut_and_misses_fewer(fake):
    runs = judgment.cost_runs(fake, (1, 5, 30), repeats=8)
    t = judgment.summarize(runs)

    assert list(t.index) == [1, 5, 30]
    assert t.loc[30, "기준값 중앙값"] <= t.loc[5, "기준값 중앙값"] <= t.loc[1, "기준값 중앙값"]
    assert t.loc[30, "놓친 단수명 비율 중앙값"] <= t.loc[1, "놓친 단수명 비율 중앙값"]
    assert t.loc[30, "버린 장수명 비율 중앙값"] >= t.loc[1, "버린 장수명 비율 중앙값"]


def test_same_splits_for_every_cost(fake, monkeypatch):
    # 나누기를 몇 번 했는지 센다. 손해 비율마다 새로 나누면 반복 4번 × 비율 2개 = 8번이 된다.
    # (예전 테스트는 정확도가 비었는지만 봐서 나누기를 매번 새로 해도 통과했다 — troubleshooting.md)
    tests, real = [], sc.stratified_split

    def spy(y, k_train, rng):
        train, test = real(y, k_train, rng)
        tests.append(test)
        return train, test

    monkeypatch.setattr(sc, "stratified_split", spy)
    runs = judgment.cost_runs(fake, (1, 30), repeats=4)
    assert len(tests) == 4
    assert len({tuple(t) for t in tests}) > 1        # 반복끼리는 다른 나누기
    assert (runs.groupby("repeat")["miss_cost"].nunique() == 2).all()


def test_table_marks_the_original_assumption_and_uses_percent_units(fake):
    t = judgment.summarize(judgment.cost_runs(fake, (5, 10), repeats=3))
    text = judgment.format_table(t)
    assert "5 (원래 가정)" in text
    assert "%" in text and "00.0%" not in text.replace("100.0%", "")  # % 단위를 두 번 곱하지 않았는지
