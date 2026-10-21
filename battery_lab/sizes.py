"""데이터 크기 실험: 셀이 124개가 아니라 70·35·10개뿐이었다면?

124개 셀에서 N개를 무작위로 뽑아 '그 N개만 있었던 연구'를 여러 번 흉내 낸다.

실험 1 · 상관관계 발견: 뽑은 N개 안에서 log Var(ΔQ₁₀₀₋₁₀)와 log 수명의 상관 r을 다시 계산한다.
실험 2 · 수명 예측: 뽑은 N개를 논문처럼 1/3은 학습, 2/3는 평가에 쓴다.
  - 보고된 오차: 뽑힌 N개 안의 평가 셀로 잰 오차 (그 연구가 논문에 썼을 숫자)
  - 실제 오차: 학습에 안 쓴 나머지 모든 셀(124 − 학습 수)로 잰 오차 (그 모델의 진짜 실력에 가까운 숫자)

한계: 같은 124개 셀에서 다시 뽑는 것이라, 정말로 다른 셀 N개로 새 실험을 한 것과 같지는 않다.
"""

import numpy as np
import pandas as pd

from .features import FEATURES
from .model import MIN_TRAIN, errors, make

SIZES = (124, 70, 35, 10)
TRAIN_SHARE = 41 / 124      # 논문: 학습 41 · 평가 83
REPEATS = 300
CORR_REPEATS = 2000
SEED = 2026


def n_train(n: int) -> int:
    return max(1, round(n * TRAIN_SHARE))


def correlation_runs(features: pd.DataFrame, sizes=SIZES, repeats=CORR_REPEATS, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = features["logvar"].to_numpy()
    y = np.log10(features["cycle_life"].to_numpy())
    rows = []
    for n in sizes:
        reps = 1 if n >= len(features) else repeats
        for k in range(reps):
            idx = rng.choice(len(features), size=n, replace=False)
            rows.append({"n": n, "repeat": k, "r": float(np.corrcoef(x[idx], y[idx])[0, 1])})
    return pd.DataFrame(rows)


def prediction_runs(features: pd.DataFrame, sizes=SIZES, repeats=REPEATS, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    life = features["cycle_life"].to_numpy()
    rows = []
    for n in sizes:
        k_train = n_train(n)
        for k in range(repeats):
            pick = rng.choice(len(features), size=n, replace=False)
            train, test = pick[:k_train], pick[k_train:]
            outside = np.setdiff1d(np.arange(len(features)), train)
            for kind in ("simple", "full"):
                if k_train < MIN_TRAIN[kind]:
                    continue
                model = make(kind, seed=k).fit(features.iloc[train], life[train])
                reported = errors(life[test], model.predict(features.iloc[test]))
                actual = errors(life[outside], model.predict(features.iloc[outside]))
                rows.append({"n": n, "repeat": k, "model": kind, "n_train": k_train, "n_test": len(test),
                             "reported_mape": reported["mape"], "actual_mape": actual["mape"],
                             "reported_rmse": reported["rmse"], "actual_rmse": actual["rmse"]})
    return pd.DataFrame(rows)


def _q(s: pd.Series, p: float) -> float:
    return float(np.percentile(s, p))


def summarize(corr: pd.DataFrame, pred: pd.DataFrame) -> dict:
    c = {}
    for n, g in corr.groupby("n"):
        c[int(n)] = {"median": float(g["r"].median()), "p5": _q(g["r"], 5), "p95": _q(g["r"], 95),
                     "weaker_08": float((g["r"] > -0.8).mean()), "weaker_05": float((g["r"] > -0.5).mean()),
                     "runs": len(g)}
    p = {}
    for (n, kind), g in pred.groupby(["n", "model"]):
        gap = g["actual_mape"] - g["reported_mape"]
        p[(int(n), kind)] = {
            "n_train": int(g["n_train"].iloc[0]), "n_test": int(g["n_test"].iloc[0]), "runs": len(g),
            "actual_median": float(g["actual_mape"].median()), "actual_p10": _q(g["actual_mape"], 10),
            "actual_p90": _q(g["actual_mape"], 90),
            "reported_median": float(g["reported_mape"].median()), "reported_p10": _q(g["reported_mape"], 10),
            "reported_p90": _q(g["reported_mape"], 90),
            "flattered": float((gap >= 5).mean()),           # 보고된 오차가 실제보다 5%p 넘게 좋게 나온 비율
            "good": float((g["actual_mape"] <= 15).mean()),  # 실제 오차 15% 이하 (논문 단순 모델 수준)
            "bad": float((g["actual_mape"] > 25).mean()),    # 실제 오차 25% 초과
        }
    return {"corr": c, "pred": p}


def run(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    missing = features[list(FEATURES)].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"특징이 비어 있는 셀: {', '.join(features.loc[missing, 'cell_id'])}")
    corr, pred = correlation_runs(features), prediction_runs(features)
    return corr, pred, summarize(corr, pred)
