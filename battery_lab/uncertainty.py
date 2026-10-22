"""예측에 오차 범위 붙이기 (불확실성).

"이 셀은 812사이클"보다 "90% 확률로 700~950사이클"이 실무에 더 쓸모 있다.
수명 예측(log10 수명)에 범위를 붙이는 방법 세 가지를 만들고, 그 범위가 정직한지(보정, calibration) 확인한다.

- 부트스트랩 · 직선만: 학습 셀을 중복 허용으로 다시 뽑아 모델을 B번 만들고, 예측값이 흔들리는 폭만 본다.
  모델(직선)이 얼마나 흔들리는지만 담고, 특징이 같아도 셀마다 수명이 다른 '셀 사이 편차'는 담지 않는다.
- 부트스트랩 · +잔차: 위 예측값마다, 그 모델이 학습에 쓰지 않은 셀(OOB, out-of-bag)에서 낸 오차를 더해 범위를 만든다.
- 분할 conformal: 학습 셀을 반으로 나눠 반으로 모델을 만들고, 나머지 반(보정 셀)에서 난 오차 크기로 폭을 정한다.
  오차가 어떤 분포든 '평균적으로' 약속한 비율을 지킨다. 단, 새 셀이 보정 셀과 같은 주머니에서 나왔다고 볼 수 있을 때만이다
  (교환 가능성, exchangeability).

정직함(보정) = "90% 범위"라고 했으면 실제로 셀의 90% 안팎이 그 범위 안에 들어오는가.
범위는 모두 log10 수명에서 만들고, 사이클로 바꿀 때 10의 거듭제곱을 한다 (그래서 사이클로는 위쪽이 더 길다).
"""

import numpy as np
import pandas as pd
from scipy.stats import betabinom, binom

from .features import FEATURES
from .model import MIN_TRAIN, errors, make
from .sizes import SEED, n_train

METHODS = ("boot_line", "boot_resid", "conformal")
METHOD_NAMES = {"boot_line": "부트스트랩 (직선만)", "boot_resid": "부트스트랩 + 잔차", "conformal": "분할 conformal"}
LEVELS = (50, 80, 90)                       # 표에 쓰는 범위 (%)
CURVE_LEVELS = tuple(range(10, 100, 5))     # 보정 그래프용 10·15·…·95%
BOOT = {"simple": 1000, "full": 200}        # 논문 나누기에서 다시 만드는 모델 수 (6개 특징 모델은 한 번에 약 0.25초)
CAL_SHARE = 0.5                             # 분할 conformal: 학습 셀 중 보정에 쓰는 비율
SPLIT_REPEATS = 100
SPLIT_BOOT = 300
# 무작위 나누기 반복에서는 시간 때문에 6개 특징 모델의 부트스트랩을 뺀다 (100번 × 300개 모델 ≈ 2시간)
REPEAT_METHODS = {"simple": METHODS, "full": ("conformal",)}
SCHEMES = {"mixed": "섞어 나누기 (세 배치를 섞어 41개 학습)", "b12": "b1·b2에서만 학습 (논문처럼 b3는 학습에 없음)"}


def check(features: pd.DataFrame) -> None:
    missing = features[list(FEATURES)].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"특징이 비어 있는 셀: {', '.join(features.loc[missing, 'cell_id'])}")


def _log_pred(model, cells: pd.DataFrame) -> np.ndarray:
    return np.log10(model.predict(cells))


# ---------------------------------------------------------------- 범위를 만드는 방법

def bootstrap(train: pd.DataFrame, evaluate: pd.DataFrame, kind: str, n_boot: int,
              seed: int = SEED) -> tuple[np.ndarray, list[np.ndarray]]:
    """학습 셀을 중복 허용으로 n_boot번 다시 뽑아 모델을 만든다.

    돌려주는 값: (평가 셀의 log10 예측, 모양 n_boot × 평가 셀 수),
                 (모델마다 OOB 잔차 = log10 실제 − log10 예측. 그 모델이 학습에 쓰지 않은 셀에서만 잰다)
    """
    rng = np.random.default_rng(seed)
    life = train["cycle_life"].to_numpy()
    n = len(train)
    preds = np.empty((n_boot, len(evaluate)))
    resid = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = make(kind, seed=b).fit(train.iloc[idx], life[idx])
        preds[b] = _log_pred(model, evaluate)
        out = np.setdiff1d(np.arange(n), idx)
        resid.append(np.log10(life[out]) - _log_pred(model, train.iloc[out]) if len(out) else np.empty(0))
    return preds, resid


def with_residuals(preds: np.ndarray, resid: list[np.ndarray]) -> np.ndarray:
    """부트스트랩 예측마다 그 모델의 OOB 잔차를 하나씩 더한 표본. 모양: (전체 OOB 잔차 수 × 평가 셀 수)."""
    lens = [len(r) for r in resid]
    if not sum(lens):
        raise ValueError("OOB 잔차가 하나도 없어요 (학습 셀이 너무 적음)")
    return np.repeat(preds, lens, axis=0) + np.concatenate(resid)[:, None]


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """보정 셀 n개의 |오차| 중 ⌈(n+1)·수준⌉번째로 작은 값. 그 순번이 n보다 크면(보정 셀이 너무 적음) 무한대."""
    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    k = int(np.ceil((n + 1) * level - 1e-9))  # 21 × 0.95 = 19.95처럼 소수 계산 찌꺼기로 한 칸 밀리지 않게
    return float(np.inf) if k > n else float(scores[max(k, 1) - 1])


def conformal(train: pd.DataFrame, evaluate: pd.DataFrame, kind: str, seed: int = SEED,
              cal_share: float = CAL_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """학습 셀을 무작위로 '모델용'과 '보정용'으로 나눈다.

    돌려주는 값: (모델용 셀로 만든 모델의 평가 셀 log10 예측, 보정 셀의 |log10 오차|)
    """
    rng = np.random.default_rng(seed)
    life = train["cycle_life"].to_numpy()
    order = rng.permutation(len(train))
    n_cal = int(len(train) * cal_share)
    cal, fit = order[:n_cal], order[n_cal:]
    if len(fit) < MIN_TRAIN[kind] or n_cal < 1:
        raise ValueError(f"분할 conformal에 쓸 셀이 모자라요 (모델용 {len(fit)}개, 보정용 {n_cal}개)")
    model = make(kind, seed).fit(train.iloc[fit], life[fit])
    scores = np.abs(np.log10(life[cal]) - _log_pred(model, train.iloc[cal]))
    return _log_pred(model, evaluate), scores


def bounds(train: pd.DataFrame, evaluate: pd.DataFrame, kind: str, levels=LEVELS, n_boot: int = 300,
           seed: int = SEED, methods=METHODS) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """방법마다 (가운데 예측, 아래 끝, 위 끝). 모두 log10 사이클.

    가운데 예측의 모양은 (평가 셀 수,), 끝은 (수준 수, 평가 셀 수). levels는 % 단위 (예: 90).
    부트스트랩 두 방법의 가운데 예측은 학습 셀 전체로 만든 모델, 분할 conformal은 모델용 셀로 만든 모델이다.
    """
    p = np.asarray(levels, float) / 100
    lo_q, hi_q = (1 - p) / 2, (1 + p) / 2
    out = {}
    if {"boot_line", "boot_resid"} & set(methods):
        point = _log_pred(make(kind, seed).fit(train, train["cycle_life"].to_numpy()), evaluate)
        preds, resid = bootstrap(train, evaluate, kind, n_boot, seed)
        if "boot_line" in methods:
            out["boot_line"] = (point, np.quantile(preds, lo_q, axis=0), np.quantile(preds, hi_q, axis=0))
        if "boot_resid" in methods:
            s = with_residuals(preds, resid)
            out["boot_resid"] = (point, np.quantile(s, lo_q, axis=0), np.quantile(s, hi_q, axis=0))
    if "conformal" in methods:
        center, scores = conformal(train, evaluate, kind, seed)
        q = np.array([conformal_quantile(scores, x) for x in p])[:, None]
        out["conformal"] = (center, center - q, center + q)
    return out


# ---------------------------------------------------------------- 평가

def cell_rows(evaluate: pd.DataFrame, kind: str, b: dict, levels=LEVELS) -> pd.DataFrame:
    """셀 하나 × 방법 × 수준마다 한 줄: 예측, 범위(사이클), 실제 수명이 범위 안인지."""
    parts = []
    for method, (center, lo, hi) in b.items():
        for i, level in enumerate(levels):
            parts.append(pd.DataFrame({
                "model": kind, "method": method, "level": int(level),
                "cell_id": evaluate["cell_id"].to_numpy(), "batch": evaluate["batch"].to_numpy(),
                "split": evaluate["split"].to_numpy(), "life": evaluate["cycle_life"].to_numpy(),
                "pred": 10 ** center, "lo": 10 ** lo[i], "hi": 10 ** hi[i]}))
    df = pd.concat(parts, ignore_index=True)
    df["below"] = df["life"] < df["lo"]    # 실제 수명이 범위보다 짧음 (보증 관점에서 위험한 쪽)
    df["above"] = df["life"] > df["hi"]    # 실제 수명이 범위보다 김
    df["covered"] = ~(df["below"] | df["above"])
    df["width"] = df["hi"] - df["lo"]
    return df


def coverage(cells: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """묶음마다 적중률(범위 안에 든 셀 비율), 범위 폭, 어느 쪽으로 빗나갔는지."""
    g = cells.groupby(by, sort=False)
    return g.agg(cells=("covered", "size"), coverage=("covered", "mean"), below=("below", "mean"),
                 above=("above", "mean"), width_mean=("width", "mean"), width_median=("width", "median"),
                 rel_half=("width", lambda w: float(np.median(w / (2 * cells.loc[w.index, "pred"])))),
                 ).reset_index()


def paper_split(features: pd.DataFrame, levels=CURVE_LEVELS, boot: dict = BOOT, seed: int = SEED,
                progress=lambda msg: None) -> pd.DataFrame:
    """논문 나누기: 학습 41개로 범위를 만들고 테스트 43 · 2차 테스트 40개에서 확인한다. 셀마다 한 줄."""
    train = features[features["split"] == "train"]
    evaluate = features[features["split"] != "train"]
    parts = []
    for kind in ("simple", "full"):
        progress(f"  논문 나누기 · {kind}: 부트스트랩 {boot[kind]}번")
        parts.append(cell_rows(evaluate, kind, bounds(train, evaluate, kind, levels, boot[kind], seed), levels))
    return pd.concat(parts, ignore_index=True)


def random_splits(features: pd.DataFrame, repeats: int = SPLIT_REPEATS, n_boot: int = SPLIT_BOOT, levels=LEVELS,
                  seed: int = SEED, schemes=tuple(SCHEMES), methods: dict = REPEAT_METHODS,
                  progress=lambda msg: None) -> pd.DataFrame:
    """학습 41개를 무작위로 여러 번 다시 뽑아 적중률이 얼마나 흔들리는지 본다.

    - mixed: 124개 전체에서 41개를 뽑는다 (b3 셀도 일부 학습에 들어감)
    - b12:   b1·b2 셀(84개)에서만 41개를 뽑는다 (논문처럼 b3는 전부 평가)
    평가는 학습에 안 쓴 나머지 모든 셀. 묶음(group)은 all(전체) · b12 · b3로 따로 센다.
    """
    k_train = n_train(len(features))
    batch = features["batch"].to_numpy()
    rows = []
    for s_i, scheme in enumerate(schemes):
        rng = np.random.default_rng([seed, s_i])
        pool = np.arange(len(features)) if scheme == "mixed" else np.flatnonzero(batch != "b3")
        progress(f"  무작위 나누기 · {scheme}: {repeats}번")
        for rep in range(repeats):
            tr = rng.choice(pool, size=k_train, replace=False)
            ev = np.setdiff1d(np.arange(len(features)), tr)
            rep_seed = int(rng.integers(2**31))
            train, evaluate = features.iloc[tr], features.iloc[ev]
            for kind, use in methods.items():
                cells = cell_rows(evaluate, kind, bounds(train, evaluate, kind, levels, n_boot, rep_seed, use), levels)
                cells["group"] = np.where(cells["batch"] == "b3", "b3", "b12")
                both = pd.concat([cells.assign(group="all"), cells], ignore_index=True)
                cov = coverage(both, ["model", "method", "level", "group"])
                rows.append(cov.assign(scheme=scheme, repeat=rep))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 요약

def chance_band(n: int, level: float, lo: float = 0.05, hi: float = 0.95) -> tuple[float, float]:
    """정직한 범위(진짜 적중 확률 = level)라도 셀 n개에서 우연히 나오는 적중률의 5~95% 범위 (이항분포).

    평가 셀끼리 독립이라고 보고 계산한 폭이다. 실제로는 모든 평가 셀이 같은 학습·보정 셀로 만든 범위를
    함께 쓰므로 적중률은 이보다 더 흔들린다 (분할 conformal은 conformal_band가 그 폭).
    """
    return float(binom.ppf(lo, n, level) / n), float(binom.ppf(hi, n, level) / n)


def conformal_band(n: int, n_cal: int, level: float, lo: float = 0.05, hi: float = 0.95) -> tuple[float, float]:
    """분할 conformal(보정 셀 n_cal개)이 셀 n개에서 낼 수 있는 적중률의 lo~hi 범위 (베타-이항분포).

    보정 셀이 정해지면 그 범위의 진짜 적중 확률은 Beta(k, n_cal + 1 − k)를 따르고(k = ⌈(n_cal+1)·level⌉),
    평가 셀 n개의 적중 수는 그 확률로 던진 이항분포다 (Angelopoulos · Bates, arXiv:2107.07511, 3.2절 · 부록 C).
    """
    k = int(np.ceil((n_cal + 1) * level - 1e-9))
    if k > n_cal:
        return 1.0, 1.0                                   # 범위가 무한대라 항상 맞는다
    dist = betabinom(n, k, n_cal + 1 - k)
    return float(dist.ppf(lo) / n), float(dist.ppf(hi) / n)


def narrower_share(runs: pd.DataFrame, level: int, scheme: str = "mixed", method: str = "conformal") -> float:
    """무작위 나누기에서 6개 특징 모델의 평균 범위 폭이 단순 모델보다 좁았던 나누기의 비율.

    같은 반복(repeat)에서는 두 모델이 같은 학습 셀·같은 보정 셀을 쓰므로 짝지어 비교한다.
    """
    r = runs[(runs["scheme"] == scheme) & (runs["method"] == method) & (runs["level"] == level)
             & (runs["group"] == "all")]
    w = r.pivot_table(index="repeat", columns="model", values="width_mean")
    return float((w["full"] < w["simple"]).mean())


def summarize_splits(runs: pd.DataFrame) -> pd.DataFrame:
    """무작위 나누기 결과를 (나누기 방식, 모델, 방법, 수준, 묶음)마다 요약."""
    def q(p):
        return lambda s: float(np.percentile(s, p))
    runs = runs.assign(short=runs["coverage"] < runs["level"] / 100 - 0.1)  # 약속보다 10%p 넘게 덜 맞음
    # 중앙값 열 이름을 mid로 둔다 (median으로 두면 표의 한 줄에서 r.median이 pandas 함수와 겹친다)
    return runs.groupby(["scheme", "model", "method", "level", "group"], sort=False).agg(
        repeats=("coverage", "size"), cells=("cells", "median"), mid=("coverage", "median"),
        p10=("coverage", q(10)), p90=("coverage", q(90)), width=("width_mean", "median"),
        below=("below", "mean"), above=("above", "mean"), short_by_10=("short", "mean")).reset_index()


SPLITS = ("train", "test", "secondary_test")


def shift(features: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """나누기마다 수명 분포와, 학습 셀 전체로 만든 모델의 log 오차(치우침·흩어짐·보통 크기)와 MAPE·RMSE.

    학습 셀의 오차는 모델이 직접 본 셀에서 잰 것이라 실제보다 작게 나오는 쪽이다.
    """
    train = features[features["split"] == "train"]
    models = {kind: make(kind, seed).fit(train, train["cycle_life"].to_numpy()) for kind in ("simple", "full")}
    rows = []
    for split in SPLITS:
        f = features[features["split"] == split]
        row = {"split": split, "cells": len(f), "batches": ", ".join(sorted(f["batch"].unique())),
               "life_median": float(f["cycle_life"].median()),
               "life_p10": float(np.percentile(f["cycle_life"], 10)),
               "life_p90": float(np.percentile(f["cycle_life"], 90))}
        for kind, model in models.items():
            r = np.log10(f["cycle_life"].to_numpy()) - _log_pred(model, f)
            row[f"bias_{kind}"] = float(10 ** r.mean() - 1)        # +면 실제가 예측보다 평균적으로 길다
            row[f"spread_{kind}"] = float(np.std(r, ddof=1))       # log10 오차의 표준편차
            row[f"abs_median_{kind}"] = float(np.median(np.abs(r)))  # 보통 셀의 |log10 오차|
            e = errors(f["cycle_life"].to_numpy(), model.predict(f))
            row[f"mape_{kind}"], row[f"rmse_{kind}"] = e["mape"], e["rmse"]   # 평균 % 오차, 사이클 오차
        rows.append(row)
    return pd.DataFrame(rows)


def feature_shift(features: pd.DataFrame) -> pd.DataFrame:
    """특징마다: 평가 셀의 중앙값이 학습 셀 중앙값에서 얼마나 떨어졌나(학습 셀 표준편차 단위),
    학습 셀의 최솟값~최댓값 밖인 셀 비율."""
    train = features[features["split"] == "train"]
    rows = []
    for name in FEATURES:
        sd, lo, hi = train[name].std(ddof=1), train[name].min(), train[name].max()
        for split in SPLITS[1:]:
            v = features.loc[features["split"] == split, name]
            rows.append({"feature": name, "split": split, "z": float((v.median() - train[name].median()) / sd),
                         "outside": float(((v < lo) | (v > hi)).mean())})
    return pd.DataFrame(rows)


def policy_overlap(features: pd.DataFrame) -> tuple[int, int]:
    """b3 충전 조건 중 (이름 끝의 '-newstructure'를 떼고) b1·b2에도 있는 조건 수, b3 조건 수."""
    strip = features["policy"].str.replace("-newstructure", "", regex=False)
    b3 = set(strip[features["batch"] == "b3"])
    rest = set(strip[features["batch"] != "b3"])
    return len(b3 & rest), len(b3)
