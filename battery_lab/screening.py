"""5사이클로 수명이 짧을 셀 골라내기 (두 그룹 분류).

논문(Severson 외, 2019)은 첫 5사이클 데이터만으로 셀을 '짧은 수명 / 긴 수명' 두 그룹으로 나눠 테스트 오차 4.9%를 보고했다.
기준 수명은 저자가 올린 논문 PDF의 Table 2와 보충 자료 Supplementary Note 4에서 확인했다 → 550사이클.
(screening_report.PAPER에 출처 주소와 확인한 숫자를 모아 두었다.)

특징 (n사이클까지의 데이터만 쓴다. n보다 뒤 사이클은 절대 쓰지 않는다)
- logvar      log10 Var(ΔQ(V)). ΔQ = (n번째 방전 곡선) − (기준 곡선).
              기준은 n ≤ 10이면 바로 앞 사이클(n−1, 논문의 ΔQ₅₋₄와 같음), n > 10이면 10번째 사이클(논문의 ΔQ₁₀₀₋₁₀과 같음)
- logmin      log10 |min ΔQ(V)|
- q2          2번째 사이클 방전 용량 (b1 배치는 1번째 사이클 기록이 없어 모두 2번째부터 쓴다)
- qmax_q2     2~n사이클 중 최대 방전 용량 − 2번째 사이클 방전 용량
- ir2, ir_min, ir_diff   2번째 사이클 내부저항, 2~n 중 최솟값, n번째 − 2번째
- chargetime  2~5사이클 충전 시간 평균 (충전 조건을 거의 그대로 담는다)
- tavg, tmax, tmin       2~n사이클 캔 표면 온도의 평균·최댓값·최솟값 (공식 설명상 믿기 어려운 측정)

두 곡선 중 하나라도 깨끗하지 않으면(features.is_clean) 두 곡선을 함께 1~2사이클 앞으로 옮겨 쓴다.
앞으로만 옮기므로 n보다 뒤의 데이터는 들어오지 않는다. 실제로 쓴 사이클은 late_used·early_used에 남긴다.

모델 (scikit-learn, 표준화한 뒤 로지스틱 회귀. 결과는 '단수명일 확률')
- variance  ΔQ 분산 1개 + 약한 L2 규제 (논문의 'variance classifier'와 같은 형태)
- cell      셀 신호만: ΔQ 2개 + 용량 2개 + 내부저항 3개, L1 규제(쓸모없는 특징의 계수를 0으로 만든다)
- full      셀 신호 + 충전 시간 + 온도 3개, L1 규제 (논문의 'full classifier'를 참고)
규제 세기는 학습 셀 안에서 5겹 교차 검증으로 고른다.

비교 기준 (모델 없이도 이만큼은 맞힌다)
- majority  학습 셀에서 더 많은 쪽으로 모두 찍기
- batch     배치(b1·b2·b3)마다 학습 셀에서 더 많은 쪽으로 찍기. 셀 자체는 보지 않고 '어느 시험 묶음인가'만 본다
"""

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import is_clean
from .quality import QD_RANGE, T_RANGE, LONG_CHARGE, load_curves, load_processed
from .sizes import SEED, n_train

THRESHOLD = 550        # 논문 기준: 550사이클 미만이면 단수명 (Table 2, Supplementary Note 4)
ALT_THRESHOLD = 700    # 논문 보충 자료가 함께 본 두 번째 기준 (Supplementary Fig. 15)
CYCLES = (5, 10, 20, 50, 100)
SHIFT = 2              # 곡선이 깨끗하지 않을 때 앞으로 옮겨 볼 사이클 수
CHARGE_CYCLES = (2, 5)
REPEATS = 100
CS = np.logspace(-2, 2, 9)   # 규제 세기 후보 (C가 작을수록 규제가 세다)
FOLDS = 5
MISS_COST = 5          # 운영점 가정: 단수명 셀 하나를 놓치면(출하) 장수명 셀 5개를 버린 만큼 손해 (우리가 정한 값)
OPERATING_KIND = "cell"  # 운영점을 정하는 모델 (이유는 docs/screening.md '방법')
CUTS = np.round(np.linspace(0, 1, 101), 2)

CELL = ("logvar", "logmin", "q2", "qmax_q2", "ir2", "ir_min", "ir_diff")
FULL = (*CELL, "chargetime", "tavg", "tmax", "tmin")
FEATURE_SETS = {"variance": ("logvar",), "cell": CELL, "full": FULL}
KINDS = tuple(FEATURE_SETS)
NAMES = {"variance": "ΔQ 분산 1개", "cell": "셀 신호 (충전 시간·온도 빼고)", "full": "전체 (충전 시간·온도 포함)",
         "majority": "모두 많은 쪽으로 찍기", "batch": "배치만 보고 찍기"}
SPLIT_ORDER = ("train", "test", "secondary_test")


# ---------------------------------------------------------------- 특징

def reference(n: int) -> int:
    """ΔQ의 기준 사이클: n ≤ 10이면 바로 앞 사이클, 아니면 10번째."""
    return n - 1 if n <= 10 else 10


def pick_pair(curves: dict[str, np.ndarray], n: int, shift: int = SHIFT) -> tuple[int, int, int, int] | None:
    """(n번째 행, 기준 행, 쓴 n, 쓴 기준). 깨끗하지 않으면 두 곡선을 함께 앞으로 옮긴다 (뒤로는 안 간다)."""
    rows = {int(c): j for j, c in enumerate(curves["cycle"])}
    for s in range(shift + 1):
        late, early = n - s, reference(n) - s
        if early < 2:  # b1은 1번째 사이클 기록이 없어 2번째부터 쓴다
            break
        if late in rows and early in rows and is_clean(curves["Qdlin"][rows[late]]) \
                and is_clean(curves["Qdlin"][rows[early]]):
            return rows[late], rows[early], late, early
    return None


def _log_abs(x: float) -> float:
    with np.errstate(divide="ignore"):
        v = np.log10(abs(x))
    return float(v) if np.isfinite(v) else np.nan  # log(0) = −∞는 조용히 쓰지 않고 비워 둔다


def early_features(curves: dict[str, np.ndarray], summary: pd.DataFrame, n: int) -> dict:
    """n사이클까지의 데이터만으로 만든 특징."""
    out = {**dict.fromkeys(FULL, np.nan), "late_used": np.nan, "early_used": np.nan}
    pair = pick_pair(curves, n)
    if pair:
        a, b, out["late_used"], out["early_used"] = pair
        d = curves["Qdlin"][a].astype(np.float64) - curves["Qdlin"][b]
        if np.var(d) > 0:
            out["logvar"] = _log_abs(np.var(d))
            out["logmin"] = _log_abs(d.min())

    s = summary[(summary["cycle"] >= 2) & (summary["cycle"] <= n)]
    s = s[~((s["QD"] == 0) & (s["IR"] == 0))]  # b1의 1번째 사이클 빈 자리 (cycle ≥ 2라 이미 빠지지만 한 번 더)
    q = s[(s["QD"] > QD_RANGE[0]) & (s["QD"] < QD_RANGE[1])]
    q2 = q.loc[q["cycle"] == 2, "QD"]
    if len(q2):
        out["q2"] = float(q2.iloc[0])
        out["qmax_q2"] = float(q["QD"].max() - q2.iloc[0])

    r = s[s["IR"] > 0]
    ir2, irn = r.loc[r["cycle"] == 2, "IR"], r.loc[r["cycle"] == n, "IR"]
    if len(ir2):
        out["ir2"] = float(ir2.iloc[0])
    if len(r):
        out["ir_min"] = float(r["IR"].min())
    if len(ir2) and len(irn):
        out["ir_diff"] = float(irn.iloc[0] - ir2.iloc[0])

    c = s[(s["cycle"] >= CHARGE_CYCLES[0]) & (s["cycle"] <= CHARGE_CYCLES[1])
          & (s["chargetime"] > 0) & (s["chargetime"] <= LONG_CHARGE)]
    if len(c):
        out["chargetime"] = float(c["chargetime"].mean())

    t = s[(s["Tavg"] > T_RANGE[0]) & (s["Tavg"] < T_RANGE[1])]
    if len(t):
        out["tavg"] = float(t["Tavg"].mean())
        out["tmax"] = float(t["Tmax"].max())
        out["tmin"] = float(t["Tmin"].min())
    return out


def build_features(folder: Path, cycles: tuple[int, ...] = CYCLES) -> dict[int, pd.DataFrame]:
    """사이클 수 n마다 셀 124개의 특징 표. 곡선 파일은 셀마다 한 번만 읽는다."""
    cells, summary = load_processed(folder)
    groups = dict(tuple(summary.groupby("cell_id")))
    rows = {n: [] for n in cycles}
    for meta in cells.to_dict("records"):
        curves = load_curves(folder, meta["cell_id"])
        base = {"cell_id": meta["cell_id"], "batch": meta["batch"], "policy": meta["policy"],
                "split": meta["split"], "cycle_life": float(meta["cycle_life"])}
        for n in cycles:
            rows[n].append({**base, **early_features(curves, groups[meta["cell_id"]], n)})
    return {n: pd.DataFrame(v) for n, v in rows.items()}


def substitutions(feats: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """깨끗하지 않은 곡선 때문에 앞 사이클로 옮겨 쓴 경우와, 특징이 비어 있는 경우."""
    rows = []
    for n, f in feats.items():
        for r in f.itertuples():
            moved = r.late_used != n or r.early_used != reference(n)
            empty = [k for k in FULL if not np.isfinite(getattr(r, k))]
            if moved or empty:
                rows.append({"cycles": n, "cell_id": r.cell_id, "late_used": r.late_used, "early_used": r.early_used,
                             "empty": ",".join(empty)})
    return pd.DataFrame(rows, columns=["cycles", "cell_id", "late_used", "early_used", "empty"])


# ---------------------------------------------------------------- 모델

def labels(life, threshold: float = THRESHOLD) -> np.ndarray:
    """1 = 단수명(기준 미만), 0 = 장수명."""
    return (np.asarray(life, float) < threshold).astype(int)


_NEW_API = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)  # 1.8부터 L1 지정 방식이 바뀌었다


def _l1_logistic(C: float, seed: int = 0):
    # saga 풀이법은 학습 중에 데이터 순서를 무작위로 섞는다. random_state를 주지 않으면 돌릴 때마다 숫자가 달라진다.
    common = {"C": C, "solver": "saga", "max_iter": 5000, "random_state": seed}
    return LogisticRegression(**common, l1_ratio=1.0) if _NEW_API else LogisticRegression(**common, penalty="l1")


def _l1_logistic_cv(seed: int):
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    common = {"Cs": CS, "cv": folds, "solver": "saga", "scoring": "neg_log_loss", "max_iter": 5000,
              "random_state": seed}
    if _NEW_API:
        return LogisticRegressionCV(**common, l1_ratios=(1.0,), use_legacy_attributes=False)
    return LogisticRegressionCV(**common, penalty="l1")


def make_classifier(kind: str, seed: int = 0):
    """빈 특징은 학습 셀의 중앙값으로 채우고, 표준화한 뒤 로지스틱 회귀."""
    head = LogisticRegression(C=1.0) if kind == "variance" else _l1_logistic_cv(seed)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), head)


def _fit(model, X: np.ndarray, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)


def fit_predict(kind: str, train: pd.DataFrame, y_train: np.ndarray, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """학습 셀로 맞춘 뒤 test 셀의 '단수명일 확률'. 비교 기준(majority·batch)도 같은 모양으로 돌려준다."""
    if kind == "majority":
        return np.full(len(test), float(y_train.mean() > 0.5))
    if kind == "batch":
        overall = float(y_train.mean() > 0.5)
        share = pd.Series(y_train, index=train["batch"].to_numpy()).groupby(level=0).mean()
        rule = {b: (overall if v == 0.5 else float(v > 0.5)) for b, v in share.items()}
        return np.array([rule.get(b, overall) for b in test["batch"]])
    cols = list(FEATURE_SETS[kind])
    model = _fit(make_classifier(kind, seed), train[cols].to_numpy(), y_train)
    return model.predict_proba(test[cols].to_numpy())[:, 1]


def coefficients(kind: str, train: pd.DataFrame, y_train: np.ndarray, seed: int = 0) -> pd.Series:
    """표준화한 특징의 계수 (+면 단수명 쪽). L1 규제로 0이 된 특징은 '안 쓴 특징'이다."""
    cols = list(FEATURE_SETS[kind])
    model = _fit(make_classifier(kind, seed), train[cols].to_numpy(), y_train)
    return pd.Series(model[-1].coef_.ravel(), index=cols)


def confusion(y: np.ndarray, p: np.ndarray, cut: float = 0.5) -> dict:
    """P(단수명) ≥ cut이면 '단수명'으로 판정(=출하하지 않고 뺀다).

    caught 잡은 단수명 · missed 놓친 단수명 · discarded 버린 장수명 · kept 통과한 장수명
    """
    y, flag = np.asarray(y, int), np.asarray(p, float) >= cut
    caught, missed = int(np.sum(flag & (y == 1))), int(np.sum(~flag & (y == 1)))
    discarded, kept = int(np.sum(flag & (y == 0))), int(np.sum(~flag & (y == 0)))
    short, long_ = caught + missed, discarded + kept
    return {"cells": len(y), "short": short, "caught": caught, "missed": missed, "discarded": discarded, "kept": kept,
            "accuracy": (caught + kept) / len(y) * 100 if len(y) else np.nan,
            "miss_rate": missed / short * 100 if short else np.nan,
            "discard_rate": discarded / long_ * 100 if long_ else np.nan}


def choose_cut(y: np.ndarray, p: np.ndarray, miss_cost: float = MISS_COST) -> float:
    """손해 = miss_cost × 놓친 단수명 + 버린 장수명이 가장 작은 기준값.

    손해가 똑같이 가장 작은 기준값이 여러 개면 그 가운데 값을 고른다 (끝값은 학습 셀에 너무 딱 맞춘 값이라 흔들린다).
    처음에는 '놓침 10% 이하인 가장 높은 기준값'을 썼다가 실패했다 (docs/screening.md 시행착오).
    """
    y, p = np.asarray(y, int), np.asarray(p, float)
    cost = np.array([miss_cost * np.sum((p < c) & (y == 1)) + np.sum((p >= c) & (y == 0)) for c in CUTS])
    best = CUTS[cost == cost.min()]
    return float(best[len(best) // 2])


def oof_proba(kind: str, train: pd.DataFrame, y_train: np.ndarray, seed: int = 0) -> np.ndarray:
    """학습 셀마다 '그 셀을 빼고 맞춘 모델'의 확률 (5겹). 운영점을 테스트 셀을 보지 않고 정하려고 쓴다.

    시간을 줄이려고 규제 세기는 학습 셀 전체로 한 번 고른 값을 5겹 모두에 쓴다 (테스트 셀은 전혀 쓰지 않는다).
    """
    cols = list(FEATURE_SETS[kind])
    X = train[cols].to_numpy()
    model = make_classifier(kind, seed)
    if kind != "variance":
        C = float(np.ravel(_fit(model, X, y_train)[-1].C_)[0])
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), _l1_logistic(C, seed))
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return cross_val_predict(model, X, y_train, cv=folds, method="predict_proba")[:, 1]


# ---------------------------------------------------------------- 평가

def paper_split(f: pd.DataFrame, threshold: float = THRESHOLD, kinds=(*KINDS, "majority", "batch"),
                seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """논문 나누기(학습 41 · 테스트 43 · 2차 테스트 40). (나누기별 성적, 셀별 확률)

    test_all = 테스트 + 2차 테스트를 합친 성적 (논문의 4.9%도 두 테스트를 합친 오차다).
    """
    y = labels(f["cycle_life"], threshold)
    tr = (f["split"] == "train").to_numpy()
    rows, probs = [], f[["cell_id", "batch", "policy", "split", "cycle_life"]].copy()
    probs["short"] = y
    for kind in kinds:
        p = fit_predict(kind, f[tr], y[tr], f, seed)
        probs[f"p_{kind}"] = p  # 'batch' 열(배치 이름)을 덮어쓰지 않도록 앞에 p_를 붙인다
        for split in (*SPLIT_ORDER, "test_all"):
            m = ~tr if split == "test_all" else (f["split"] == split).to_numpy()
            rows.append({"kind": kind, "split": split, **confusion(y[m], p[m])})
    return pd.DataFrame(rows), probs


def operating_point(f: pd.DataFrame, kind: str = OPERATING_KIND, threshold: float = THRESHOLD,
                    seed: int = 0) -> tuple[float, pd.DataFrame, np.ndarray]:
    """논문 나누기에서 운영점 정하기: 학습 셀의 교차 검증 확률만 보고 기준값을 고른 뒤 테스트에 그대로 쓴다.

    돌려주는 값: (기준값, 기준값 0.5와 운영점에서 나누기별 성적, 셀별 확률)
    """
    y = labels(f["cycle_life"], threshold)
    tr = (f["split"] == "train").to_numpy()
    cut = choose_cut(y[tr], oof_proba(kind, f[tr], y[tr], seed))
    p = fit_predict(kind, f[tr], y[tr], f, seed)
    rows = []
    for name, c in (("default", 0.5), ("operating", cut)):
        for split in (*SPLIT_ORDER, "test_all"):
            m = ~tr if split == "test_all" else (f["split"] == split).to_numpy()
            rows.append({"point": name, "cut": c, "split": split, **confusion(y[m], p[m], c)})
    return cut, pd.DataFrame(rows), p


def stratified_split(y: np.ndarray, k_train: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """단수명·장수명 비율을 지키며 k_train개를 학습으로 뽑는다."""
    short, long_ = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k_short = int(round(k_train * len(short) / len(y)))
    train = np.r_[rng.choice(short, k_short, replace=False), rng.choice(long_, k_train - k_short, replace=False)]
    return np.sort(train), np.setdiff1d(np.arange(len(y)), train)


def random_splits(feats: dict[int, pd.DataFrame], threshold: float = THRESHOLD, repeats: int = REPEATS,
                  seed: int = SEED, kinds=(*KINDS, "majority", "batch"), cut_kind: str | None = OPERATING_KIND,
                  cut_cycles: int = 5, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """124개를 41(학습) / 83(평가)으로 여러 번 무작위로 나눈다. 반복마다 같은 나누기를 모든 n·모델에 쓴다.

    돌려주는 값: (반복마다 성적, cut_kind·cut_cycles 모델의 기준값별 놓침·버림)
    운영점: 반복마다 학습 셀의 교차 검증 확률로 기준값을 정하고(choose_cut), 그 기준값을 평가 셀에 그대로 쓴다.
    """
    rng = np.random.default_rng(seed)
    first = next(iter(feats.values()))
    y = labels(first["cycle_life"], threshold)
    k_train = n_train(len(first))
    rows, curves = [], []
    for rep in range(repeats):
        if progress and rep and rep % 25 == 0:
            progress(f"  무작위 나누기 {rep}/{repeats}")
        train, test = stratified_split(y, k_train, rng)
        for n, f in sorted(feats.items()):
            for kind in kinds:
                p = fit_predict(kind, f.iloc[train], y[train], f.iloc[test], seed=rep)
                rows.append({"cycles": n, "kind": kind, "repeat": rep, "cut": 0.5, **confusion(y[test], p)})
                if kind == cut_kind and n == cut_cycles:
                    cut = choose_cut(y[train], oof_proba(kind, f.iloc[train], y[train], seed=rep))
                    rows.append({"cycles": n, "kind": f"{kind}@op", "repeat": rep, "cut": cut,
                                 **confusion(y[test], p, cut)})
                    for c in CUTS:
                        m = confusion(y[test], p, c)
                        curves.append({"repeat": rep, "cut": c, "miss_rate": m["miss_rate"],
                                       "discard_rate": m["discard_rate"]})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["cycles", "kind"])
    out = {}
    for col in ("accuracy", "miss_rate", "discard_rate", "cut"):
        out[f"{col}_median"] = g[col].median()
        out[f"{col}_p10"] = g[col].quantile(0.1)
        out[f"{col}_p90"] = g[col].quantile(0.9)
    return pd.DataFrame(out).reset_index()


def summarize_tradeoff(curves: pd.DataFrame) -> pd.DataFrame:
    g = curves.groupby("cut")
    return pd.DataFrame({"miss_median": g["miss_rate"].median(), "miss_p10": g["miss_rate"].quantile(0.1),
                         "miss_p90": g["miss_rate"].quantile(0.9),
                         "discard_median": g["discard_rate"].median(), "discard_p10": g["discard_rate"].quantile(0.1),
                         "discard_p90": g["discard_rate"].quantile(0.9)}).reset_index()


# ---------------------------------------------------------------- 같은 조건 안에서

def _pairs(group) -> tuple[np.ndarray, np.ndarray]:
    """같은 묶음 안의 모든 셀 짝 (i, j)."""
    group = np.asarray(group)
    idx = [np.flatnonzero(group == g) for g in pd.unique(group)]
    pairs = [(a, b) for members in idx for a, b in combinations(members, 2)]
    if not pairs:
        return np.array([], int), np.array([], int)
    return np.array([a for a, _ in pairs], int), np.array([b for _, b in pairs], int)


def pair_concordance(score, life, group) -> tuple[float, int]:
    """같은 묶음(group) 안의 셀 두 개씩 짝지어, 점수가 높은 셀이 먼저 수명이 다했는지 맞힌 비율(%)과 짝 수.

    50%면 동전 던지기와 같다. 수명이 같은 짝, 점수가 같은 짝, 점수가 빈 셀은 뺀다.
    """
    s, life = np.asarray(score, float), np.asarray(life, float)
    i, j = _pairs(group)
    ok = np.isfinite(s[i]) & np.isfinite(s[j]) & (life[i] != life[j]) & (s[i] != s[j])
    i, j = i[ok], j[ok]
    if not len(i):
        return np.nan, 0
    right = (s[i] > s[j]) == (life[i] < life[j])
    return float(right.mean() * 100), int(len(i))


def shuffled_concordance(score, life, group, repeats: int = 2000, seed: int = SEED) -> np.ndarray:
    """같은 묶음 안에서 수명을 무작위로 섞었을 때(= 점수가 수명과 상관없을 때) 나오는 맞힌 비율들."""
    rng = np.random.default_rng(seed)
    life, group = np.asarray(life, float), np.asarray(group)
    members = [np.flatnonzero(group == g) for g in pd.unique(group)]
    out = np.empty(repeats)
    for k in range(repeats):
        shuffled = life.copy()
        for m in members:
            shuffled[m] = life[rng.permutation(m)]
        out[k] = pair_concordance(score, shuffled, group)[0]
    return out


def within_condition(feats: dict[int, pd.DataFrame], repeats: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """같은 배치·같은 충전 조건 셀끼리만 비교: ΔQ 분산이 큰 셀이 먼저 수명이 다했나? (논문: 분산이 클수록 수명이 짧다)

    shuffled_p5·p95: 수명을 묶음 안에서 섞었을 때의 5~95% 범위. 이 안이면 '운으로도 나올 수 있는 값'이다.
    """
    rows = []
    for n, f in sorted(feats.items()):
        group = (f["batch"] + "|" + f["policy"]).to_numpy()
        conc, pairs = pair_concordance(f["logvar"], f["cycle_life"], group)
        null = shuffled_concordance(f["logvar"], f["cycle_life"], group, repeats, seed)
        sizes = pd.Series(group).value_counts()
        rows.append({"cycles": n, "concordance": conc, "pairs": pairs,
                     "shuffled_p5": float(np.nanpercentile(null, 5)), "shuffled_p95": float(np.nanpercentile(null, 95)),
                     "groups": int((sizes >= 2).sum()), "cells": int(sizes[sizes >= 2].sum())})
    return pd.DataFrame(rows)


def batch_table(f: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """배치마다 셀 수, 단수명 셀 수, 수명 범위."""
    y = labels(f["cycle_life"], threshold)
    g = f.assign(short=y).groupby("batch")
    return pd.DataFrame({"cells": g.size(), "short": g["short"].sum(), "life_min": g["cycle_life"].min(),
                         "life_median": g["cycle_life"].median(), "life_max": g["cycle_life"].max()}).reset_index()


def same_policy_across_batches(f: pd.DataFrame) -> pd.DataFrame:
    """충전 조건 이름이 똑같은데 배치가 다른 셀들 (b3의 '-newstructure'는 다른 조건으로 본다)."""
    multi = f.groupby("policy")["batch"].nunique()
    shared = multi[multi >= 2].index
    g = f[f["policy"].isin(shared)].groupby(["policy", "batch"])["cycle_life"]
    return pd.DataFrame({"cells": g.size(), "life_min": g.min(), "life_max": g.max()}).reset_index()
