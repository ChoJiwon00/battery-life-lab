"""사이클 수 실험: 100사이클 대신 20·30·…·90사이클만 보고도 수명을 예측할 수 있을까?

ΔQ₁₀₀₋₁₀ 대신 ΔQ_k−10(k번째 − 10번째 사이클의 방전 곡선 차이)으로 특징을 만들어 k에 따라 비교한다.

- 신호 세기: k마다 log Var(ΔQ_k−10)와 log 수명의 상관 r
- 논문 나누기: 학습 41 · 테스트 43 · 2차 테스트 40 그대로
- 무작위 나누기: 124개를 41(학습) / 83(평가)으로 여러 번 무작위로 나눈다.
  반복마다 같은 나누기를 모든 k에 쓴다 → k끼리의 차이는 사이클 수 효과만 남는다.
- 오차 지도: 셀 수 N(124·70·35·10) × 사이클 수 k. 데이터 크기 실험(sizes.py)과 같은 방식으로,
  반복마다 같은 셀 뽑기를 모든 k에 쓴다. 단순 모델만 (셀 10개면 6개 특징 모델을 만들 수 없어서).
"""

import numpy as np
import pandas as pd

from .features import FEATURES
from .model import errors, make
from .sizes import n_train

LATES = (20, 30, 40, 50, 60, 70, 80, 90, 100)
SIZES = (124, 70, 35, 10)
SPLIT_REPEATS = 100
GRID_REPEATS = 300
SEED = 2026


def check(feats: dict[int, pd.DataFrame]) -> None:
    for k, f in feats.items():
        missing = f[list(FEATURES)].isna().any(axis=1)
        if missing.any():
            raise ValueError(f"{k}사이클 특징이 비어 있는 셀: {', '.join(f.loc[missing, 'cell_id'])}")


def substitutions(feats: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """깨끗하지 않은 곡선 대신 가까운 사이클을 쓴 경우."""
    rows = []
    for k, f in feats.items():
        moved = f[(f["late_used"] != k) | (f["early_used"] != 10)]
        rows += [{"late": k, "cell_id": r.cell_id, "late_used": int(r.late_used), "early_used": int(r.early_used)}
                 for r in moved.itertuples()]
    return pd.DataFrame(rows, columns=["late", "cell_id", "late_used", "early_used"])


def correlations(feats: dict[int, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([{"late": k, "r": float(np.corrcoef(f["logvar"], np.log10(f["cycle_life"]))[0, 1])}
                         for k, f in sorted(feats.items())])


def paper_split(feats: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for k, f in sorted(feats.items()):
        train = f[f["split"] == "train"]
        for kind in ("simple", "full"):
            model = make(kind).fit(train, train["cycle_life"].to_numpy())
            for split in ("train", "test", "secondary_test"):
                part = f[f["split"] == split]
                rows.append({"late": k, "model": kind, "split": split,
                             **errors(part["cycle_life"], model.predict(part))})
    return pd.DataFrame(rows)


def random_splits(feats: dict[int, pd.DataFrame], repeats: int = SPLIT_REPEATS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(next(iter(feats.values())))
    k_train = n_train(n)
    rows = []
    for rep in range(repeats):
        order = rng.permutation(n)
        train, test = order[:k_train], order[k_train:]
        for k, f in sorted(feats.items()):
            life = f["cycle_life"].to_numpy()
            for kind in ("simple", "full"):
                model = make(kind, seed=rep).fit(f.iloc[train], life[train])
                rows.append({"late": k, "model": kind, "repeat": rep,
                             **errors(life[test], model.predict(f.iloc[test]))})
    return pd.DataFrame(rows)


def grid(feats: dict[int, pd.DataFrame], sizes=SIZES, repeats: int = GRID_REPEATS, seed: int = SEED) -> pd.DataFrame:
    """셀 수 N × 사이클 수 k의 실제 오차 (단순 모델). 실제 오차 = 학습에 안 쓴 나머지 모든 셀."""
    rng = np.random.default_rng(seed)
    total = len(next(iter(feats.values())))
    rows = []
    for n in sizes:
        k_train = n_train(n)
        for rep in range(repeats):
            pick = rng.choice(total, size=n, replace=False)
            train = pick[:k_train]
            outside = np.setdiff1d(np.arange(total), train)
            for k, f in sorted(feats.items()):
                life = f["cycle_life"].to_numpy()
                model = make("simple").fit(f.iloc[train], life[train])
                rows.append({"n": n, "late": k, "repeat": rep,
                             "actual_mape": errors(life[outside], model.predict(f.iloc[outside]))["mape"]})
    return pd.DataFrame(rows)


def summarize_splits(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["late", "model"])["mape"]
    return pd.DataFrame({"median": g.median(), "p10": g.quantile(0.1), "p90": g.quantile(0.9)}).reset_index()


def summarize_grid(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["n", "late"])["actual_mape"]
    return pd.DataFrame({"median": g.median(), "p90": g.quantile(0.9),
                         "over25": g.apply(lambda s: float((s > 25).mean()))}).reset_index()


def shortest_enough(summary: pd.DataFrame, model: str = "simple", margin: float = 1.0) -> int | None:
    """중앙값 오차가 100사이클보다 margin %p 이내로 나빠지는 가장 짧은 사이클 수."""
    s = summary[summary["model"] == model].set_index("late")["median"]
    if 100 not in s:
        return None
    ok = [k for k in sorted(s.index) if s[k] <= s[100] + margin]
    return ok[0] if ok else None


def minutes_per_cycle(folder) -> float:
    """원시 데이터가 있는 100사이클 이내 사이클들의 한 사이클 시간 중앙값(분)."""
    from .quality import load_processed  # noqa: PLC0415

    cells, _ = load_processed(folder)
    durations = []
    for cid in cells["cell_id"]:
        with np.load(folder / "raw" / f"{cid}.npz") as z:
            o, idx, t = z["raw_offsets"], z["raw_index"], z["raw_t"]
        for k in range(len(idx)):
            part = t[o[k]:o[k + 1]]
            if part.size and 1 <= idx[k] <= 99:  # raw_index는 0부터 세는 줄 번호: 줄 1~99 = 사이클 2~100
                durations.append(float(np.nanmax(part)))
    return float(np.median(durations)) if durations else float("nan")


def signal_curve(folder, lates=tuple(range(20, 101, 5)), windows=(1, 5)) -> pd.DataFrame:
    """k마다·곡선 방식마다 상관 r. 특징이 비는 셀은 빼고 계산하고 몇 개로 계산했는지 남긴다."""
    from .features import build_many  # noqa: PLC0415

    rows = []
    for w in windows:
        for k, f in build_many(folder, lates, window=w).items():
            ok = f["logvar"].notna()
            rows.append({"window": w, "late": k, "cells": int(ok.sum()),
                         "r": float(np.corrcoef(f.loc[ok, "logvar"], np.log10(f.loc[ok, "cycle_life"]))[0, 1])})
    return pd.DataFrame(rows)


def deepest_dip(curve: pd.DataFrame, window: int = 1) -> dict | None:
    """곡선 한 개 방식에서 r이 양옆 평균보다 가장 많이 약해진 k (안쪽 점만)."""
    s = curve[curve["window"] == window].sort_values("late").set_index("late")["r"]
    best = None
    for prev, k, nxt in zip(s.index[:-2], s.index[1:-1], s.index[2:]):
        drop = s[k] - (s[prev] + s[nxt]) / 2  # r은 음수라 클수록 약해진 것
        if best is None or drop > best["drop"]:
            best = {"late": int(k), "prev": int(prev), "next": int(nxt), "r": float(s[k]),
                    "r_prev": float(s[prev]), "r_next": float(s[nxt]), "drop": float(drop)}
    return best


def dip_checks(folder, dip: dict, summary: pd.DataFrame, gap: int = 10) -> dict:
    """꺾인 지점 k에서 셀마다 logvar가 양옆(k±gap) 평균보다 벗어난 정도가, 같은 방식으로 잰 온도 벗어남과 엮여 있나."""
    from .features import build_many  # noqa: PLC0415

    k = dip["late"]
    feats = build_many(folder, (k - gap, k, k + gap))
    lv = {kk: f.set_index("cell_id")["logvar"] for kk, f in feats.items()}
    dev = lv[k] - (lv[k - gap] + lv[k + gap]) / 2
    s = summary[(summary["QD"] > 0) & (summary["Tavg"] > 0)]
    temp = s.pivot_table(index="cell_id", columns="cycle", values="Tavg")
    dt = (temp[k] - (temp[k - gap] + temp[k + gap]) / 2).reindex(dev.index)
    ok = dev.notna() & dt.notna()
    out = {"gap": gap, "cells": int(ok.sum()), "corr_all": float(np.corrcoef(dev[ok], dt[ok])[0, 1]),
           "dev_abs_median": float(dev.abs().median()), "top": dev.abs().sort_values(ascending=False).head(5).index.tolist(),
           "by_batch": {}}
    for b in ("b1", "b2", "b3"):
        m = ok & dev.index.str.startswith(b)
        if m.sum() > 2:
            out["by_batch"][b] = float(np.corrcoef(dev[m], dt[m])[0, 1])
    batch_t = s.assign(batch=s["cell_id"].str[:2]).groupby(["batch", "cycle"])["Tavg"].median().unstack(0)
    out["batch_temp_shift"] = {b: float(batch_t.loc[k, b] - (batch_t.loc[k - gap, b] + batch_t.loc[k + gap, b]) / 2)
                               for b in batch_t.columns}
    return out


def local_dips(curve: pd.DataFrame, window: int = 1) -> list[dict]:
    """곡선 한 개 방식에서 r이 양옆보다 모두 약한 지점 전부 (꺾인 정도가 큰 순서).

    처음엔 가장 크게 꺾인 한 곳만 골랐는데, 신호가 빠르게 올라오는 30사이클 근처가 뽑혀
    70사이클을 염두에 두고 쓴 설명과 어긋났다 (troubleshooting.md). 그래서 모두 나열한다.
    """
    s = curve[curve["window"] == window].sort_values("late").set_index("late")["r"]
    out = []
    for prev, k, nxt in zip(s.index[:-2], s.index[1:-1], s.index[2:]):
        if s[k] > s[prev] and s[k] > s[nxt]:
            out.append({"late": int(k), "prev": int(prev), "next": int(nxt), "r": float(s[k]),
                        "r_prev": float(s[prev]), "r_next": float(s[nxt]), "drop": float(s[k] - (s[prev] + s[nxt]) / 2)})
    return sorted(out, key=lambda d: -d["drop"])


def error_bumps(summary: pd.DataFrame, model: str = "simple") -> list[int]:
    """오차 중앙값이 바로 앞 k보다 커진 k (오차 그래프의 '혹')."""
    s = summary[summary["model"] == model].sort_values("late")
    return [int(k) for k, a, b in zip(s["late"].iloc[1:], s["median"].iloc[:-1], s["median"].iloc[1:]) if b > a]
