"""ΔQ 특징이 왜 통하나 (해석): 어느 전압 구간이 수명 신호를 만들까?

ΔQ₁₀₀₋₁₀(V) = 100번째 방전 곡선 Q(V) − 10번째 방전 곡선 Q(V). 전압 칸 1,000개(3.5V → 2.0V)마다 값이 하나씩 있다.
논문의 단순 모델은 이 1,000개 값의 분산 하나만 쓴다. 여기서는 분산으로 뭉치기 전에 전압 칸마다 따로 본다.

- 상관 스펙트럼: 전압 칸마다 셀들의 ΔQ(V)와 log10(수명)의 상관 r
- 전압별 단일 특징 모델: 전압 칸 하나의 값만으로 log10(수명)에 직선을 맞춘다 (논문 나누기: 학습 41 → 테스트 43 · 2차 테스트 40)
- dQ/dV(미분 용량): 10번째 사이클 곡선으로 4C 방전의 평탄 구간(봉우리)이 어느 전압에 있는지 본다
- 그룹 비교: 수명 순으로 셀을 세 무리(짧은·중간·긴 수명)로 나눠 ΔQ(V)의 평균과 퍼짐을 비교한다

주의
- ΔQ(V)는 '누적' 값이다. Q(V)는 3.5V부터 V까지 나온 용량을 모두 더한 값이라, 2.9V의 ΔQ에는 그보다 높은 전압에서
  생긴 변화가 모두 들어 있다. 어느 전압에서 변화가 '생겼는지'는 ΔQ를 전압으로 미분한 값(moved)으로 본다.
- 관찰 데이터라 원인을 증명하지 못한다. 전기화학 해석은 가설로만 적는다. 물리 시뮬레이션은 하지 않는다.
- 곡선은 행 번호가 아니라 사이클 번호로 찾는다 (features.find_row: 깨끗하지 않으면 가까운 곡선. 100번째는 앞쪽으로만, 10번째는 ±2).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .features import find_row
from .model import SimpleModel, errors
from .quality import QD_RANGE, load_curves, load_processed
from .sizes import SEED

LATE, EARLY = 100, 10
SPLIT_KEYS = ("train", "test", "secondary_test")
TRANSFORMS = ("log", "raw")      # log = log10|ΔQ(V)| (논문 분산 모델처럼 로그), raw = ΔQ(V) 그대로
PEAK_WINDOW = (3.0, 3.3)         # V. 4C 방전의 주 봉우리를 찾는 범위 (2.0V 끝의 튐을 봉우리로 잘못 잡지 않도록)
COUNT_WINDOW = (2.5, 3.45)       # V. 셀마다 봉우리 개수를 셀 범위
PROMINENCE = 0.05                # 봉우리로 셀 최소 높이 (그 셀의 주 봉우리 높이 대비)
BAND_FRAC = 0.95                 # |r|이 최댓값의 95% 이상인 전압 구간 = 신호가 모인 구간
SMOOTH = 15                      # 칸. ΔQ를 미분하기 전에 이동평균으로 다듬는 폭 (15칸 × 1.5mV ≈ 23mV)
GROUP_NAMES = ("짧은 수명", "중간 수명", "긴 수명")
BOOT = 1000                      # 신호 중심 전압의 흔들림을 볼 때 셀을 다시 뽑는 횟수
IR_REACH = 2                     # 내부저항은 그 사이클 ±2사이클의 중앙값 (한 번 잰 값은 잡음이 크다)


# ---------------------------------------------------------------- 데이터 모으기

def cell_curves(curves: dict[str, np.ndarray], late: int = LATE, early: int = EARLY) -> dict | None:
    """한 셀의 ΔQ(V)와 두 사이클의 dQ/dV. 사이클 번호로 깨끗한 곡선을 찾는다. 없으면 None."""
    a, b = find_row(curves, late, forward=False), find_row(curves, early)  # late 이후 곡선은 쓰지 않는다
    if a is None or b is None:
        return None
    q, p = curves["Qdlin"], curves["dQdV"]
    return {"dq": q[a[0]].astype(np.float64) - q[b[0]].astype(np.float64),
            "dqdv_early": p[b[0]].astype(np.float64), "dqdv_late": p[a[0]].astype(np.float64),
            "late_used": a[1], "early_used": b[1]}


def collect(folder: Path, late: int = LATE, early: int = EARLY) -> dict:
    """모든 셀의 ΔQ(V)·dQ/dV를 (셀 수 × 1000) 행렬로 모은다. 곡선을 못 찾은 셀은 skipped에 남긴다."""
    cells, summary = load_processed(folder)
    keep, dq, p_early, p_late, used, skipped = [], [], [], [], [], []
    voltage = None
    for meta in cells.to_dict("records"):
        curves = load_curves(folder, meta["cell_id"])
        got = cell_curves(curves, late, early)
        if got is None:
            skipped.append(meta["cell_id"])
            continue
        v = np.asarray(curves["Vdlin"], dtype=np.float64)
        if voltage is None:
            voltage = v
        elif not np.allclose(v, voltage):
            raise ValueError(f"{meta['cell_id']}의 전압 칸이 다른 셀과 달라요")
        keep.append(meta)
        dq.append(got["dq"]); p_early.append(got["dqdv_early"]); p_late.append(got["dqdv_late"])
        used.append((got["late_used"], got["early_used"]))
    table = pd.DataFrame(keep)
    table["cycle_life"] = table["cycle_life"].astype(float)
    table["late_used"] = [u[0] for u in used]
    table["early_used"] = [u[1] for u in used]
    return {"cells": table.reset_index(drop=True), "summary": summary, "voltage": voltage,
            "dq": np.array(dq), "dqdv_early": np.array(p_early), "dqdv_late": np.array(p_late), "skipped": skipped}


# ---------------------------------------------------------------- 어느 전압에 신호가 있나

def transform(dq: np.ndarray, kind: str) -> np.ndarray:
    """log = log10|ΔQ| (0이면 비워 둠), raw = ΔQ 그대로."""
    if kind == "raw":
        return np.asarray(dq, dtype=np.float64)
    with np.errstate(divide="ignore"):
        out = np.log10(np.abs(dq))
    return np.where(np.isfinite(out), out, np.nan)


def column_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x의 열(전압 칸)마다 y와의 피어슨 상관 r. 값이 모두 같은 열은 nan."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(axis=0), y - y.mean()
    sx, sy = np.sqrt((xc ** 2).sum(axis=0)), np.sqrt((yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xc * yc[:, None]).sum(axis=0) / (sx * sy)
    return np.where(sx > 0, r, np.nan)


def correlation_spectrum(voltage: np.ndarray, dq: np.ndarray, life: np.ndarray) -> pd.DataFrame:
    """전압 칸마다 ΔQ(V)·log10|ΔQ(V)|와 log10(수명)의 상관, ΔQ의 평균·표준편차(mAh)."""
    y = np.log10(np.asarray(life, dtype=np.float64))
    return pd.DataFrame({"voltage": voltage,
                         "r_raw": column_corr(transform(dq, "raw"), y),
                         "r_log": column_corr(transform(dq, "log"), y),
                         "dq_mean_mah": dq.mean(axis=0) * 1000, "dq_sd_mah": dq.std(axis=0) * 1000})


def strongest(spec: pd.DataFrame, column: str) -> dict:
    """|r|이 가장 큰 전압과 그 r."""
    j = int(np.nanargmax(np.abs(spec[column].to_numpy())))
    return {"voltage": float(spec["voltage"].iloc[j]), "r": float(spec[column].iloc[j]), "index": j}


def band(voltage: np.ndarray, r: np.ndarray, frac: float = BAND_FRAC) -> tuple[float, float]:
    """|r|이 최댓값의 frac 이상으로 이어지는 구간 (높은 전압, 낮은 전압). 최댓값 칸에서 양쪽으로 넓힌다."""
    a = np.abs(np.asarray(r, dtype=np.float64))
    j = int(np.nanargmax(a))
    ok = np.nan_to_num(a, nan=0.0) >= frac * a[j]
    lo, hi = j, j
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(a) - 1 and ok[hi + 1]:
        hi += 1
    v = np.asarray(voltage)[[lo, hi]]
    return float(v.max()), float(v.min())


def boot_peak_voltage(voltage: np.ndarray, x: np.ndarray, life: np.ndarray, repeats: int = BOOT,
                      seed: int = SEED) -> np.ndarray:
    """셀을 복원 추출로 다시 뽑을 때마다 |r|이 가장 큰 전압. 신호 중심이 셀 구성에 따라 얼마나 흔들리는지 본다."""
    rng = np.random.default_rng(seed)
    y = np.log10(np.asarray(life, dtype=np.float64))
    out = np.empty(repeats)
    for k in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        out[k] = voltage[int(np.nanargmax(np.abs(column_corr(x[idx], y[idx]))))]
    return out


def univariate_errors(voltage: np.ndarray, dq: np.ndarray, life: np.ndarray, split: np.ndarray,
                      kind: str = "log") -> pd.DataFrame:
    """전압 칸마다 그 칸의 값 하나로 log10(수명)에 직선을 맞춘다 (학습 셀로만 맞춤). 칸마다 세 나누기의 RMSE·MAPE."""
    x = transform(dq, kind)
    life = np.asarray(life, dtype=np.float64)
    split = np.asarray(split)
    train = split == "train"
    y = np.log10(life)
    rows = []
    for j, v in enumerate(voltage):
        col = x[:, j]
        if not np.all(np.isfinite(col)) or np.ptp(col[train]) == 0:
            rows.append({"voltage": v, "kind": kind})
            continue
        slope, intercept = np.polyfit(col[train], y[train], 1)
        pred = 10 ** (slope * col + intercept)
        row = {"voltage": v, "kind": kind, "slope": slope, "intercept": intercept}
        for key in SPLIT_KEYS:
            m = split == key
            if m.any():
                e = errors(life[m], pred[m])
                row[f"rmse_{key}"], row[f"mape_{key}"] = e["rmse"], e["mape"]
        rows.append(row)
    return pd.DataFrame(rows)


def pick(table: pd.DataFrame, by: str = "train", voltage: float | None = None) -> dict:
    """by 나누기의 RMSE가 가장 작은 칸, 또는 voltage에 가장 가까운 칸."""
    if voltage is not None:
        j = int(np.argmin(np.abs(table["voltage"].to_numpy() - voltage)))
    else:
        j = int(np.nanargmin(table[f"rmse_{by}"].to_numpy()))
    return {**table.iloc[j].to_dict(), "index": j}


def variance_model(dq: np.ndarray, life: np.ndarray, split: np.ndarray) -> dict:
    """비교 기준: log10 Var(ΔQ) 하나로 맞춘 단순 모델 (model.SimpleModel, 논문의 variance 모델과 같은 형태)."""
    X = pd.DataFrame({"logvar": np.log10(np.var(dq, axis=1))})
    life = np.asarray(life, dtype=np.float64)
    split = np.asarray(split)
    train = split == "train"
    model = SimpleModel().fit(X[train], life[train])
    pred = model.predict(X)
    out = {"r": float(np.corrcoef(X["logvar"], np.log10(life))[0, 1])}
    for key in SPLIT_KEYS:
        m = split == key
        if m.any():
            e = errors(life[m], pred[m])
            out[f"rmse_{key}"], out[f"mape_{key}"] = e["rmse"], e["mape"]
    return out


# ---------------------------------------------------------------- 수명별 무리

def life_groups(life: np.ndarray, k: int = 3) -> np.ndarray:
    """수명 순서로 셀을 k개 무리로 나눈다. 0 = 가장 짧은 수명 무리."""
    order = np.argsort(np.asarray(life, dtype=np.float64), kind="stable")
    groups = np.empty(len(order), dtype=int)
    for g, idx in enumerate(np.array_split(order, k)):
        groups[idx] = g
    return groups


def group_curves(values: np.ndarray, groups: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """무리마다 전압 칸별 평균과 25·75 백분위수."""
    out = {}
    for g in sorted(set(groups.tolist())):
        part = values[groups == g]
        p25, p75 = np.percentile(part, [25, 75], axis=0)
        out[g] = {"mean": part.mean(axis=0), "p25": p25, "p75": p75, "n": len(part)}
    return out


def moved(voltage: np.ndarray, dq_curve: np.ndarray, smooth: int = SMOOTH) -> np.ndarray:
    """ΔQ 곡선을 전압으로 미분한 값의 부호를 바꾼 것 = Δ(−dQ/dV), 단위 Ah/V.
    음수 = 100번째 사이클에서 그 전압 칸에 나온 용량이 줄었다, 양수 = 늘었다.
    ΔQ(V)는 이 값을 3.5V부터 V까지 더한 것이라 둘은 같은 정보를 다르게 보여 준다."""
    d = np.asarray(dq_curve, dtype=np.float64)
    if smooth > 1:
        pad = smooth // 2
        d = np.convolve(np.pad(d, pad, mode="edge"), np.ones(smooth) / smooth, mode="valid")
    return -np.gradient(d, voltage)


def depth_and_recovery(voltage: np.ndarray, dq_curve: np.ndarray) -> dict:
    """ΔQ가 가장 깊은 전압과 깊이, 끝(가장 낮은 전압)의 ΔQ, 가장 깊은 곳에서 끝까지 되찾은 비율."""
    d = np.asarray(dq_curve, dtype=np.float64)
    j = int(np.argmin(d))
    end = float(d[np.argmin(voltage)])
    deepest = float(d[j])
    return {"voltage": float(voltage[j]), "deepest_mah": deepest * 1000, "end_mah": end * 1000,
            "recovered": (1 - end / deepest) if deepest < 0 else np.nan}


# ---------------------------------------------------------------- dQ/dV 봉우리

def peak_metrics(voltage: np.ndarray, dqdv_early: np.ndarray, dqdv_late: np.ndarray,
                 window: tuple[float, float] = PEAK_WINDOW) -> pd.DataFrame:
    """셀마다 window 안 −dQ/dV 주 봉우리의 위치(V)와 높이(Ah/V), early → late 이동(mV)과 높이 변화(%)."""
    inside = (voltage >= window[0]) & (voltage <= window[1])
    v = voltage[inside]
    e, la = -np.asarray(dqdv_early)[:, inside], -np.asarray(dqdv_late)[:, inside]
    je, jl = np.argmax(e, axis=1), np.argmax(la, axis=1)
    he, hl = e[np.arange(len(e)), je], la[np.arange(len(la)), jl]
    return pd.DataFrame({"peak_v_early": v[je], "peak_v_late": v[jl], "shift_mv": (v[jl] - v[je]) * 1000,
                         "height_early": he, "height_late": hl, "height_change_pct": (hl / he - 1) * 100})


def count_peaks(voltage: np.ndarray, curve: np.ndarray, window: tuple[float, float] = COUNT_WINDOW,
                prominence: float = PROMINENCE) -> int:
    """window 안에서 −dQ/dV 봉우리 개수 (가장 큰 봉우리 높이의 prominence 이상 솟은 것만)."""
    from scipy.signal import find_peaks

    inside = (voltage >= window[0]) & (voltage <= window[1])
    c = -np.asarray(curve, dtype=np.float64)[inside]
    found, _ = find_peaks(c, prominence=prominence * c.max())
    return len(found)


def ir_at(summary: pd.DataFrame, cell_ids, cycle: int, reach: int = IR_REACH) -> np.ndarray:
    """셀마다 그 사이클 ±reach의 내부저항 중앙값(mΩ). b1의 빈 1번째 줄과 0·범위 밖 값은 뺀다. 값이 없으면 nan."""
    s = summary[~((summary["QD"] == 0) & (summary["IR"] == 0))]
    s = s[(s["IR"] > 0) & (s["QD"] > QD_RANGE[0]) & (s["QD"] < QD_RANGE[1])]
    groups = dict(tuple(s.groupby("cell_id")))
    out = []
    for cid in cell_ids:
        g = groups.get(cid)
        part = g.loc[(g["cycle"] - cycle).abs() <= reach, "IR"] if g is not None else []
        out.append(float(np.median(part)) * 1000 if len(part) else np.nan)
    return np.array(out, dtype=np.float64)


def ir_change(summary: pd.DataFrame, cell_ids, late: int = LATE, early: int = EARLY,
              reach: int = IR_REACH) -> np.ndarray:
    """셀마다 내부저항 변화(late − early, mΩ). 각 사이클 ±reach의 중앙값 (ir_at)."""
    return ir_at(summary, cell_ids, late, reach) - ir_at(summary, cell_ids, early, reach)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """둘 다 값이 있는 셀만으로 피어슨 상관. 셀이 3개 미만이거나 한쪽 값이 모두 같으면 nan."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])
