"""급속충전 조건(policy)과 수명 (관찰 연구, 셀 개발 관점).

정책 표기 "C1(Q1%)-C2": SOC(충전 상태) 0→Q1%는 C1, Q1→80%는 C2 전류로 충전하고, 80%부터는 모든 셀이 1C CC-CV.
3배치(b3)는 끝에 "-newstructure"가 붙는다. C-rate 1C = 1시간에 공칭 용량(1.1Ah)을 채우는 전류.

이 모듈이 하는 일 (작은 순수 함수들)
- parse_policy: 문자열 → C1, Q1, C2 (+ newstructure 표시, 끝의 C가 빠진 표기 표시)
- window_minutes / window_rate: 표기대로라면 SOC 구간을 지나는 데 걸리는 시간과 평균 C-rate (정전압 구간은 무시)
- charge_profile / raw_cell: 원시 기록에서 실제로 SOC 40%·80%에 닿은 시간과 실제 전류
- cell_table: 셀마다 한 줄 (정책, 계산한 시간, 측정한 시간, 실측 전류, 수명)
- time_check, life_vs_time, effects, leave_out_shortest, replicates, across_batches, frontier: 질문별 계산 (analyze가 한 번에)

주의: 이 데이터는 실험 설계상 배치마다 충전 조건의 종류가 다르다 (b1은 9~13분, b2·b3는 대부분 10분).
'조건의 효과'와 '배치의 효과'가 섞여 있어서, 여기 숫자는 모두 관찰(함께 나타남)이지 인과가 아니다.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .quality import NOMINAL_AH, load_processed

TARGET_SOC = 80.0     # 급속충전이 끝나는 SOC(%)
SPLIT_SOC = 40.0      # 낮은 SOC 구간 0~40%, 높은 SOC 구간 40~80%
# 원시 기록에서 '80%에 닿음'으로 볼 SOC. 급속충전이 79.9999%에서 끝나는 사이클이 있어서, 80.00%로 잡으면
# 그 뒤 쉬는 시간(b2는 약 5분)까지 급속충전 시간에 들어간다 (시행착오 기록). 79.9→80%는 1초 남짓이라 무시한다.
END_SOC = 79.9
RAW_TOL = 0.5         # 분. 원시 기록으로 잰 0→80% 시간과 summary 충전 시간이 이보다 다르면 표시
EARLY = (2, 100)      # 충전 시간 중앙값을 잴 사이클 범위
CHARGE_ON = 0.05      # 이보다 센 + 전류(C-rate)면 충전 중으로 본다
V_HIT = 3.595         # 급속충전 중 이 전압을 넘으면 상한(3.6V)에 닿았다고 본다
TIME_TOL = 0.3        # 분. 계산한 시간과 측정한 시간이 이보다 많이 다르면 표시
RATE_TOL = 0.03       # 실제 전류가 표기와 3% 넘게 다르면 표시
TIME_BINS = (8.5, 9.5, 10.5, 11.5, 12.5, 13.5)  # 분. 프런티어(관찰된 최고) 구간
BOOT = 2000           # 부트스트랩 반복 수
PERM = 2000           # 순열(섞기) 반복 수
SEED = 2026

POLICY = re.compile(r"^(?P<c1>\d+(?:\.\d+)?)C\((?P<q1>\d+(?:\.\d+)?)%\)-(?P<c2>\d+(?:\.\d+)?)(?P<unit>C?)"
                    r"(?P<ns>-newstructure)?$")
NEW = "-newstructure"


# ---------------------------------------------------------------- 표기 읽기와 '이상적인' 충전 시간

def parse_policy(text: str) -> dict | None:
    """"5.4C(40%)-3.6C" → {c1: 5.4, q1: 40, c2: 3.6, ...}. 읽을 수 없는 형식이면 None.

    끝의 C가 빠진 표기("4C(31%)-5")도 숫자는 읽되 unit_missing으로 표시한다.
    Q1이 80% 이상이면 C2는 쓰이지 않는다 (한 단계 충전, single_step).
    """
    m = POLICY.match(str(text).strip())
    if m is None:
        return None
    c1, q1, c2 = float(m["c1"]), float(m["q1"]), float(m["c2"])
    if c1 <= 0 or c2 <= 0 or not 0 < q1 <= 100:
        return None
    return {"c1": c1, "q1": q1, "c2": c2, "newstructure": bool(m["ns"]), "unit_missing": not m["unit"],
            "single_step": q1 >= TARGET_SOC or c1 == c2}


def base_policy(text: str) -> str:
    """배치끼리 같은 표기를 비교하려고 "-newstructure"를 뗀 이름."""
    return str(text).removesuffix(NEW)


def window_minutes(c1: float, q1: float, c2: float, lo: float = 0.0, hi: float = TARGET_SOC) -> float:
    """표기대로 SOC lo→hi%를 채우는 데 걸리는 시간(분). 정전압(CV)으로 전류가 줄어드는 것은 무시한다.

    cC로 x%를 채우는 시간 = x/100/c 시간 (1C는 1시간에 100%).
    """
    first = max(0.0, min(hi, q1) - lo)    # C1 구간과 겹치는 SOC 폭
    second = max(0.0, hi - max(lo, q1))   # C2 구간과 겹치는 SOC 폭
    return (first / c1 + second / c2) / 100 * 60


def window_rate(c1: float, q1: float, c2: float, lo: float, hi: float) -> float:
    """표기대로라면 SOC lo→hi% 구간의 평균 C-rate (구간 폭 ÷ 걸린 시간)."""
    return (hi - lo) / 100 / (window_minutes(c1, q1, c2, lo, hi) / 60)


# ---------------------------------------------------------------- 원시 기록에서 실제 충전 재기

def _reach(t: np.ndarray, soc: np.ndarray, level: float) -> float:
    """SOC가 level에 처음 닿은 시각 (앞뒤 두 점 사이를 직선으로 이어 계산)."""
    hit = np.flatnonzero(soc >= level)
    if not hit.size:
        return np.nan
    k = hit[0]
    if k == 0 or soc[k] == soc[k - 1]:
        return float(t[k])
    return float(t[k - 1] + (level - soc[k - 1]) / (soc[k] - soc[k - 1]) * (t[k] - t[k - 1]))


def _median_or_nan(x: np.ndarray, at_least: int = 3) -> float:
    return float(np.median(x)) if x.size >= at_least else np.nan


def charge_profile(t: np.ndarray, v: np.ndarray, i: np.ndarray, qc: np.ndarray, q1: float,
                   nominal: float = NOMINAL_AH) -> dict:
    """원시 기록 한 사이클 → 충전 시작부터 SOC 40%·80%에 닿은 시간(분), 구간 평균 C-rate, 실제 C1·C2, 최고 전압.

    i는 C-rate 단위(+ 충전), qc는 충전 용량(Ah). SOC = 충전 시작 뒤 넣은 용량 ÷ 공칭 용량(1.1Ah).
    """
    on = np.flatnonzero(i > CHARGE_ON)
    if not on.size:
        return {}
    k0 = on[0]
    t, v, i = t[k0:] - t[k0], v[k0:], i[k0:]
    soc = (qc[k0:] - qc[k0]) / nominal * 100
    t_low, t80 = _reach(t, soc, SPLIT_SOC), _reach(t, soc, END_SOC)
    charging = i > CHARGE_ON
    fast = charging & (soc < END_SOC)
    s1 = charging & (soc > max(0.5, q1 * 0.25)) & (soc < min(q1, TARGET_SOC) - 0.5)   # C1 단계 가운데
    s2 = charging & (soc > q1 + 0.5) & (soc < TARGET_SOC - 0.5)                        # C2 단계 가운데
    return {
        "t40": t_low, "t80": t80,
        "low_rate": SPLIT_SOC / 100 / (t_low / 60),
        "high_rate": (END_SOC - SPLIT_SOC) / 100 / ((t80 - t_low) / 60),
        "c1_real": _median_or_nan(i[s1]),
        "c2_real": _median_or_nan(i[s2]) if q1 < TARGET_SOC else np.nan,
        "v_max": float(v[fast].max()) if fast.any() else np.nan,
        "cycle_minutes": float(np.nanmax(t)),
    }


def raw_cell(folder: Path, cell_id: str, q1: float, rows: tuple[int, int] = (EARLY[0] - 1, EARLY[1] - 1)) -> dict:
    """원시 기록(raw/<셀>.npz)에서 EARLY 범위 사이클들의 charge_profile 중앙값.

    raw_index는 0부터 세는 줄 번호라 사이클 2~100은 줄 1~99다 (원시 기록은 대개 10·100번째 사이클만 남아 있다).
    """
    path = folder / "raw" / f"{cell_id}.npz"
    if not path.exists():
        return {"raw_cycles": 0}
    with np.load(path) as z:
        idx, o = z["raw_index"], z["raw_offsets"]
        t, v, i, qc = z["raw_t"], z["raw_V"], z["raw_I"], z["raw_Qc"]
    found = []
    for k, row in enumerate(idx):
        if rows[0] <= row <= rows[1]:
            p = charge_profile(t[o[k]:o[k + 1]], v[o[k]:o[k + 1]], i[o[k]:o[k + 1]], qc[o[k]:o[k + 1]], q1)
            if p:
                found.append(p)
    if not found:
        return {"raw_cycles": 0}
    df = pd.DataFrame(found)
    out = {k: float(df[k].median()) for k in df.columns if k != "v_max"}
    out["v_hit"] = bool((df["v_max"] >= V_HIT).any())
    out["raw_cycles"] = len(found)
    return out


# ---------------------------------------------------------------- 셀마다 한 줄

def measured_minutes(summary: pd.DataFrame, cycles: tuple[int, int] = EARLY) -> pd.Series:
    """셀마다 사이클 2~100 충전 시간(chargetime, 분)의 중앙값. b1의 1번째 사이클 빈 자리는 뺀다.
    중앙값이라 시험이 멈췄던 사이클의 아주 긴 값(수백 분)에 흔들리지 않는다."""
    s = summary[~((summary["QD"] == 0) & (summary["IR"] == 0))]
    s = s[(s["cycle"] >= cycles[0]) & (s["cycle"] <= cycles[1])]
    return s.groupby("cell_id")["chargetime"].median()


def cell_table(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    """셀마다 정책·계산한 시간·측정한 시간·실측 전류·수명. 읽지 못한 정책 표기는 따로 돌려준다."""
    cells, summary = load_processed(folder)
    measured = measured_minutes(summary)
    rows, unparsed = [], []
    for meta in cells.to_dict("records"):
        p = parse_policy(meta["policy"])
        if p is None:
            unparsed.append(meta["policy"])
            continue
        life = float(meta["cycle_life"]) if str(meta["cycle_life"]).strip() else np.nan
        c1, q1, c2 = p["c1"], p["q1"], p["c2"]
        rows.append({
            "cell_id": meta["cell_id"], "batch": meta["batch"], "policy": meta["policy"],
            "base_policy": base_policy(meta["policy"]), "group": f"{meta['batch']} · {meta['policy']}",
            **p,
            "ideal_min": window_minutes(c1, q1, c2),
            "nominal_low": window_rate(c1, q1, c2, 0, SPLIT_SOC),
            "nominal_high": window_rate(c1, q1, c2, SPLIT_SOC, TARGET_SOC),
            "measured_min": float(measured.get(meta["cell_id"], np.nan)),
            **raw_cell(folder, meta["cell_id"], q1),
            "cycle_life": life, "log_life": np.log10(life) if life > 0 else np.nan,
        })
    table = pd.DataFrame(rows)
    table["shift"] = table["high_rate"] - table["low_rate"]               # 실측: 뒤쪽(높은 SOC) 쏠림
    table["nominal_shift"] = table["nominal_high"] - table["nominal_low"]  # 표기: 뒤쪽 쏠림
    return table, sorted(set(unparsed))


# ---------------------------------------------------------------- 질문 1 · 표기 vs 측정

def _rate_note(label: float, real: float, v_hit: bool) -> str:
    if not (np.isfinite(label) and np.isfinite(real)) or abs(real / label - 1) <= RATE_TOL:
        return ""
    if real > label:
        return "표기보다 센 전류"
    return "3.6V 상한에 닿아 전류가 줄어듦 (정전압)" if v_hit else "표기보다 약한 전류"


def time_check(table: pd.DataFrame) -> pd.DataFrame:
    """계산한 0→80% 시간과 측정한 충전 시간 비교. 차이가 TIME_TOL보다 크거나 실제 전류가 표기와 다르면 이유를 붙인다."""
    out = table[["cell_id", "batch", "policy", "c1", "c2", "q1", "ideal_min", "measured_min", "t80",
                 "c1_real", "c2_real", "v_hit"]].copy()
    out["diff_min"] = out["measured_min"] - out["ideal_min"]
    out["raw_gap"] = out["t80"] - out["measured_min"]  # 원시 기록으로 잰 시간 − summary 충전 시간 (서로 맞는지 점검)
    out["raw_disagree"] = out["raw_gap"].abs() > RAW_TOL
    # 표기 대신 실제로 잰 전류를 넣어 다시 계산한 시간 (설명이 맞으면 측정 시간에 가까워진다)
    c2_used = out["c2_real"].where(out["q1"] < TARGET_SOC, out["c1_real"]).fillna(out["c2"])
    out["real_ideal_min"] = [window_minutes(a, q, b) for a, q, b in
                             zip(out["c1_real"].fillna(out["c1"]), out["q1"], c2_used)]
    reasons = []
    for r in out.itertuples():
        notes = [n for n in (_rate_note(r.c1, r.c1_real, bool(r.v_hit)),
                             _rate_note(r.c2, r.c2_real, bool(r.v_hit)) if r.q1 < TARGET_SOC else "") if n]
        if not notes and abs(r.diff_min) > TIME_TOL:
            notes = ["그 밖의 이유"]
        reasons.append(" · ".join(dict.fromkeys(notes)))
    out["reason"] = reasons
    out["flagged"] = (out["diff_min"].abs() > TIME_TOL) | (out["reason"] != "")
    return out


# ---------------------------------------------------------------- 질문 2 · 충전 시간과 수명

def spearman(x, y) -> float:
    """순위 상관(스피어만 ρ): 값 대신 순위로 계산해 극단값에 덜 흔들린다."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan
    return float(np.corrcoef(df["x"].rank(), df["y"].rank())[0, 1])


def life_vs_time(table: pd.DataFrame) -> pd.DataFrame:
    """전체·배치별: 측정한 충전 시간의 범위와, 충전 시간과 수명의 순위 상관."""
    rows = []
    for scope, g in [("전체", table), *[(b, g) for b, g in table.groupby("batch")]]:
        rows.append({"scope": scope, "cells": len(g), "policies": g["policy"].nunique(),
                     "time_min": g["measured_min"].min(), "time_max": g["measured_min"].max(),
                     "life_median": g["cycle_life"].median(), "life_min": g["cycle_life"].min(),
                     "life_max": g["cycle_life"].max(),
                     "cycle_minutes": g["cycle_minutes"].median() if "cycle_minutes" in g else np.nan,
                     "rho_time": spearman(g["measured_min"], g["log_life"]),
                     "rho_shift": spearman(g["shift"], g["log_life"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 질문 3 · 낮은 SOC 전류 vs 높은 SOC 전류

def _design(df: pd.DataFrame, terms: tuple[str, ...], batch_effects: bool) -> np.ndarray:
    cols = [np.ones(len(df))] + [df[t].to_numpy(float) for t in terms]
    if batch_effects:  # 배치마다 다른 절편 (배치 보정)
        cols += [(df["batch"] == b).to_numpy(float) for b in sorted(df["batch"].unique())[1:]]
    return np.column_stack(cols)


def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _pct(coef: float) -> float:
    """log10 수명의 기울기 → '+1C마다 수명 몇 % 변화'."""
    return (10 ** coef - 1) * 100


def effects(table: pd.DataFrame, terms: tuple[str, ...], batches=None, boot: int = BOOT, seed: int = SEED) -> dict:
    """log10(수명) = a + Σ b·(항) (+ 여러 배치면 배치마다 다른 절편)을 맞춘다.

    90% 범위는 같은 조건(배치·정책)의 셀을 한 덩어리로 다시 뽑는 부트스트랩(cluster bootstrap)으로 구한다.
    같은 조건의 셀끼리는 닮아서, 셀 하나하나를 독립으로 보면 범위가 실제보다 좁게 나온다.
    """
    df = table if batches is None else table[table["batch"].isin(batches)]
    df = df.dropna(subset=[*terms, "log_life"]).reset_index(drop=True)
    multi = df["batch"].nunique() > 1
    X, y = _design(df, terms, multi), df["log_life"].to_numpy()
    coef = _fit(X, y)
    resid = y - X @ coef
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan

    rng = np.random.default_rng(seed)
    # 배치 안에서 조건 덩어리를 다시 뽑는다. 줄 번호는 df 전체 기준이어야 한다
    # (배치별로 나눈 표에서 .indices를 쓰면 그 작은 표 기준 번호가 나와 다른 배치 셀을 가리킨다: 시행착오 기록)
    where = df.groupby(["batch", "group"]).indices
    blocks = [[where[key] for key in where if key[0] == b] for b in sorted(df["batch"].unique())]
    draws = []
    for _ in range(boot):
        idx = np.concatenate([blk[j] for blk in blocks for j in rng.integers(0, len(blk), len(blk))])
        Xb = X[idx]
        if np.linalg.matrix_rank(Xb) < X.shape[1]:
            continue  # 같은 조건만 뽑혀 기울기를 정할 수 없는 경우는 건너뛴다
        draws.append(_fit(Xb, y[idx])[1:1 + len(terms)])
    draws = np.array(draws).reshape(-1, len(terms))
    out = {"cells": len(df), "groups": int(df["group"].nunique()), "batches": sorted(df["batch"].unique()),
           "r2": float(r2), "boot_used": len(draws), "terms": {}}
    for k, t in enumerate(terms):
        lo, hi = np.percentile(draws[:, k], [5, 95]) if len(draws) else (np.nan, np.nan)
        out["terms"][t] = {"coef": float(coef[1 + k]), "lo": float(lo), "hi": float(hi),
                           "pct": _pct(coef[1 + k]), "pct_lo": _pct(lo), "pct_hi": _pct(hi),
                           "clear": bool(lo > 0 or hi < 0)}  # 90% 범위가 0을 포함하지 않음
    if len(terms) == 2 and len(draws):
        # (두 번째 항) − (첫 번째 항). %로 바꾸면 '두 번째 항 +1C가 첫 번째 항 +1C보다 수명을 몇 % 더 바꾸나'
        gap = draws[:, 1] - draws[:, 0]
        out["second_worse_share"] = float((gap < 0).mean())
        out["gap_pct"] = _pct(coef[2] - coef[1])
        out["gap_pct_lo"], out["gap_pct_hi"] = (_pct(x) for x in np.percentile(gap, [5, 95]))
    return out


def leave_out_shortest(table: pd.DataFrame, batch: str = "b2", n: int = 2, boot: int = BOOT,
                       seed: int = SEED) -> dict:
    """batch에서 수명이 가장 짧은 n개 셀을 빼고 다시 맞춘다 (민감도 점검: 몇 개 셀이 결론을 만드는지).

    pooled = 두 구간 전류를 함께 넣은 전체(배치 보정) 모델, within = 그 배치의 '뒤쪽 쏠림' 모델.
    """
    if batch not in set(table["batch"]):
        return {}
    gone = table[table["batch"] == batch].nsmallest(n, "cycle_life")
    rest = table.drop(gone.index)
    return {"batch": batch, "cells": gone["cell_id"].tolist(), "policies": gone["policy"].tolist(),
            "pooled": effects(rest, ("low_rate", "high_rate"), None, boot, seed),
            "within": effects(rest, ("shift",), (batch,), boot, seed)}


def soc_effects(table: pd.DataFrame, boot: int = BOOT, seed: int = SEED) -> list[dict]:
    """질문 3에 쓰는 모델들. 시간이 바뀌는 b1과 전체(배치 보정)는 두 구간 전류를 함께,
    시간이 거의 10분으로 고정된 b2·b3는 '뒤쪽 쏠림(높은 SOC 전류 − 낮은 SOC 전류)' 하나로 본다.
    (시간이 고정되면 한쪽을 세게 하면 다른 쪽은 약해져 두 전류를 따로 떼어 볼 수 없다.)"""
    specs = [("b1", ("b1",), ("low_rate", "high_rate")),
             ("전체 (배치 보정)", None, ("low_rate", "high_rate")),
             ("b2 (시간 거의 고정)", ("b2",), ("shift",)),
             ("b3 (시간 거의 고정)", ("b3",), ("shift",))]
    specs = [s for s in specs if s[1] is None or set(s[1]) <= set(table["batch"])]
    rows = []
    for name, batches, terms in specs:
        for source, cols in (("실측", terms), ("표기", tuple("nominal_" + t.removesuffix("_rate") if t != "shift"
                                                             else "nominal_shift" for t in terms))):
            fit = effects(table, cols, batches, boot, seed)
            fit.update({"name": name, "source": source, "kind": "two" if len(terms) == 2 else "shift"})
            rows.append(fit)
    return rows


# ---------------------------------------------------------------- 질문 4 · 같은 조건 반복 셀

def _ss(y: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(조건 사이 제곱합, 전체 제곱합)."""
    grand = y.mean()
    between = sum(len(y[labels == g]) * (y[labels == g].mean() - grand) ** 2 for g in np.unique(labels))
    return float(between), float(((y - grand) ** 2).sum())


def eta2(y, labels) -> float:
    """η² = 조건 사이 제곱합 ÷ 전체 제곱합: 수명 차이 중 '어느 조건이냐'로 설명되는 몫 (0~1)."""
    y, labels = np.asarray(y, float), np.asarray(labels)
    between, total = _ss(y, labels)
    return between / total if total > 0 else np.nan


def variance_share(y, labels, perm: int = PERM, seed: int = SEED) -> dict:
    """η²와, 조건 이름을 무작위로 섞었을 때의 η² (조건이 아무 상관 없어도 나오는 값)."""
    y, labels = np.asarray(y, float), np.asarray(labels)
    rng = np.random.default_rng(seed)
    obs = eta2(y, labels)
    null = np.array([eta2(rng.permutation(y), labels) for _ in range(perm)])
    k, n = len(np.unique(labels)), len(y)
    within = np.sqrt((1 - obs) * ((y - y.mean()) ** 2).sum() / (n - k)) if n > k else np.nan
    means = pd.Series(y).groupby(labels).mean()
    return {"cells": n, "groups": k, "eta2": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p": float((null >= obs).mean()),
            "within_sd": float(within),                                          # 조건 안 log10 수명의 표준편차
            "between_sd": float(means.std(ddof=1)) if k > 1 else np.nan}         # 조건 평균끼리의 표준편차


def replicates(table: pd.DataFrame, perm: int = PERM, seed: int = SEED) -> dict:
    """같은 배치·같은 정책 셀이 2개 이상인 조건만 모아, 조건 안 차이와 조건 사이 차이를 비교한다."""
    rep = table[table.groupby("group")["group"].transform("size") >= 2].dropna(subset=["log_life"])
    groups = (rep.groupby("group")
                 .agg(batch=("batch", "first"), policy=("policy", "first"), cells=("cell_id", "size"),
                      life_min=("cycle_life", "min"), life_median=("cycle_life", "median"),
                      life_max=("cycle_life", "max"), lives=("cycle_life", lambda s: sorted(s.astype(int))))
                 .reset_index())
    groups["ratio"] = groups["life_max"] / groups["life_min"]
    out = {"all": variance_share(rep["log_life"], rep["group"], perm, seed),
           "batch_only": eta2(rep["log_life"], rep["batch"]), "groups": groups, "by_batch": {}}
    for b, g in rep.groupby("batch"):
        if g["group"].nunique() >= 2:
            out["by_batch"][b] = variance_share(g["log_life"], g["group"], perm, seed)
    return out


def across_batches(table: pd.DataFrame) -> pd.DataFrame:
    """같은 표기('-newstructure'를 뗀 이름)가 여러 배치에 있는 경우, 배치마다 수명과 실제 전류."""
    multi = table.groupby("base_policy")["batch"].nunique()
    keep = table[table["base_policy"].isin(multi[multi >= 2].index)]
    return (keep.groupby(["base_policy", "batch"])
                .agg(cells=("cell_id", "size"), life_median=("cycle_life", "median"),
                     life_min=("cycle_life", "min"), life_max=("cycle_life", "max"),
                     c1_label=("c1", "first"), c1_real=("c1_real", "median"), minutes=("measured_min", "median"))
                .reset_index())


# ---------------------------------------------------------------- 질문 5 · 빠르면서 오래 간 조건 (관찰된 최고)

def group_table(table: pd.DataFrame) -> pd.DataFrame:
    """배치·정책 조건마다 한 줄: 셀 수, 수명 중앙값·범위, 측정한 충전 시간, 실측 전류."""
    return (table.groupby("group")
                 .agg(batch=("batch", "first"), policy=("policy", "first"), cells=("cell_id", "size"),
                      life_median=("cycle_life", "median"), life_min=("cycle_life", "min"),
                      life_max=("cycle_life", "max"), minutes=("measured_min", "median"),
                      ideal=("ideal_min", "first"), low_rate=("low_rate", "median"),
                      high_rate=("high_rate", "median"))
                 .reset_index())


def frontier(groups: pd.DataFrame, bins=TIME_BINS) -> pd.DataFrame:
    """충전 시간 구간마다 관찰된 최고 수명 (최적화한 결과가 아니라 시험한 조건 중 가장 좋았던 것).

    best_*: 구간 안에서 수명 중앙값이 가장 긴 조건, so_far: 이 구간까지(이 시간 이하) 본 것 중 최고.
    """
    rows, so_far = [], -np.inf
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = groups[(groups["minutes"] >= lo) & (groups["minutes"] < hi)]
        if g.empty:
            rows.append({"lo": lo, "hi": hi, "groups": 0, "cells": 0})
            continue
        best = g.loc[g["life_median"].idxmax()]
        so_far = max(so_far, best["life_median"])
        rows.append({"lo": lo, "hi": hi, "groups": len(g), "cells": int(g["cells"].sum()),
                     "batches": " ".join(sorted(g["batch"].unique())), "best_group": best["group"],
                     "best_batch": best["batch"], "best_policy": best["policy"], "best_cells": int(best["cells"]),
                     "best_life": float(best["life_median"]), "best_minutes": float(best["minutes"]),
                     "best_cell_life": float(g["life_max"].max()), "so_far": float(so_far)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 한 번에

def analyze(table: pd.DataFrame, boot: int = BOOT, perm: int = PERM, seed: int = SEED) -> dict:
    groups = group_table(table)
    return {
        "time_check": time_check(table),
        "life_vs_time": life_vs_time(table),
        "effects": soc_effects(table, boot, seed),
        "leave_out": leave_out_shortest(table, "b2", 2, boot, seed),
        "replicates": replicates(table, perm, seed),
        "across": across_batches(table),
        "groups": groups,
        "frontier": frontier(groups),
    }
