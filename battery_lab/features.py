"""수명 예측에 쓰는 특징 (초반 k사이클만 사용, 기본 k = 100).

논문(Severson 외, 2019)의 'variance 모델'과 'discharge 모델'을 참고해 직접 구현했다.
논문의 모델 코드는 공개돼 있지 않아서, 논문 설명과 다를 수 있는 부분은 docs/data_size.md에 적었다.

- logvar   log10 Var(ΔQ_k−10(V))       ← 단순 모델은 이것 하나만 쓴다
- logmin   log10 |min ΔQ_k−10(V)|
- logskew  log10 |왜도(ΔQ_k−10(V))|
- logkurt  log10 |첨도(ΔQ_k−10(V))|   (정규분포를 0으로 둔 초과 첨도)
- q2       2번째 사이클 방전 용량 (Ah)
- qmax_q2  2~k사이클 중 최대 방전 용량 − 2번째 사이클 방전 용량 (Ah)

방전 곡선이 깨끗하지 않으면(품질 점검 기준) 가까운 깨끗한 곡선을 대신 쓰고, 실제로 쓴 사이클을
late_used·early_used에 남긴다. k번째는 k−1·k−2처럼 **앞쪽으로만** 찾는다(k번째 이후 데이터를 쓰면 안 되므로).
10번째는 ±2사이클 안에서 찾는다(k ≥ 20이라 미래 데이터가 아니다).
처음엔 k번째도 ±2로 찾아서 k+1·k+2번째 곡선이 들어갈 수 있었다 (troubleshooting.md 33번).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .quality import CURVE_RANGE, FEATURE_BACKSTEP, QD_RANGE, load_curves, load_processed

FEATURES = ("logvar", "logmin", "logskew", "logkurt", "q2", "qmax_q2")
SIMPLE = ("logvar",)
REACH = 2  # 깨끗한 곡선을 찾을 때 앞뒤로 볼 사이클 수


def _moments(d: np.ndarray) -> tuple[float, float]:
    """(왜도, 초과 첨도)"""
    c = d - d.mean()
    m2 = np.mean(c ** 2)
    return float(np.mean(c ** 3) / m2 ** 1.5), float(np.mean(c ** 4) / m2 ** 2 - 3)


def is_clean(row: np.ndarray) -> bool:
    """빈칸·말이 안 되는 값·5mAh 넘는 되돌림이 없는 방전 곡선."""
    row = np.asarray(row, dtype=np.float64)
    return bool(row.size and np.all(np.isfinite(row)) and row.min() >= CURVE_RANGE[0]
                and row.max() <= CURVE_RANGE[1] and np.min(np.diff(row)) >= -FEATURE_BACKSTEP)


def find_row(curves: dict[str, np.ndarray], cycle: int, reach: int = REACH,
             forward: bool = True) -> tuple[int, int] | None:
    """cycle번째 곡선이 깨끗하면 그것을, 아니면 가까운 깨끗한 곡선. (행 번호, 쓴 사이클)

    forward=True면 ±reach 안에서(가까운 쪽, 같은 거리면 앞쪽 먼저), False면 cycle−reach ~ cycle 안에서만 찾는다.
    'k사이클까지만 본다'는 실험에서는 반드시 forward=False로 써야 한다.
    """
    rows = {int(c): j for j, c in enumerate(curves["cycle"])}
    steps = sorted(range(-reach, reach + 1), key=lambda s: (abs(s), s)) if forward else range(0, -reach - 1, -1)
    for step in steps:
        c = cycle + step
        if c in rows and is_clean(curves["Qdlin"][rows[c]]):
            return rows[c], c
    return None


def median_curve(curves: dict[str, np.ndarray], cycles: range) -> tuple[np.ndarray, list[int]] | None:
    """여러 사이클의 깨끗한 곡선을 전압 칸마다 중앙값으로 합친다. 곡선 하나의 잡음을 줄이려고 쓴다."""
    rows = {int(c): j for j, c in enumerate(curves["cycle"])}
    used = [c for c in cycles if c in rows and is_clean(curves["Qdlin"][rows[c]])]
    if not used:
        return None
    return np.median(np.stack([curves["Qdlin"][rows[c]].astype(np.float64) for c in used]), axis=0), used


def _curve(curves: dict[str, np.ndarray], cycle: int, window: int, trailing: bool) -> tuple[np.ndarray, int] | None:
    """window = 1이면 그 사이클 곡선(없으면 가까운 곡선: trailing이면 앞쪽으로만), 아니면 여러 사이클의 중앙값 곡선."""
    if window == 1:
        found = find_row(curves, cycle, forward=not trailing)
        return (curves["Qdlin"][found[0]].astype(np.float64), found[1]) if found else None
    span = range(cycle - window + 1, cycle + 1) if trailing else range(cycle - window // 2, cycle + window // 2 + 1)
    merged = median_curve(curves, span)
    return (merged[0], int(np.median(merged[1]))) if merged else None


def cell_features(curves: dict[str, np.ndarray], summary: pd.DataFrame, late: int = 100, early: int = 10,
                  window: int = 1) -> dict:
    """window > 1이면 late는 직전 window개 사이클(late−window+1 ~ late, 미래 데이터를 쓰지 않음),
    early는 앞뒤 window개 사이클(예: 5 → 8~12)의 중앙값 곡선으로 ΔQ를 만든다."""
    out = {**dict.fromkeys(FEATURES, np.nan), "late_used": np.nan, "early_used": np.nan}
    a, b = _curve(curves, late, window, trailing=True), _curve(curves, early, window, trailing=False)
    if a and b:
        out["late_used"], out["early_used"] = a[1], b[1]
        d = a[0] - b[0]
        if d.size > 3 and np.var(d) > 0:
            skew, kurt = _moments(d)
            with np.errstate(divide="ignore"):
                values = {"logvar": np.log10(np.var(d)), "logmin": np.log10(abs(d.min())),
                          "logskew": np.log10(abs(skew)), "logkurt": np.log10(abs(kurt))}
            # log(0) = −∞ 같은 값은 조용히 쓰지 않고 비워 둔다 → sizes.run이 어느 셀인지 알려 준다
            out.update({k: (float(v) if np.isfinite(v) else np.nan) for k, v in values.items()})
    s = summary[(summary["cycle"] >= 2) & (summary["cycle"] <= late)]
    s = s[(s["QD"] > QD_RANGE[0]) & (s["QD"] < QD_RANGE[1])]
    q2 = s.loc[s["cycle"] == 2, "QD"]
    if len(q2):
        out["q2"] = float(q2.iloc[0])
        out["qmax_q2"] = float(s["QD"].max() - q2.iloc[0])
    return out


def build_many(folder: Path, lates: tuple[int, ...], early: int = 10, window: int = 1) -> dict[int, pd.DataFrame]:
    """사이클 k마다 특징 표. 곡선 파일은 셀마다 한 번만 읽는다."""
    cells, summary = load_processed(folder)
    groups = dict(tuple(summary.groupby("cell_id")))
    rows = {k: [] for k in lates}
    for meta in cells.to_dict("records"):
        curves = load_curves(folder, meta["cell_id"])
        base = {"cell_id": meta["cell_id"], "batch": meta["batch"], "policy": meta["policy"],
                "split": meta["split"], "cycle_life": float(meta["cycle_life"])}
        for k in lates:
            rows[k].append({**base, **cell_features(curves, groups[meta["cell_id"]], k, early, window)})
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def build(folder: Path) -> pd.DataFrame:
    """셀마다 100사이클 특징을 계산해 data/processed/features.csv로 저장한다."""
    df = build_many(folder, (100,))[100]
    df.to_csv(folder / "features.csv", index=False, encoding="utf-8")
    return df
