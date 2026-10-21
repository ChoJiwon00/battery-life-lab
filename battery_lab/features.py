"""수명 예측에 쓰는 특징 (초반 100사이클만 사용).

논문(Severson 외, 2019)의 'variance 모델'과 'discharge 모델'을 참고해 직접 구현했다.
논문의 모델 코드는 공개돼 있지 않아서, 논문 설명과 다를 수 있는 부분은 docs/data_size.md에 적었다.

- logvar   log10 Var(ΔQ₁₀₀₋₁₀(V))       ← 단순 모델은 이것 하나만 쓴다
- logmin   log10 |min ΔQ₁₀₀₋₁₀(V)|
- logskew  log10 |왜도(ΔQ₁₀₀₋₁₀(V))|
- logkurt  log10 |첨도(ΔQ₁₀₀₋₁₀(V))|   (정규분포를 0으로 둔 초과 첨도)
- q2       2번째 사이클 방전 용량 (Ah)
- qmax_q2  2~100사이클 중 최대 방전 용량 − 2번째 사이클 방전 용량 (Ah)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .quality import QD_RANGE, delta_q, load_curves, load_processed

FEATURES = ("logvar", "logmin", "logskew", "logkurt", "q2", "qmax_q2")
SIMPLE = ("logvar",)


def _moments(d: np.ndarray) -> tuple[float, float]:
    """(왜도, 초과 첨도)"""
    c = d - d.mean()
    m2 = np.mean(c ** 2)
    return float(np.mean(c ** 3) / m2 ** 1.5), float(np.mean(c ** 4) / m2 ** 2 - 3)


def cell_features(curves: dict[str, np.ndarray], summary: pd.DataFrame) -> dict[str, float]:
    d = delta_q(curves, 100, 10)
    out = dict.fromkeys(FEATURES, np.nan)
    if d is not None:
        d = d[np.isfinite(d)]
        if d.size > 3 and np.var(d) > 0:
            skew, kurt = _moments(d)
            with np.errstate(divide="ignore"):
                values = {"logvar": np.log10(np.var(d)), "logmin": np.log10(abs(d.min())),
                          "logskew": np.log10(abs(skew)), "logkurt": np.log10(abs(kurt))}
            # log(0) = −∞ 같은 값은 조용히 쓰지 않고 비워 둔다 → sizes.run이 어느 셀인지 알려 준다
            out.update({k: (float(v) if np.isfinite(v) else np.nan) for k, v in values.items()})
    s = summary[(summary["cycle"] >= 2) & (summary["cycle"] <= 100)]
    s = s[(s["QD"] > QD_RANGE[0]) & (s["QD"] < QD_RANGE[1])]
    q2 = s.loc[s["cycle"] == 2, "QD"]
    if len(q2):
        out["q2"] = float(q2.iloc[0])
        out["qmax_q2"] = float(s["QD"].max() - q2.iloc[0])
    return out


def build(folder: Path) -> pd.DataFrame:
    """셀마다 특징을 계산해 data/processed/features.csv로 저장한다."""
    cells, summary = load_processed(folder)
    groups = dict(tuple(summary.groupby("cell_id")))
    rows = []
    for meta in cells.to_dict("records"):
        f = cell_features(load_curves(folder, meta["cell_id"]), groups[meta["cell_id"]])
        rows.append({"cell_id": meta["cell_id"], "batch": meta["batch"], "policy": meta["policy"],
                     "split": meta["split"], "cycle_life": float(meta["cycle_life"]), **f})
    df = pd.DataFrame(rows)
    df.to_csv(folder / "features.csv", index=False, encoding="utf-8")
    return df
