"""판단 실험: '불량(단수명) 1개를 놓치는 손해 = 정상(장수명) 셀 몇 개를 버리는 손해'를 바꾸면 결과가 어떻게 달라질까?

불량 셀 선별 실험(screening.py)은 이 값을 5로 가정했다. 이 값은 정답이 없는 공정의 판단이라,
사용자가 직접 정하고 비교해 보도록 따로 뺀 명령이다. 원래 문서(docs/screening.md)는 건드리지 않는다.

    python -m battery_lab judgment --miss-cost 10
    python -m battery_lab judgment --miss-cost 2 10 30

무작위 나누기마다 같은 학습·평가 셀과 같은 모델 확률을 쓰고 손해 비율만 바꾼다 → 차이는 손해 비율 때문이다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import screening as sc
from .sources import PROCESSED
from .sizes import n_train

DEFAULT_COST = sc.MISS_COST
REPEATS = 100


def cost_runs(f: pd.DataFrame, costs: tuple[float, ...], repeats: int = REPEATS, seed: int = sc.SEED,
              kind: str = sc.OPERATING_KIND) -> pd.DataFrame:
    """반복마다: 학습 셀로 모델을 맞추고, 손해 비율마다 학습 셀만 보고 기준값을 정해 평가 셀에 쓴다."""
    rng = np.random.default_rng(seed)
    y = sc.labels(f["cycle_life"])
    k_train = n_train(len(f))
    rows = []
    for rep in range(repeats):
        train, test = sc.stratified_split(y, k_train, rng)
        p = sc.fit_predict(kind, f.iloc[train], y[train], f.iloc[test], seed=rep)
        oof = sc.oof_proba(kind, f.iloc[train], y[train], seed=rep)
        for cost in costs:
            cut = sc.choose_cut(y[train], oof, miss_cost=cost)
            rows.append({"miss_cost": cost, "repeat": rep, "cut": cut, **sc.confusion(y[test], p, cut)})
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby("miss_cost")
    q = lambda col, p: g[col].quantile(p)  # noqa: E731
    return pd.DataFrame({
        "기준값 중앙값": g["cut"].median(),
        "놓친 단수명 비율 중앙값": g["miss_rate"].median(),
        "놓친 단수명 90% 끝": q("miss_rate", 0.9),
        "버린 장수명 비율 중앙값": g["discard_rate"].median(),
        "버린 장수명 90% 끝": q("discard_rate", 0.9),
        "정확도 중앙값": g["accuracy"].median(),
    })


def run(costs: tuple[float, ...], processed: Path = PROCESSED, repeats: int = REPEATS, progress=print) -> pd.DataFrame:
    progress("5사이클 특징 계산 중...")
    f = sc.build_features(processed, cycles=(5,))[5]
    compare = tuple(sorted({DEFAULT_COST, *costs}))
    progress(f"무작위 나누기 {repeats}번 × 손해 비율 {', '.join(f'{c:g}' for c in compare)} ...")
    runs = cost_runs(f, compare, repeats=repeats)
    runs.to_csv(processed / "judgment_cost_runs.csv", index=False)
    return summarize(runs)


def format_table(table: pd.DataFrame, default: float = DEFAULT_COST) -> str:
    lines = ["| 놓침 1개 = 버림 몇 개 | 기준값 | 놓친 단수명 (90% 끝) | 버린 장수명 (90% 끝) | 정확도 |",
             "|---|---|---|---|---|"]
    for cost, r in table.iterrows():
        mark = " (원래 가정)" if cost == default else ""
        # screening.confusion()의 비율은 이미 % 단위(×100)라 그대로 붙인다
        lines.append(f"| {cost:g}{mark} | {r['기준값 중앙값']:.2f} | {r['놓친 단수명 비율 중앙값']:.1f}% ({r['놓친 단수명 90% 끝']:.1f}%) | "
                     f"{r['버린 장수명 비율 중앙값']:.1f}% ({r['버린 장수명 90% 끝']:.1f}%) | {r['정확도 중앙값']:.1f}% |")
    return "\n".join(lines)
