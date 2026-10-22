"""예측 범위(불확실성) 실험 전체를 돌려 docs/uncertainty.md와 그래프를 만든다.

    python -m battery_lab.uncertainty_report

문서의 숫자는 모두 이 파일의 build()가 계산한다 (손으로 고치지 않는다).
"""

import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mdtext import write_doc
from . import uncertainty as unc
from .features import build_many
from .figures import GRID, INK, INK_2, MUTED, SERIOUS, SURFACE, _save, _style
from .korean import euro, ieyo
from .model import NAMES
from .sources import BATCH_BY_KEY, PROCESSED, ROOT

DOCS = ROOT / "docs"
SHORT_NAMES = {"simple": "단순 모델", "full": "6개 특징 모델"}
SPLIT_NAMES = {"test": "테스트", "secondary_test": "2차 테스트"}
SPLITS = unc.SPLITS
FEATURE_NAMES = {"logvar": "ΔQ 분산", "logmin": "ΔQ 최솟값", "logskew": "ΔQ 왜도", "logkurt": "ΔQ 첨도",
                 "q2": "2번째 사이클 용량", "qmax_q2": "초반 용량 증가분"}
# 방법 색: 색맹에도 구분되는 범주 색을 고정 순서로 (figures.py의 파랑·주황·청록)
METHOD_COLORS = {"boot_line": "#2a78d6", "boot_resid": "#eb6834", "conformal": "#1baf7a"}
# 범위 수준은 크기라서 파랑 한 가지의 진하기로 (넓은 90%가 가장 연하게)
LEVEL_BLUES = {90: "#b7d3f6", 80: "#86b6ef", 50: "#2a78d6"}
EXAMPLE = ("simple", "conformal")   # 예시 그래프: 논문 variance 모델과 같은 형태 + 분포 가정이 없는 방법 (결과를 보기 전에 정함)
EXAMPLE_CELLS = 10
TYP_CLEARLY_BIGGER = 1.1   # '학습 셀의 보통 오차가 평가 셀보다 더 크다'고 쓰는 기준 (10% 넘게 클 때만)
CONFORMAL_SOURCE = ("Angelopoulos · Bates, \"A Gentle Introduction to Conformal Prediction and Distribution-Free "
                    "Uncertainty Quantification\" (arXiv:2107.07511)", "https://arxiv.org/abs/2107.07511")


# ---------------------------------------------------------------- 계산

def build(processed: Path = PROCESSED, docs: Path = DOCS, progress=print, repeats: int = unc.SPLIT_REPEATS,
          boot: dict | None = None, split_boot: int = unc.SPLIT_BOOT) -> dict:
    """특징 계산 → 논문 나누기 범위 → 무작위 나누기 반복 → 그래프 3장 → docs/uncertainty.md.

    중간 결과는 data/processed/uncertainty_*.csv에 저장하고, 문서에 쓴 주요 숫자를 돌려준다.
    진짜 데이터에서 약 2~3분 걸린다 (6개 특징 모델의 부트스트랩이 대부분).
    """
    start = time.perf_counter()
    boot = boot or unc.BOOT
    progress("특징 계산 (100사이클)")
    f = build_many(processed, (100,))[100]
    unc.check(f)
    progress("논문 나누기로 범위 만들기")
    paper = unc.paper_split(f, boot=boot, progress=progress)
    progress(f"무작위 나누기 {repeats}번 × 2가지 방식")
    runs = unc.random_splits(f, repeats=repeats, n_boot=split_boot, progress=progress)
    paper.to_csv(processed / "uncertainty_paper_cells.csv", index=False, encoding="utf-8")
    runs.to_csv(processed / "uncertainty_splits.csv", index=False, encoding="utf-8")

    cov = unc.coverage(paper, ["model", "method", "split", "level"])
    cov_all = unc.coverage(paper, ["model", "method", "level"])          # 테스트 + 2차 테스트 83개
    splits = unc.summarize_splits(runs)
    shift = unc.shift(f)
    fshift = unc.feature_shift(f)
    overlap = unc.policy_overlap(f)
    cov.to_csv(processed / "uncertainty_coverage.csv", index=False, encoding="utf-8")
    splits.to_csv(processed / "uncertainty_splits_summary.csv", index=False, encoding="utf-8")

    n = {sp: int((f["split"] == sp).sum()) for sp in SPLIT_NAMES}
    n_tr = int((f["split"] == "train").sum())
    n_eval = len(f) - n_tr
    n_cal = int(n_tr * unc.CAL_SHARE)
    chance = {(sp, lv): unc.chance_band(n[sp], lv / 100) for sp in SPLIT_NAMES for lv in unc.CURVE_LEVELS}
    chance.update({("all", lv): unc.chance_band(n_eval, lv / 100) for lv in unc.LEVELS})
    # 이항분포 폭은 같은 보정 셀을 함께 쓰는 흔들림을 빼서 좁다 → 분할 conformal은 이론 폭(베타-이항분포)도 계산
    theory = {"conformal": {(sp, lv): unc.conformal_band(n[sp], n_cal, lv / 100) for sp in SPLIT_NAMES
                            for lv in unc.LEVELS},
              "repeat_conformal": unc.conformal_band(n_eval, n_cal, 0.9, 0.1, 0.9),   # 무작위 나누기 표와 같은 10~90%
              "repeat_binom": unc.chance_band(n_eval, 0.9, 0.1, 0.9),
              "narrower": {lv: unc.narrower_share(runs, lv) for lv in (50, 90)}}

    fig_dir = docs / "figures"
    progress("그래프 그리기")
    calibration_figure(cov, chance, n, n_tr, fig_dir / "uncertainty_calibration.png")
    examples = example_cells(paper)
    examples_figure(examples, fig_dir / "uncertainty_examples.png")
    stability_figure(runs, cov, cov_all, fig_dir / "uncertainty_stability.png")

    seconds = time.perf_counter() - start
    md = markdown(f, cov, splits, shift, fshift, overlap, chance, n, n_eval, examples, repeats, boot, split_boot,
                  seconds, theory)
    write_doc(docs / "uncertainty.md", md)
    progress(f"끝 ({seconds:.0f}초) → {docs / 'uncertainty.md'}")
    key = {(r.model, r.method, r.split, int(r.level)): float(r.coverage) for r in cov.itertuples()}
    return {"coverage": cov, "coverage_all": cov_all, "splits": splits, "shift": shift, "feature_shift": fshift,
            "policy_overlap": overlap,
            "chance": chance, "theory": theory, "seconds": seconds,
            "coverage_90": {k[:3]: v for k, v in key.items() if k[3] == 90},
            "width_90": {(r.model, r.method, r.split): float(r.width_mean) for r in cov.itertuples() if r.level == 90}}


def example_cells(paper: pd.DataFrame, per_split: int = EXAMPLE_CELLS) -> pd.DataFrame:
    """예시 그래프용: 나누기마다 실제 수명 순으로 고르게 per_split개. 수준 50·80·90%를 한 줄에."""
    kind, method = EXAMPLE
    p = paper[(paper["model"] == kind) & (paper["method"] == method) & paper["level"].isin(unc.LEVELS)]
    wide = p.pivot_table(index=["cell_id", "split", "life", "pred"], columns="level", values=["lo", "hi"]).reset_index()
    wide.columns = [f"{a}{b}" if b != "" else a for a, b in wide.columns]
    picks = []
    for sp in SPLIT_NAMES:
        g = wide[wide["split"] == sp].sort_values("life").reset_index(drop=True)
        idx = np.unique(np.linspace(0, len(g) - 1, min(per_split, len(g))).round().astype(int))
        picks.append(g.iloc[idx])
    out = pd.concat(picks, ignore_index=True)
    out["inside90"] = (out["life"] >= out["lo90"]) & (out["life"] <= out["hi90"])
    return out


# ---------------------------------------------------------------- 그래프

def calibration_figure(cov: pd.DataFrame, chance: dict, n: dict, n_train: int, path: Path) -> Path:
    """보정 그래프: 약속한 범위(가로) vs 실제로 범위 안에 든 셀 비율(세로). 대각선이 완벽."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), sharex=True, sharey=True)
    levels = np.array(unc.CURVE_LEVELS)
    for r, kind in enumerate(("simple", "full")):
        for c, sp in enumerate(SPLIT_NAMES):
            ax = axes[r, c]
            band = np.array([chance[(sp, lv)] for lv in levels]) * 100
            ax.fill_between(levels, band[:, 0], band[:, 1], color=GRID, alpha=0.8, lw=0,
                            label="정직해도 평가 셀 수가 적어 우연히 나올 수 있는 폭")
            ax.plot([0, 100], [0, 100], color=INK_2, lw=1, ls=(0, (4, 3)), label="완벽 (약속 = 실제)")
            for method in unc.METHODS:
                g = cov[(cov["model"] == kind) & (cov["split"] == sp) & (cov["method"] == method)].sort_values("level")
                ax.plot(g["level"], g["coverage"] * 100, color=METHOD_COLORS[method], lw=1.8, marker="o", ms=3.5,
                        label=unc.METHOD_NAMES[method], zorder=3)
            ax.set(xlim=(5, 100), ylim=(0, 100))
            ax.set_xticks(range(10, 101, 20))
            ax.set_yticks(range(0, 101, 20))
            ax.set_title(f"{SHORT_NAMES[kind]} · {SPLIT_NAMES[sp]} ({n[sp]}개)", fontsize=11, pad=8)
            if r == 1:
                ax.set_xlabel("약속한 범위 (%)")
            if c == 0:
                ax.set_ylabel("실제로 범위 안에 든 셀 (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, labelcolor=INK_2, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("범위가 정직한가: 약속한 비율 vs 실제로 맞은 비율", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.935, f"논문 나누기(학습 {n_train}개) · 선이 대각선 위면 범위가 넉넉하고, 아래면 과신"
             " · 회색 띠 = 이항분포 5~95% (학습·보정 셀이 바뀌는 흔들림은 빠져 실제보다 좁음)",
             color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return _save(fig, path)


def examples_figure(ex: pd.DataFrame, path: Path) -> Path:
    """셀 몇 개의 50·80·90% 범위와 실제 수명."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharex=True)
    top = float(max(ex["hi90"].max(), ex["life"].max())) * 1.05
    for ax, sp in zip(axes, SPLIT_NAMES):
        g = ex[ex["split"] == sp].reset_index(drop=True)
        for y, row in g.iterrows():
            for lv, lw in ((90, 9), (80, 9), (50, 9)):
                ax.plot([row[f"lo{lv}"], row[f"hi{lv}"]], [y, y], color=LEVEL_BLUES[lv], lw=lw,
                        solid_capstyle="butt", zorder=2)
            ax.scatter([row["pred"]], [y], s=26, color=SURFACE, edgecolor=INK_2, linewidth=1, zorder=3)
            miss = not row["inside90"]
            ax.scatter([row["life"]], [y], s=60 if miss else 44, marker="D", color=SERIOUS if miss else INK,
                       edgecolor=SURFACE, linewidth=0.8, zorder=4)
        ax.set_yticks(range(len(g)), labels=g["cell_id"])
        ax.set_ylim(len(g) - 0.4, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, top)
        ax.set_xlabel("사이클 수명")
        ax.set_title(f"{SPLIT_NAMES[sp]} ({'b1·b2' if sp == 'test' else 'b3'}) · 실제 수명 순으로 {len(g)}개",
                     fontsize=11, pad=8)
    handles = [plt.Line2D([], [], color=LEVEL_BLUES[lv], lw=8, label=f"{lv}% 범위") for lv in (50, 80, 90)]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=6, markerfacecolor=SURFACE, markeredgecolor=INK_2,
                           label="예측"),
                plt.Line2D([], [], ls="", marker="D", ms=6, color=INK, label="실제 수명"),
                plt.Line2D([], [], ls="", marker="D", ms=7, color=SERIOUS, label="실제가 90% 범위 밖")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(handles), labelcolor=INK_2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    kind, method = EXAMPLE
    fig.suptitle("셀마다 붙인 범위와 실제 수명", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, f"{NAMES[kind]} + {unc.METHOD_NAMES[method]} · log 수명에서 같은 폭이라 "
             "사이클로는 오래 사는 셀일수록 범위가 넓어요", color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    return _save(fig, path)


STABILITY_ROWS = (("simple", "boot_resid"), ("simple", "conformal"), ("full", "conformal"))


def _box(ax, y: float, s: dict, color: str, filled: bool) -> None:
    """10~90% 선 + 25~75% 상자 + 중앙값 막대."""
    ax.plot([s["p10"], s["p90"]], [y, y], color=color, lw=1.4, solid_capstyle="round", zorder=2)
    ax.add_patch(plt.Rectangle((s["p25"], y - 0.13), s["p75"] - s["p25"], 0.26, lw=1.3, edgecolor=color,
                               facecolor=color if filled else SURFACE, alpha=0.9 if filled else 1, zorder=3))
    ax.plot([s["p50"]] * 2, [y - 0.13, y + 0.13], color=INK if filled else color, lw=2, zorder=4)


def _stats(runs_cov: np.ndarray) -> dict:
    p10, p25, p50, p75, p90 = np.percentile(runs_cov * 100, [10, 25, 50, 75, 90])
    return {"p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90}


def stability_figure(runs: pd.DataFrame, cov: pd.DataFrame, cov_all: pd.DataFrame, path: Path) -> Path:
    """무작위 나누기에서 90% 범위의 적중률이 얼마나 흔들리는지 (왼쪽: 평가 셀 전체, 오른쪽: b3 셀만)."""
    _style()
    r90 = runs[runs["level"] == 90]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    lows = []
    n_all = int(r90.loc[r90["group"] == "all", "cells"].median())
    for ax, group, title in ((axes[0], "all", f"평가 셀 전체 ({n_all}개)"), (axes[1], "b3", "b3 셀만")):
        for y, (kind, method) in enumerate(STABILITY_ROWS):
            color = METHOD_COLORS[method]
            for offset, scheme, filled in ((-0.19, "mixed", False), (0.19, "b12", True)):
                v = r90[(r90["scheme"] == scheme) & (r90["model"] == kind) & (r90["method"] == method)
                        & (r90["group"] == group)]["coverage"].to_numpy()
                s = _stats(v)
                lows.append(s["p10"])
                _box(ax, y + offset, s, color, filled)
            if group == "all":
                paper = cov_all[(cov_all["model"] == kind) & (cov_all["method"] == method) & (cov_all["level"] == 90)]
            else:
                paper = cov[(cov["model"] == kind) & (cov["method"] == method) & (cov["level"] == 90)
                            & (cov["split"] == "secondary_test")]
            x = float(paper["coverage"].iloc[0]) * 100
            lows.append(x)
            ax.scatter([x], [y + 0.19], marker="*", s=120, color=INK, edgecolor=SURFACE, linewidth=0.6, zorder=5)
        ax.axvline(90, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("90% 범위에 실제로 든 셀 (%)")
        ax.grid(axis="y", visible=False)
    left = 5 * np.floor(min(lows) / 5) - 5
    for ax in axes:
        ax.set_xlim(left, 101)
    axes[0].set_yticks(range(len(STABILITY_ROWS)),
                       labels=[f"{SHORT_NAMES[k]}\n{unc.METHOD_NAMES[m]}" for k, m in STABILITY_ROWS])
    axes[0].set_ylim(len(STABILITY_ROWS) - 0.5, -0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=MUTED, label=unc.SCHEMES["mixed"]),
               plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=MUTED, label=unc.SCHEMES["b12"]),
               plt.Line2D([], [], ls="", marker="*", ms=11, color=INK, label="논문 나누기 한 번의 결과")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3, labelcolor=INK_2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    reps = int(r90["repeat"].nunique())
    fig.suptitle("학습 셀을 바꿔 뽑으면 적중률은 얼마나 흔들릴까", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.9, f"방식마다 {reps}번 · 선 = 10~90% · 상자 = 25~75% · 굵은 막대 = 중앙값 · 점선 = 약속한 90%",
             color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    return _save(fig, path)


# ---------------------------------------------------------------- 문서

def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _cyc(x: float) -> str:
    return "무한대" if not np.isfinite(x) else f"{x:,.0f}"


def _signed_pct(x: float) -> str:
    """+1.2% / −3.4% (반올림하면 0이 되는 값은 −0.0%가 아니라 0.0%로)."""
    return "0.0%" if round(x * 100, 1) == 0 else f"{x * 100:+.1f}%"


def _band(b: tuple[float, float]) -> str:
    return f"{b[0] * 100:.0f}~{b[1] * 100:.0f}%"


def _get(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    rows = df[m]
    if len(rows) != 1:
        raise KeyError(f"{kw}: {len(rows)}줄")
    return rows.iloc[0]


def _months_between_batches() -> float:
    """b1·b2 중 마지막 시험 시작일부터 b3 시험 시작일까지 몇 달인가 (sources.py의 날짜로 계산)."""
    first = max(date.fromisoformat(BATCH_BY_KEY[k].date) for k in ("b1", "b2"))
    return (date.fromisoformat(BATCH_BY_KEY["b3"].date) - first).days / 30.44


def markdown(f, cov, splits, shift, fshift, overlap, chance, n, n_eval, ex, repeats, boot, split_boot,
             seconds, theory) -> str:
    def c(kind, method, sp, lv):
        return _get(cov, model=kind, method=method, split=sp, level=lv)

    def s(scheme, kind, method, lv, group):
        return _get(splits, scheme=scheme, model=kind, method=method, level=lv, group=group)

    def inside(sp, lv, x):
        lo, hi = chance[(sp, lv)]
        return lo <= x <= hi

    sh = {r.split: r for r in shift.itertuples()}
    fs = {(r.feature, r.split): r for r in fshift.itertuples()}
    n_tr = int(sh["train"].cells)
    n_cal = int(n_tr * unc.CAL_SHARE)
    k90 = int(np.ceil((n_cal + 1) * 0.9 - 1e-9))          # 90% 범위의 폭을 정하는 보정 오차의 순번
    nth_big = n_cal - k90 + 1                              # 큰 쪽에서 몇 번째인가
    lv_lo, lv_hi = min(unc.CURVE_LEVELS), max(unc.CURVE_LEVELS)
    kind, method = EXAMPLE
    t = ex[ex["split"] == "test"].sort_values("pred").reset_index(drop=True)
    mid = t.iloc[len(t) // 2]                              # 예시: 테스트 셀 중 예측이 가운데인 셀
    policy = f.set_index("cell_id").loc[mid["cell_id"], "policy"]
    # 따옴표는 굵게 밖에 둔다: "**…**"이고 (굵게 안에 따옴표를 넣으면 '"**이고'에서 굵게가 닫히지 않는다)
    say = f"예측 {mid['pred']:,.0f}사이클, 90% 확률로 {mid['lo90']:,.0f}~{mid['hi90']:,.0f}사이클"
    lines = [
        "# 예측에 오차 범위 붙이기 (불확실성)",
        "",
        f"`python -m battery_lab.uncertainty_report`가 자동으로 만든 문서예요 (만든 날 {date.today().isoformat()}, "
        f"계산 {seconds / 60:.1f}분).",
        "",
        "## 질문",
        "",
        f"수명 예측 모델은 \"이 셀은 {mid['pred']:,.0f}사이클\"처럼 숫자 하나를 내놓아요. 하지만 실무에서 정말 필요한 건",
        f"**\"90% 확률로 {mid['lo90']:,.0f}~{mid['hi90']:,.0f}사이클\"** 같은 범위예요"
        f" (이 문서의 예시 셀 {mid['cell_id']}에서 실제로 계산한 결과). 그래서 이렇게 물었어요.",
        "",
        "1. 초반 100사이클로 한 수명 예측에 **범위**를 붙일 수 있을까?",
        "2. \"90% 범위\"(실제 수명이 그 안에 들어올 확률이 90%라고 내놓은 범위)라고 했으면"
        " **정말로 셀의 90% 안팎이 그 안에 들어올까?** (범위가 정직한가 = 보정, calibration)",
        "3. 다른 시기에 시험한 3배치(b3, 2차 테스트)에서도 그럴까?",
        "",
        "## 왜 중요한가",
        "",
        "- **셀 개발:** 새 소재·설계 후보를 100사이클 데이터로 비교할 때, 두 후보의 예측 차이가 범위보다 작으면"
        " \"아직 구분할 수 없음\"이에요. 범위가 있어야 시험을 더 돌릴지, 지금 고를지 근거를 갖고 정할 수 있어요.",
        "- **공정 개발·품질:** 보증 수명을 정할 때 중요한 건 예측값이 아니라 **범위의 아래 끝**이에요."
        " 그래서 실제 수명이 범위보다 **짧게** 빗나간 비율(위험한 쪽)을 따로 셌어요.",
        "- **정직하지 않은 범위는 숫자 하나보다 위험해요.** \"90%\"라고 믿고 결정했는데 실제로는 절반도 안 맞는 범위라면 과신이에요."
        " 그래서 범위를 만드는 것보다 **범위가 약속을 지키는지 확인하는 것**에 더 힘을 썼어요.",
        "",
        "## 방법",
        "",
        "- **예측 모델:** [데이터 크기 실험](data_size.md)과 같은 두 모델이에요. 둘 다 log10(수명)을 예측해요"
        " ([model.py](../battery_lab/model.py)).",
        f"  - **{NAMES['simple']}:** 논문의 variance 모델과 같은 형태",
        f"  - **{NAMES['full']}:** 논문의 discharge 모델을 참고해 직접 구현",
        "- **범위를 만드는 방법 세 가지** ([uncertainty.py](../battery_lab/uncertainty.py))",
        f"  - **{unc.METHOD_NAMES['boot_line']}:** 학습 셀 {n_tr}개를 중복을 허용해 다시 뽑아(부트스트랩, bootstrap) 모델을"
        f" 여러 번 만들고(단순 {boot['simple']:,}번 · 6개 특징 {boot['full']:,}번), 예측값이 흔들리는 폭을 범위로 써요."
        " **모델(직선)이 얼마나 불확실한지**만 담아요.",
        f"  - **{unc.METHOD_NAMES['boot_resid']}:** 위의 예측값마다, 그 모델이 학습에 쓰지 않은 셀(OOB, out-of-bag)에서 낸"
        " 오차를 더해요. **특징이 같아도 셀마다 수명이 다른 '셀 사이 편차'까지** 담아요.",
        f"  - **{unc.METHOD_NAMES['conformal']}:** 학습 셀을 무작위로 나눠 {n_tr - n_cal}개로 모델을 만들고, 나머지 {n_cal}개"
        " (보정 셀)의 오차 크기 순위로 폭을 정해요. 오차가 어떤 모양이든 **평균적으로** 약속한 비율을 지키도록 설계된 방법이에요"
        " (분포 가정이 없음, distribution-free). 단, 새 셀이 보정 셀과 같은 주머니에서 나왔다고 볼 수 있어야 해요"
        f" (교환 가능성, exchangeability). 입문 자료: [{CONFORMAL_SOURCE[0]}]({CONFORMAL_SOURCE[1]}).",
        "  - 세 방법 모두 log10 수명에서 범위를 만들고 사이클로 바꿔요. 그래서 사이클로는 위쪽이 아래쪽보다 조금 길어요.",
        f"- **정직함 확인:** 학습 {n_tr}개로 범위를 만들고, 테스트 {n['test']}개({sh['test'].batches})와"
        f" 2차 테스트 {n['secondary_test']}개({sh['secondary_test'].batches})에서 셌어요.",
        "  - **적중률(coverage):** 실제 수명이 범위 안에 든 셀의 비율. 90% 범위라면 90% 근처여야 정직해요.",
        "  - **범위 폭:** 범위의 위 끝 − 아래 끝(사이클). 적중률이 같다면 좁을수록 쓸모 있어요.",
        f"  - **보정 그래프:** {lv_lo}%부터 {lv_hi}%까지 범위를 바꿔 가며 약속한 비율과 실제 비율을 비교해요.",
        "- **우연의 폭:** 셀이 40개 남짓이면 정직한 범위라도 적중률이 우연히 흔들려요. 진짜 적중 확률이 약속과 같을 때"
        " 셀 n개에서 나올 수 있는 적중률의 5~95% 범위를 이항분포(동전 던지기 계산)로 구해 같이 적었어요."
        " 단, 이 폭은 평가 셀이 서로 따로 맞고 틀린다고(독립) 보고 계산해서 **실제 흔들림보다 좁아요.** 실제로는 모든 평가"
        " 셀이 같은 학습·보정 셀로 만든 범위를 함께 쓰기 때문이에요. 분할 conformal은 그 흔들림까지 넣은 폭을 이론(베타-이항분포)으로"
        " 계산할 수 있어서 '읽는 법'에 같이 적었어요 (위 입문 자료의 3.2절·부록 C).",
        f"- **나누기를 바꿔 보기:** 학습 {n_tr}개를 무작위로 다시 뽑는 실험을 두 방식으로 {repeats}번씩 했어요"
        f" (부트스트랩 방법은 나누기마다 모델을 {split_boot}개씩 만들어요). 평가는 학습에 안 쓴 나머지 {n_eval}개 셀 전체예요.",
        f"  - **{unc.SCHEMES['mixed']}**",
        f"  - **{unc.SCHEMES['b12']}**",
        f"  - 6개 특징 모델의 부트스트랩은 모델을 {2 * repeats * split_boot:,}번 다시 맞춰야 해서 이 실험에서는 뺐어요"
        " (분할 conformal만).",
        f"- 무작위 뽑기는 모두 고정한 시드(seed {unc.SEED})로 해서, 다시 돌려도 같은 숫자가 나와요.",
        "",
        "## 결과",
        "",
        f"### 1. 논문 나누기: 학습 {n_tr}개 → 테스트 {n['test']}개 · 2차 테스트 {n['secondary_test']}개",
        "",
        "칸 = **실제로 범위 안에 든 셀 비율** · 범위의 평균 폭(사이클)",
        "",
    ]
    for sp in SPLIT_NAMES:
        lines += [f"**{SPLIT_NAMES[sp]} {n[sp]}개 ({sh[sp].batches})**", "",
                  "| 모델 | 방법 | 50% 범위 | 80% 범위 | 90% 범위 |", "|---|---|---|---|---|"]
        for k in ("simple", "full"):
            for m in unc.METHODS:
                cells = []
                for lv in unc.LEVELS:
                    r = c(k, m, sp, lv)
                    txt = f"{_pct(r.coverage)} · {_cyc(r.width_mean)}"
                    cells.append(f"**{txt}**" if lv == 90 else txt)
                lines.append(f"| {SHORT_NAMES[k]} | {unc.METHOD_NAMES[m]} | " + " | ".join(cells) + " |")
        lines += ["", f"*정직한 범위라도 셀 {n[sp]}개면 적중률이 우연히 50% 범위 {_band(chance[(sp, 50)])}, "
                      f"80% 범위 {_band(chance[(sp, 80)])}, 90% 범위 {_band(chance[(sp, 90)])} 사이로 나올 수 있어요"
                      " (이항분포 5~95%).*", ""]
    lines += ["![보정 그래프](figures/uncertainty_calibration.png)", ""]

    misses = ex.groupby("split")["inside90"].agg(["size", "sum"])
    lines += [
        "### 2. 예시: 셀마다 붙인 범위",
        "",
        f"테스트 셀 {mid['cell_id']}({policy})의 경우, {SHORT_NAMES[kind]} + {unc.METHOD_NAMES[method]}의 답은",
        f"\"**{say}**\"이고 실제 수명은 **{mid['life']:,.0f}사이클**이에요 ({'범위 안' if mid['inside90'] else '범위 밖'}).",
        "",
        "![셀마다 붙인 범위](figures/uncertainty_examples.png)",
        "",
        f"*그래프의 셀은 결과를 보고 고르지 않고, 나누기마다 실제 수명 순으로 고르게 {EXAMPLE_CELLS}개씩 뽑았어요."
        f" 이 중 90% 범위 밖은 테스트 {int(misses.loc['test', 'size'] - misses.loc['test', 'sum'])}개, "
        f"2차 테스트 {int(misses.loc['secondary_test', 'size'] - misses.loc['secondary_test', 'sum'])}개예요.*",
        "",
        "### 3. 학습 셀을 바꿔 뽑으면 적중률은 얼마나 흔들릴까",
        "",
        f"90% 범위의 적중률 · 방식마다 {repeats}번 · 칸 = 중앙값 (10~90% 범위)"
        f" · '80% 미만' 칸은 {repeats}번 중 비율, '논문 나누기' 칸은 한 번의 값",
        "",
        f"*10~90% 범위 = {repeats}번의 적중률을 작은 것부터 줄 세워 아래 10%와 위 10%를 뺀 가운데 80% 구간이에요.*",
        "",
        "| 모델 | 방법 | 섞어 나누기: 평가 셀 전체 | 섞어 나누기: 적중률 80% 미만이었던 나누기 비율 | 섞어 나누기: b3 셀만 |"
        " b1·b2에서만 학습: b3 셀만 | 논문 나누기: b3 (한 번) |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in ("simple", "full"):
        for m in unc.REPEAT_METHODS[k]:
            a, b3m, b3b = s("mixed", k, m, 90, "all"), s("mixed", k, m, 90, "b3"), s("b12", k, m, 90, "b3")
            p = c(k, m, "secondary_test", 90)
            lines.append(f"| {SHORT_NAMES[k]} | {unc.METHOD_NAMES[m]} | {_pct(a.mid)} ({_pct(a.p10)}~{_pct(a.p90)}) | "
                         f"{_pct(a.short_by_10)} | {_pct(b3m.mid)} ({_pct(b3m.p10)}~{_pct(b3m.p90)}) | "
                         f"{_pct(b3b.mid)} ({_pct(b3b.p10)}~{_pct(b3b.p90)}) | {_pct(p.coverage)} |")
    b3_cells = int(s("mixed", "simple", "conformal", 90, "b3").cells)
    lines += [
        "",
        f"*섞어 나누기에서는 평가 셀 {n_eval}개 중 b3가 보통 {b3_cells}개쯤 들어가요."
        f" b1·b2에서만 학습하면 b3 {n['secondary_test']}개가 모두 평가에 들어가요.*",
        "",
        "![나누기에 따른 적중률](figures/uncertainty_stability.png)",
        "",
    ]

    ov, nb3 = overlap
    lines += [
        "### 4. 2차 테스트(b3)는 학습 셀과 무엇이 다를까",
        "",
        "**수명과 예측 오차** (학습 셀 전체로 만든 모델 기준)",
        "",
        "| 나누기 | 셀 | 배치 | 수명 중앙값 (10~90%) | 단순 모델 치우침 | 단순 모델 log 오차 흩어짐 | 단순 모델 보통 셀의 log 오차 | 6개 특징 모델 치우침 | 6개 특징 모델 log 오차 흩어짐 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sp in SPLITS:
        r = sh[sp]
        name = {"train": "학습 (모델이 본 셀)", **SPLIT_NAMES}[sp]
        lines.append(f"| {name} | {r.cells} | {r.batches} | {r.life_median:,.0f} ({r.life_p10:,.0f}~{r.life_p90:,.0f}) | "
                     f"{_signed_pct(r.bias_simple)} | {r.spread_simple:.3f} | {r.abs_median_simple:.3f} | "
                     f"{_signed_pct(r.bias_full)} | {r.spread_full:.3f} |")
    lines += [
        "",
        "*치우침 = 실제 수명이 예측보다 평균 몇 % 긴가(+) 짧은가(−) (log 오차 평균을 %로 바꾼 값). 흩어짐 = log10 오차의 표준편차."
        " 보통 셀의 log 오차 = |log10 오차|의 중앙값. 학습 셀은 모델이 직접 보고 맞춘 셀이라 오차가 작게 나오는 쪽이에요.*",
        "",
        f"**특징 분포** (학습 셀 {n_tr}개 기준)",
        "",
        "| 특징 | 테스트: 중앙값 차이 | 2차 테스트: 중앙값 차이 | 2차 테스트 중 학습 셀 범위 밖 |",
        "|---|---|---|---|",
    ]
    for name in unc.FEATURES:
        t_, s_ = fs[(name, "test")], fs[(name, "secondary_test")]
        lines.append(f"| {FEATURE_NAMES[name]} | {t_.z:+.2f} | **{s_.z:+.2f}** | {_pct(s_.outside)} |")
    lines += [
        "",
        "*중앙값 차이 = (그 나누기의 중앙값 − 학습 셀 중앙값) ÷ 학습 셀 표준편차. ±1이면 학습 셀이 흩어진 폭만큼 옮겨 간 것이에요."
        " 범위 밖 = 학습 셀의 최솟값~최댓값 밖인 셀 비율. ΔQ 특징 네 개는 모델에 넣는 그대로 log10 값이에요"
        " ([features.py](../battery_lab/features.py)).*",
        "",
        f"b3의 충전 조건 {nb3}가지는 이름 끝의 `-newstructure`를 떼고 비교해도 b1·b2와 겹치는 것이 {ov}가지예요.",
        "",
    ]

    # ---------------------------------------------------------------- 읽는 법
    bl = {sp: c("simple", "boot_line", sp, 90) for sp in SPLIT_NAMES}
    br = {sp: c("simple", "boot_resid", sp, 90) for sp in SPLIT_NAMES}
    cf = {sp: c("simple", "conformal", sp, 90) for sp in SPLIT_NAMES}
    honest = [(k, m, sp) for k in ("simple", "full") for m in ("boot_resid", "conformal") for sp in SPLIT_NAMES]
    in_band = [x for x in honest if inside(x[2], 90, c(*x, 90).coverage)]
    over50 = [x for x in honest if c(*x, 50).coverage > chance[(x[2], 50)][1]]
    tb = theory["conformal"]
    conf50 = [x for x in honest if x[1] == "conformal"]
    conf_over50 = [x for x in conf50 if c(*x, 50).coverage > tb[(x[2], 50)][1]]
    mix50 = {m: s("mixed", "simple", m, 50, "all") for m in ("boot_resid", "conformal")}
    w50 = (c("full", "conformal", "test", 50).width_mean, c("simple", "conformal", "test", 50).width_mean)
    w90 = (c("full", "conformal", "test", 90).width_mean, c("simple", "conformal", "test", 90).width_mean)
    sec_ratio = sh["secondary_test"].life_median / sh["test"].life_median
    lines += [
        "## 읽는 법 (숫자는 코드가 계산)",
        "",
        f"1. **모델만 흔들면 범위가 너무 좁아요.** {unc.METHOD_NAMES['boot_line']}의 90% 범위는 평균 폭이"
        f" {_cyc(bl['test'].width_mean)}사이클(테스트)로 좁지만, 실제로 맞은 셀은 테스트 {_pct(bl['test'].coverage)},"
        f" 2차 테스트 {_pct(bl['secondary_test'].coverage)}뿐이에요. 학습 셀 {n_tr}개로 맞춘 직선 자체는 꽤 안정적이지만,"
        " **같은 특징값을 가진 셀끼리도 수명이 다르기 때문**이에요. \"직선이 얼마나 확실한가\"와 \"셀 하나의 수명이 얼마나"
        " 확실한가\"는 다른 질문이에요.",
        f"2. **셀 사이 편차를 넣으면 약속에 가까워져요.** 단순 모델의 90% 범위는 {unc.METHOD_NAMES['boot_resid']}가"
        f" 테스트 {_pct(br['test'].coverage)} · 2차 테스트 {_pct(br['secondary_test'].coverage)},"
        f" {unc.METHOD_NAMES['conformal']}이 {_pct(cf['test'].coverage)} · {_pct(cf['secondary_test'].coverage)}를 맞혔어요."
        f" 두 방법 × 두 모델 × 두 나누기 {len(honest)}경우 중 **{len(in_band)}경우가 우연의 폭 안**이에요."
        f" 대신 폭은 넓어요: 단순 모델 + {unc.METHOD_NAMES['conformal']}의 90% 범위는 예측값의 약"
        f" ±{cf['test'].rel_half * 100:.0f}%(테스트 평균 폭 {_cyc(cf['test'].width_mean)}사이클)예요.",
    ]
    if over50:
        typical = all(abs(v.mid - 0.5) <= 0.07 for v in mix50.values())
        # 보통 셀의 |log 오차|는 모델마다 다르다 → 넉넉했던 경우마다 그 모델·그 평가 셀로 따로 본다
        typ = {k: {sp: getattr(sh[sp], f"abs_median_{k}") for sp in SPLITS} for k in ("simple", "full")}

        def typ_txt(k):
            t = typ[k]
            return (f"{SHORT_NAMES[k]}의 보통 셀 |log 오차|: 학습 셀 {t['train']:.3f} · 테스트 {t['test']:.3f}"
                    f" · 2차 테스트 {t['secondary_test']:.3f}")

        def by_model(cases):
            return " · ".join(f"{SHORT_NAMES[k]} {sum(x[0] == k for x in cases)}경우"
                              for k in ("simple", "full") if any(x[0] == k for x in cases))

        # 학습 셀의 보통 오차가 그 평가 셀보다 뚜렷이(10% 넘게) 클 때만 '더 커서'라고 쓴다
        explained = [x for x in over50 if typ[x[0]]["train"] > TYP_CLEARLY_BIGGER * typ[x[0]][x[2]]]
        rest = [x for x in over50 if x not in explained]
        if not typical:
            why = (" 무작위 나누기에서도 50%를 넘어서, **오차가 대부분 작고 몇 개만 큰 모양**(꼬리가 두꺼움)이라 가운데 범위가"
                   f" 넉넉해지는 것으로 보여요(가설). 참고로 {typ_txt('simple')}.")
        elif explained:
            kinds = [k for k in ("simple", "full") if any(x[0] == k for x in explained)]
            why = (" 그래서 논문 나누기에서 50%가 넉넉했던 것은 **이 나누기만의 우연**일 가능성이 있어요(가설)."
                   f" {' / '.join(typ_txt(k) for k in kinds)}. 넉넉했던 {len(over50)}경우 중 {by_model(explained)}는"
                   " 모델이 직접 본 학습 셀의 보통 오차가 그 평가 셀보다 오히려 더 커서, 학습 셀로 정한 50% 폭이 평가 셀에는"
                   " 넉넉했던 것으로 보여요.")
            if rest:
                detail = ", ".join(f"{SHORT_NAMES[k]} + {unc.METHOD_NAMES[m]} · {SPLIT_NAMES[sp]}: 학습 셀"
                                   f" {typ[k]['train']:.3f} vs {SPLIT_NAMES[sp]} {typ[k][sp]:.3f}" for k, m, sp in rest)
                why += (f" 나머지 {by_model(rest)}는 학습 셀의 보통 오차가 그 평가 셀보다 뚜렷이"
                        f"({(TYP_CLEARLY_BIGGER - 1) * 100:.0f}% 넘게) 크지 않아서"
                        f" 이 설명이 맞지 않아요 ({detail}).")
        else:
            why = (" 그래서 이 나누기만의 일로 보이지만, 보통 오차 크기로는 설명되지 않았어요"
                   f" ({typ_txt('simple')} / {typ_txt('full')}).")
        lines.append(
            f"3. **좁은 범위(50%)는 오히려 넉넉했어요.** 50% 범위의 적중률이 {len(honest)}경우 중 {len(over50)}경우에서"
            " 이항분포 우연의 폭보다 높았어요(보정 그래프에서 선이 대각선 위). 다만 이항분포 폭은 실제보다 좁아요."
            f" {unc.METHOD_NAMES['conformal']}만 이론 폭(베타-이항분포: 테스트 {_band(tb[('test', 50)])},"
            f" 2차 테스트 {_band(tb[('secondary_test', 50)])})으로 다시 보면 {len(conf50)}경우 중 {len(conf_over50)}경우만"
            f" 그 폭 위예요. 섞어 나누기 {repeats}번에서는 단순 모델 50% 범위의 적중률 중앙값이"
            f" {unc.METHOD_NAMES['boot_resid']} {_pct(mix50['boot_resid'].mid)},"
            f" {unc.METHOD_NAMES['conformal']} {_pct(mix50['conformal'].mid)}예요." + why)
    else:
        lines.append("3. **좁은 범위(50%)도 우연의 폭 안이었어요.**")
    # 논문 나누기 한 번만 보면 우연에 휘둘리므로, 무작위 나누기(같은 학습·보정 셀로 두 모델을 짝지어 비교)도 같이 본다
    wm = {(k, lv): s("mixed", k, "conformal", lv, "all").width for k in ("simple", "full") for lv in (50, 90)}
    gain = {lv: 1 - wm[("full", lv)] / wm[("simple", lv)] for lv in (50, 90)}
    narrow = theory["narrower"]
    paper_txt = (f" 논문 나누기 테스트에서는 50% 범위 6개 특징 {_cyc(w50[0])} vs 단순 {_cyc(w50[1])}사이클,"
                 f" 90% 범위 {_cyc(w90[0])} vs {_cyc(w90[1])}사이클이었어요.")
    repeat_txt = (f" 섞어 나누기 {repeats}번의 중앙값은 50% 범위 {_cyc(wm[('full', 50)])} vs {_cyc(wm[('simple', 50)])}사이클,"
                  f" 90% 범위 {_cyc(wm[('full', 90)])} vs {_cyc(wm[('simple', 90)])}사이클이고, 6개 특징 모델이 더 좁았던"
                  f" 나누기는 50% 범위 {_pct(narrow[50])}, 90% 범위 {_pct(narrow[90])}예요.")
    if gain[50] > 0 and gain[50] > gain[90]:
        one_off = (" 그래서 논문 나누기에서 90% 범위가 오히려 넓었던 것은 이 나누기 한 번의 결과로 보이고,"
                   if w90[0] >= w90[1] and narrow[90] > 0.5 else "")
        lines.append(
            f"4. **6개 특징 모델의 범위는 좁은 범위(50%)에서 더 많이 좁아져요.** {unc.METHOD_NAMES['conformal']}의 평균 폭을"
            " 비교했어요." + paper_txt + repeat_txt + one_off
            + " 넓은 범위로 갈수록 이득이 줄어드는 것은 공통이에요. 대부분의 셀은 더 잘 맞히지만 크게 빗나가는 몇 개는 비슷하게"
            " 빗나가서, 넓은 범위의 폭은 그 몇 개가 정하는 것으로 보여요(가설).")
    else:
        lines.append(
            f"4. **두 모델의 범위 폭 비교:** {unc.METHOD_NAMES['conformal']}의 평균 폭이에요." + paper_txt + repeat_txt)
    a_cf, a_br = s("mixed", "simple", "conformal", 90, "all"), s("mixed", "simple", "boot_resid", 90, "all")
    steadier = a_br.short_by_10 < a_cf.short_by_10 and (a_br.p90 - a_br.p10) < (a_cf.p90 - a_cf.p10)
    lines.append(
        f"5. **분할 conformal은 '평균적으로' 정직하지만 한 번 한 번은 흔들려요.** 섞어 나누기 {repeats}번에서 단순 모델의 90% 범위"
        f" 적중률은 중앙값 {_pct(a_cf.mid)}이지만 10~90%가 {_pct(a_cf.p10)}~{_pct(a_cf.p90)}였어요."
        + (f" {_pct(a_cf.short_by_10)}의 경우 80%도 안 됐어요." if a_cf.short_by_10 > 0 else " 80% 아래로 떨어진 경우는 없었어요.")
        + f" 보정 셀이 {n_cal}개뿐이라 90% 범위의 폭이 사실상"
        f" **보정 셀 {n_cal}개 중 {nth_big}번째로 큰 오차** 하나로 정해지기 때문이에요."
        f" 이론(베타-이항분포)으로 계산해도 보정 셀 {n_cal}개 · 평가 셀 {n_eval}개면 적중률의 10~90%가"
        f" {_band(theory['repeat_conformal'])}로, 실제로 본 흔들림과 비슷해요. 평가 셀만의 우연(이항분포)이었다면"
        f" {_band(theory['repeat_binom'])}였을 폭이에요."
        f" {unc.METHOD_NAMES['boot_resid']}는 같은 조건에서 {_pct(a_br.p10)}~{_pct(a_br.p90)}"
        f"({_pct(a_br.short_by_10)}가 80% 미만)였어요."
        + (" 학습 셀을 반으로 나누지 않고 모두 쓰기 때문으로 보여요(가설)." if steadier else ""))
    b3_mixed, b3_b12 = s("mixed", "simple", "conformal", 90, "b3"), s("b12", "simple", "conformal", 90, "b3")
    sec_in = [inside("secondary_test", 90, c(k, m, "secondary_test", 90).coverage)
              for k in ("simple", "full") for m in ("boot_resid", "conformal")]
    gap = (b3_mixed.mid - b3_b12.mid) * 100
    f3m, f3b = s("mixed", "full", "conformal", 90, "b3"), s("b12", "full", "conformal", 90, "b3")
    far = [name for name in unc.FEATURES if abs(fs[(name, "secondary_test")].z) >= 1]
    out_feats = [name for name in unc.FEATURES if fs[(name, "secondary_test")].outside > 0]
    lv = fs[("logvar", "secondary_test")]
    ts, ss = sh["test"], sh["secondary_test"]
    lines += [
        "6. **2차 테스트(b3)는 '다른 묶음'이지만, log 기준 범위는 "
        + ("대체로 지켜졌어요 (관찰).**" if all(sec_in) else "일부 지켜지지 않았어요 (관찰).**"),
        f"   - 다른 점: b3 셀은 수명 중앙값이 {sh['secondary_test'].life_median:,.0f}사이클로 테스트"
        f"({sh['test'].life_median:,.0f})의 {sec_ratio:.1f}배예요. 특징 {len(unc.FEATURES)}개 중 {len(far)}개"
        + (f"({', '.join(FEATURE_NAMES[x] for x in far)})" if far else "")
        + "의 중앙값이 학습 셀에서 표준편차 1배 넘게 떨어져 있어요. 테스트 셀은 모든 특징이 학습 셀과 거의 같은 자리예요."
        f" 충전 조건은 b1·b2와 겹치는 것이 {nb3}가지 중 {ov}가지이고, 시험 시작일은 b2보다 약 {_months_between_batches():.0f}개월"
        " 뒤예요([sources.py](../battery_lab/sources.py)).",
        f"   - 비슷한 점: 단순 모델이 쓰는 특징 하나({FEATURE_NAMES['logvar']})는 b3 셀의 {_pct(1 - lv.outside)}가 학습 셀 범위 안에 있어요"
        " (학습 셀이 본 범위 안에서 옮겨 간 것). 단순 모델의 log 오차 흩어짐도"
        f" {euro(format(sh['secondary_test'].spread_simple, '.3f'))} 테스트({sh['test'].spread_simple:.3f})와 비슷해요."
        f" 90% 범위의 적중률은 두 방법 × 두 모델 {len(sec_in)}경우 중 {sum(sec_in)}경우가 우연의 폭 안이었어요.",
        f"   - 무작위 나누기에서 b3 셀만 본 적중률은 단순 모델 + {unc.METHOD_NAMES['conformal']}이 b3가 학습에 일부 있을 때"
        f" {_pct(b3_mixed.mid)}, 전혀 없을 때 {_pct(b3_b12.mid)}로 "
        + ("차이가 작아요." if abs(gap) <= 5 else f"{abs(gap):.0f}%p 차이가 나요.")
        + f" 6개 특징 모델은 중앙값 {_pct(f3m.mid)} → {_pct(f3b.mid)}, 나쁜 쪽 끝(10%)은 {_pct(f3m.p10)} → {_pct(f3b.p10)}예요."
        + (" **6개 특징 모델은 b3 셀 일부가 학습 셀 범위 밖에 있는 특징"
           + (f"({', '.join(FEATURE_NAMES[x] for x in out_feats)})" if out_feats else "")
           + "까지 쓰기 때문에 새 묶음에서 더 흔들리는 것**일 수 있어요(가설. 모델 두 개만 비교했어요)."
           if f3b.p10 < f3m.p10 - 0.05 and f3b.p10 < s("b12", "simple", "conformal", 90, "b3").p10 else ""),
        "   - **가설:** 2차 테스트의 RMSE(평균 제곱근 오차. 예측이 대략 몇 사이클 빗나갔나)가 테스트보다 큰 것(단순 모델"
        f" {_cyc(ss.rmse_simple)} vs {_cyc(ts.rmse_simple)}사이클, [데이터 크기 실험](data_size.md))은 b3 셀이 더 나쁘게"
        " 맞아서라기보다, **수명이 길어서 같은 % 오차가 더 많은 사이클이 되기 때문**일 수 있어요. 평균 % 오차(MAPE)는"
        f" 2차 테스트 {ss.mape_simple:.1f}% · 테스트 {ts.mape_simple:.1f}%로"
        + (" 오히려 작고" if ss.mape_simple < ts.mape_simple else " 비슷하거나 크고")
        + f", log 오차 흩어짐도 {ss.spread_simple:.3f} vs {ieyo(format(ts.spread_simple, '.3f'))}."
        f" (분할 conformal의 범위는 log에서 모든 셀에 같은 폭이라, 예측값 대비 ±{cf['test'].rel_half * 100:.0f}%로"
        " 두 나누기가 같은 것은 계산 방식 때문이에요. 이 가설의 근거가 아니에요.)"
        f" 셀 {n['secondary_test']}개로는 이 가설을 확정할 수 없고, 새 묶음이 학습 셀과 같은 주머니에서 나왔다는 보장도 없어요.",
    ]
    risky = {(k, m): c(k, m, "test", 90).below for k in ("simple", "full") for m in ("boot_resid", "conformal")}
    worst = max(risky, key=risky.get)
    tail = (1 - 0.9) / 2                                   # 90% 범위가 위아래로 고르게 빗나갈 때 아래쪽 몫
    lines += [
        f"7. **위험한 쪽 빗나감:** 실제 수명이 90% 범위보다 **짧았던** 테스트 셀은 네 경우 중 가장 많을 때"
        f" {_pct(risky[worst])}(셀 {n['test']}개 중 {round(risky[worst] * n['test'])}개,"
        f" {SHORT_NAMES[worst[0]]} + {unc.METHOD_NAMES[worst[1]]})였어요"
        f" (범위가 위아래로 고르게 빗나간다면 약 {_pct(tail)}, 셀 {n['test']}개 중 약 {round(tail * n['test'])}개)."
        " 보증 수명을 범위의 아래 끝으로 잡는다면, 이 비율이 \"보증보다 일찍 죽는 셀\"의 대략적인 비율이에요.",
        "",
        "## 이 결과가 말해 주는 것",
        "",
        "- **범위를 붙일 때는 '모델의 불확실성'과 '셀 사이 편차'를 모두 넣어야 해요.** 모델만 흔드는 범위는 보기엔 깔끔하지만 과신이에요.",
        "- **범위는 만들었으면 반드시 확인해요.** \"90%\"라는 이름표가 아니라, 학습에 안 쓴 셀에서 실제로 몇 %가 들어오는지가 범위의 실력이에요.",
        f"- **현업으로 옮기면:** 새 셀의 수명을 \"예측 {mid['pred']:,.0f}사이클\" 대신 \"90% 범위 아래 끝 {mid['lo90']:,.0f}사이클\"처럼"
        " 보고하고, 새 묶음(새 소재·새 공정 조건)이 학습 셀과 다를수록 그 범위를 덜 믿어야 해요.",
        "",
        "## 한계",
        "",
        f"- **셀이 적어요.** 테스트 {n['test']}개 · 2차 테스트 {n['secondary_test']}개라서 정직한 90% 범위도 적중률이"
        f" {_band(chance[('test', 90)])}·{_band(chance[('secondary_test', 90)])} 사이에서 우연히 흔들려요(이항분포)."
        f" 같은 보정 셀을 함께 쓰는 흔들림까지 넣으면(분할 conformal, 베타-이항분포) {_band(tb[('test', 90)])}·"
        f"{_band(tb[('secondary_test', 90)])}로 더 넓어요. 몇 %p 차이로 방법의 우열을 가릴 수 없어요.",
        "- **보장은 '평균적으로'예요.** 분할 conformal의 약속은 새 셀이 보정 셀과 같은 주머니에서 나왔다는 가정(교환 가능성) 아래에서,"
        " 여러 번 했을 때의 평균이에요. 셀 하나하나나 b3 같은 새 묶음에 대한 보장이 아니에요.",
        "- **폭이 모든 셀에 같아요(log 기준).** 분할 conformal은 어떤 셀이 더 불확실한지 구분하지 않아요."
        " 특징이 극단적인 셀이라고 범위를 더 넓히지 않아요.",
        "- **부트스트랩 표본에는 같은 셀이 여러 번 들어가요.** 그래서 6개 특징 모델 안의 교차 검증(규제 세기 고르기)이"
        " 조금 낙관적일 수 있어요.",
        "- **6개 특징 모델의 부트스트랩은 무작위 나누기 실험에서 뺐어요** (계산 시간). 그 모델의 흔들림은 분할 conformal로만 봤어요.",
        "- **관찰 데이터예요.** b3가 왜 다르게(또는 비슷하게) 보이는지는 가설이고, 충전 조건·시험 시기·셀 묶음 중 무엇 때문인지"
        " 가릴 수 없어요.",
        "- 분위수 회귀(quantile regression), 베이지안 회귀 같은 다른 방법은 시도하지 않았어요.",
        "",
        "## 시행착오",
        "",
    ]
    lines += trials(months=_months_between_batches(), n_cal=n_cal, nth_big=nth_big,
                    rel=cf["test"].rel_half, repeat_binom=theory["repeat_binom"],
                    repeat_seen=(a_cf.p10, a_cf.p90), narrow90=narrow[90])
    lines += ["", "더 해 볼 만한 응용은 [응용 아이디어](ideas.md)에 정리해 두었어요.", ""]
    return "\n".join(lines)


def trials(months: float, n_cal: int, nth_big: int, rel: float, repeat_binom: tuple[float, float],
           repeat_seen: tuple[float, float], narrow90: float) -> list[str]:
    """이번 작업에서 실제로 잘못 짐작했거나 잘못 본 것. 숫자는 손으로 쓰지 않고 계산값을 넣는다."""
    return [
        "| 상황 | 처음 생각 | 어떻게 알아챘나 | 원인 → 수정 |",
        "|---|---|---|---|",
        "| 2차 테스트(b3)의 범위 | b3는 시기·충전 조건이 달라 어느 모델이든 범위가 덜 맞을 것(과신)이라고 짐작 |"
        " 계산해 보니 논문 나누기에서는 네 경우 모두 우연의 폭 안. 무작위 나누기에서는 6개 특징 모델만 나쁜 쪽 끝이 낮아짐 |"
        " 짐작을 결론으로 쓰지 않고, 'b3를 학습에서 뺀 무작위 나누기'를 따로 돌려 숫자로 비교. 다른 점(수명·특징)과"
        " 비슷한 점(log 오차 크기)을 나눠 적고 '가설'로 표시 |",
        "| 문서 초안의 숫자 | 문서를 만드는 코드 안에 예시 숫자(셀 수, 보정 셀 수, 배치 사이 간격 등)를 손으로 적음 |"
        " 문서 규칙(숫자는 코드가 계산)과 맞춰 보다 발견. 배치 사이 간격은 날짜로 계산하니 어림한 값과 달랐음 |"
        f" 모두 데이터·설정에서 계산하도록 바꿈 (예: 시험 시작일 간격 약 {months:.0f}개월, 보정 셀 {n_cal}개 중"
        f" {nth_big}번째로 큰 오차) |",
        "| 가짜 데이터 테스트 | 평가 셀만 많으면 정직한 범위의 적중률이 약속에 가깝게 나올 것 | 테스트가 실패(90% 범위인데"
        " 더 적게 맞음) → 학습 셀의 오차 크기를 직접 찍어 봄 | 학습 셀이 적으면 학습 셀에 실제로 붙은 잡음이 우연히 작게 나와"
        " 범위도 좁아짐. 코드가 아니라 테스트의 짐작이 틀림 → 학습 셀도 크게 잡음 |",
        "| 요약표 열 이름 | 중앙값 열을 `median`으로 이름 붙임 | 문서를 만들다 오류 | pandas 표의 한 줄에서 `r.median`이"
        " 값이 아니라 함수로 읽힘 → 열 이름을 `mid`로 바꿈 |",
        "| 그래프 범례 | 범례를 그래프 안 빈 곳에 둠 | 그림을 직접 열어 보니 선·점과 겹침 | 범례를 그래프 아래로 옮김 |",
        "| 2차 테스트 가설의 근거 (검증 단계) | 분할 conformal 90% 범위의 예측값 대비 비율이 두 나누기 모두"
        f" ±{rel * 100:.0f}%로 같으니, b3도 % 오차가 비슷하다는 근거라고 씀 | 별도 검증에서 conformal 폭 계산식을 다시 보다 발견 |"
        " log에서 폭이 모든 셀에 같아 이 비율은 계산상 항상 같음(근거가 될 수 없음) → 근거를 MAPE와 log 오차 흩어짐으로 바꿈 |",
        "| 우연의 폭 (검증 단계) | 이항분포 5~95%면 '우연'을 충분히 담는다고 봄 | 무작위 나누기에서 본 흔들림"
        f"({_band(repeat_seen)})이 이항분포({_band(repeat_binom)})보다 넓은 것을 보고 입문 자료를 다시 확인 |"
        " 모든 평가 셀이 같은 보정 셀을 쓰면 적중률은 베타-이항분포를 따름 → 분할 conformal의 이론 폭을 함께 적고,"
        " 50% 범위가 '넉넉했다'는 경우를 다시 셈 |",
        "| 6개 특징 모델의 90% 폭 (검증 단계) | 논문 나누기 한 번에서 90% 폭이 더 넓은 것을 '좁은 범위에서만 이득'이라고"
        f" 일반화 | 무작위 나누기 결과와 맞춰 보니 90% 범위도 {_pct(narrow90)}의 나누기에서 6개 특징 모델이 더 좁음 |"
        " 나누기 한 번으로 일반화함 → 무작위 나누기의 중앙값과 짝 비교 비율을 같이 적음 |",
    ]


if __name__ == "__main__":
    build()
