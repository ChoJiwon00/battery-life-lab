"""급속충전 조건과 수명 분석을 돌려 docs/policies.md와 그래프를 만든다.

    python -m battery_lab.policies_report

문서의 숫자는 모두 여기서 계산한다 (손으로 고치지 않는다).
"""

import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from .mdtext import write_doc
from . import policies as pol
from .figures import (AXIS, BLUES, GRID, INK, INK_2, LIFE_TICKS, MUTED, SPLITS, SURFACE, WARN, _save, _style,
                      _subtitle)
from .korean import euro, ieyo
from .quality import NOMINAL_AH, check_cell, load_processed
from .sources import BATCH_BY_KEY, LICENSE, PROCESSED, PROJECT_URL, ROOT

DOCS = ROOT / "docs"
_CAT = [color for _, color in SPLITS.values()]  # 검증된 범주색 세 가지(파랑·주황·청록)를 고정 순서로
BATCH_COLORS = {"b1": _CAT[0], "b2": _CAT[1], "b3": _CAT[2]}
SEPARATE = {"low_rate": ("낮은 SOC(0~40%) 전류 +1C", _CAT[0]), "high_rate": ("높은 SOC(40~80%) 전류 +1C", _CAT[1]),
            "shift": ("뒤쪽 쏠림 +1C", _CAT[2])}


def _batch_name(b: str) -> str:
    return f"{b} ({BATCH_BY_KEY[b].date[:7]})" if b in BATCH_BY_KEY else b


def _short(policy: str) -> str:
    return pol.base_policy(policy)


# ---------------------------------------------------------------- 그래프

def fig_time_check(check: pd.DataFrame, path: Path) -> Path:
    """계산한 0→80% 시간 vs 측정한 충전 시간. 대각선에서 벗어난 셀은 이유를 직접 적는다."""
    _style()
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    lo = min(8.5, float(np.nanmin(check[["ideal_min", "measured_min"]].to_numpy())) - 0.3)
    hi = max(13.8, float(np.nanmax(check[["ideal_min", "measured_min"]].to_numpy())) + 0.3)
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=1)
    mid = lo + (hi - lo) * 0.78
    ax.text(mid + 0.3, mid - 0.3, "계산 = 측정", ha="center", va="center", rotation=45, rotation_mode="anchor",
            color=MUTED, fontsize=9, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1}, zorder=2)
    for b, g in check.groupby("batch"):
        ok, bad = g[~g["flagged"]], g[g["flagged"]]
        ax.scatter(ok["ideal_min"], ok["measured_min"], s=30, color=BATCH_COLORS.get(b, INK_2), edgecolor=SURFACE,
                   linewidth=1, zorder=3, label=f"{_batch_name(b)} · {len(g)}개")
        ax.scatter(bad["ideal_min"], bad["measured_min"], s=64, color=BATCH_COLORS.get(b, INK_2), edgecolor=INK,
                   linewidth=1.1, zorder=4)
    # 이유별 번호를 점 옆에 달고, 설명은 비어 있는 왼쪽 위에 한 덩어리로 (화살표가 서로 엇갈리지 않게)
    order = {"표기보다 약한 전류": 0, "3.6V": 1, "표기보다 센 전류": 2}
    flagged = check[check["flagged"]]
    reasons = sorted(flagged["reason"].unique(), key=lambda r: next((v for k, v in order.items() if r.startswith(k)), 9))
    notes = []
    for n, reason in enumerate(reasons):
        g = flagged[flagged["reason"] == reason]
        mark = "①②③④⑤⑥"[n] if n < 6 else f"({n + 1})"
        x, y = g["ideal_min"].median(), g["measured_min"].median()
        first = g.iloc[0]
        if reason == "표기보다 약한 전류":
            text = (f"{first['batch']} '{_short(first['policy'])}' {len(g)}개: 실제 전류 {g['c1_real'].median():.2f}C\n"
                    f"     → 0→80%에 {y:.1f}분 (표기대로면 {x:.1f}분)")
        elif reason == "표기보다 센 전류":
            text = (f"{first['batch']} '{first['policy']}': 실제 C2 {first['c2_real']:.2f}C\n"
                    f"     → 표기 '{first['c2']:g}'로 계산하면 {x:.1f}분, 실제 {y:.1f}분")
        elif reason.startswith("3.6V"):
            text = (f"{', '.join(sorted(g['batch'].unique()))} {len(g)}개: 높은 SOC에서 3.6V 상한에 닿아 전류 감소\n"
                    f"     → {g['measured_min'].min():.1f}~{g['measured_min'].max():.1f}분 (표기대로면 {x:.1f}분)")
        else:
            text = f"{len(g)}개: {reason}"
        notes.append(f"{mark} {text}")
        left = reason.startswith("3.6V")  # 세로로 늘어선 묶음은 왼쪽에 번호
        ax.annotate(mark, (x, y), xytext=(-9 if left else 9, 0), textcoords="offset points", ha="right" if left
                    else "left", va="center", fontsize=11, color=INK, fontweight="bold", zorder=6)
    ax.text(0.03, 0.97, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=8.8, color=INK_2,
            linespacing=1.5, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2}, zorder=6)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="표기대로 계산한 0→80% 충전 시간 (분, 정전압 무시)",
           ylabel="측정한 충전 시간 (분, 사이클 2~100 중앙값)")
    ax.set_aspect("equal")
    ax.set_title(f"표기대로 충전됐을까? {len(check)}개 중 {int(check['flagged'].sum())}개 셀은 달라요")
    _subtitle(ax, f"검은 테두리 = 계산과 {pol.TIME_TOL:g}분 넘게 다르거나 실제 전류가 표기와 {pol.RATE_TOL:.0%} 넘게 다른 셀")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2, fontsize=9)
    return _save(fig, path)


def _iso_time(ax, minutes: float, x_max: float, y_lim: tuple[float, float]) -> None:
    """0→80% 시간이 같은 (낮은 SOC 전류, 높은 SOC 전류) 조합: 40/L + 40/H 가 일정한 곡선."""
    inv = minutes / 24  # 0.4/L + 0.4/H = 분/60  →  1/L + 1/H = 분/24
    xs = np.linspace(1 / inv * 1.001, x_max, 300)
    ys = 1 / (inv - 1 / xs)
    keep = (ys >= y_lim[0]) & (ys <= y_lim[1])
    if not keep.any():
        return
    ax.plot(xs[keep], ys[keep], color=AXIS, lw=0.9, ls=(0, (4, 3)), zorder=1)
    k = np.flatnonzero(keep)[-1]
    ax.text(xs[k] - 0.05, ys[k] + 0.07, f"{minutes:g}분", ha="right", va="bottom", fontsize=8, color=MUTED,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.5}, zorder=2)


def fig_map(groups: pd.DataFrame, path: Path) -> Path:
    """충전 조건 지도: 배치마다 (낮은 SOC 평균 전류, 높은 SOC 평균 전류), 색 = 수명 중앙값, 크기 = 셀 수."""
    _style()
    batches = sorted(groups["batch"].unique())
    fig, axes = plt.subplots(1, len(batches), figsize=(4.6 * len(batches) + 1.2, 4.9), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    life = groups["life_median"]
    norm = LogNorm(vmin=float(life.min()), vmax=float(life.max()))
    x_lim = (np.floor(groups["low_rate"].min() * 2) / 2 - 0.3, np.ceil(groups["low_rate"].max() * 2) / 2 + 0.6)
    y_lim = (np.floor(groups["high_rate"].min() * 2) / 2 - 0.3, np.ceil(groups["high_rate"].max() * 2) / 2 + 0.6)
    for ax, b in zip(axes, batches):
        g = groups[groups["batch"] == b].sort_values("life_median", ascending=False)
        for m in (9, 10, 11, 12, 13):
            _iso_time(ax, m, x_lim[1], y_lim)
        d = (max(x_lim[0], y_lim[0]), min(x_lim[1], y_lim[1]))
        ax.plot(d, d, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.scatter(g["low_rate"], g["high_rate"], s=26 + 20 * g["cells"], c=BLUES(norm(g["life_median"])),
                   edgecolor=INK_2, linewidth=0.6, zorder=3)
        ax.set_title(f"{_batch_name(b)} · 조건 {len(g)}개 · 셀 {int(g['cells'].sum())}개", fontsize=11, pad=8)
        ax.set_xlabel("낮은 SOC(0~40%) 평균 전류 (C-rate)")
    axes[0].set_ylabel("높은 SOC(40~80%) 평균 전류 (C-rate)")
    axes[0].set(xlim=x_lim, ylim=y_lim)
    axes[0].text(x_lim[0] + 0.15, y_lim[1] - 0.12, "↑ 뒤쪽(높은 SOC)이 더 셈", ha="left", va="top", fontsize=8.5,
                 color=INK_2, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.5}, zorder=4)
    axes[0].text(x_lim[1] - 0.15, y_lim[0] + 0.12, "앞쪽(낮은 SOC)이 더 셈 →", ha="right", va="bottom",
                 fontsize=8.5, color=INK_2, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.5}, zorder=4)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=BLUES)
    cb = fig.colorbar(sm, ax=list(axes), pad=0.012, fraction=0.025)
    ticks = [t for t in LIFE_TICKS if norm.vmin <= t <= norm.vmax]
    cb.set_ticks(ticks, labels=[f"{t:,}" for t in ticks])
    cb.ax.minorticks_off()
    cb.set_label("수명 중앙값 (사이클)", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    fig.suptitle("충전 조건 지도: 배치마다 시험한 영역이 달라요", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK, y=1.04)
    fig.text(0.01, 0.975, "원시 기록에서 잰 실제 평균 전류 · 점선 = 0→80% 시간이 같은 조합 · 점 = 배치·조건 하나 "
             "(크기 = 셀 수, 진할수록 오래 버팀) · 대각선 = 앞뒤 같은 전류", color=INK_2, fontsize=9.5)
    return _save(fig, path)


def _effect_rows(effects: list[dict]) -> list[dict]:
    rows = []
    for e in effects:
        if e["source"] != "실측":
            continue
        for term, v in e["terms"].items():
            label, color = SEPARATE[term]
            rows.append({"label": f"{e['name']} · {label}", "color": color, **v, "model": e["name"]})
    return rows


def fig_effects(effects: list[dict], path: Path) -> Path:
    """'+1C마다 수명이 몇 % 변하나'와 90% 범위 (조건 덩어리 부트스트랩)."""
    _style()
    rows = _effect_rows(effects)
    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(rows) + 1.6))
    lo = min(r["pct_lo"] for r in rows)
    hi = max(r["pct_hi"] for r in rows)
    for y, r in enumerate(rows):
        ax.plot([r["pct_lo"], r["pct_hi"]], [y, y], color=r["color"], lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([r["pct"]], [y], s=56, color=r["color"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.text(hi + (hi - lo) * 0.06, y, f"{r['pct']:+.0f}%  ({r['pct_lo']:+.0f} ~ {r['pct_hi']:+.0f})",
                va="center", ha="left", fontsize=9, color=INK_2)  # 값은 오른쪽 한 줄에 모아 0 선과 겹치지 않게
    for y in range(1, len(rows)):
        if rows[y]["model"] != rows[y - 1]["model"]:
            ax.axhline(y - 0.5, color=GRID, lw=1)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(range(len(rows)), labels=[r["label"] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.42)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("수명 변화 (%) · 0보다 왼쪽 = 수명이 짧아지는 쪽")
    ax.set_title("전류를 1C 더 세게 했을 때 함께 나타난 수명 변화")
    _subtitle(ax, f"점 = 직선 모델의 추정 · 선 = 90% 범위 (같은 조건 셀을 묶어 {pol.BOOT:,}번 다시 뽑기) · 관찰이지 인과가 아니에요")
    return _save(fig, path)


def fig_replicates(groups: pd.DataFrame, table: pd.DataFrame, cross: set[str], real_c1: dict[str, float],
                   path: Path) -> Path:
    """같은 배치·같은 조건 셀이 2개 이상인 조건: 셀마다 수명(점)과 최소~최대(선)."""
    _style()
    g = groups.sort_values(["batch", "life_median"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 0.3 * len(g) + 1.9))
    for y, r in g.iterrows():
        color = BATCH_COLORS.get(r["batch"], INK_2)
        lives = table.loc[table["group"] == r["group"], "cycle_life"].to_numpy()
        ax.plot([lives.min(), lives.max()], [y, y], color=color, lw=1.4, alpha=0.6, zorder=2)
        ax.scatter(lives, np.full(len(lives), y), s=26, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for y, r in g.iterrows():
        ax.text(1.01, y, f"×{r['ratio']:.2f}", transform=trans, va="center", ha="left", fontsize=8.5, color=INK_2)
    ax.text(1.01, -1.0, "최대÷최소", transform=trans, va="center", ha="left", fontsize=8.5, color=MUTED)
    labels = [f"{r['batch']} · {_short(r['policy'])}"
              + (f" (실제 C1 {real_c1[r['group']]:.2f}C)" if r["group"] in real_c1 else "")
              + (" ◆" if _short(r["policy"]) in cross else "") for _, r in g.iterrows()]
    ax.set_yticks(range(len(g)), labels=labels, fontsize=8.8)
    for tick, (_, r) in zip(ax.get_yticklabels(), g.iterrows()):
        tick.set_color(BATCH_COLORS.get(r["batch"], INK_2))
    ax.set_ylim(len(g) - 0.5, -1.5)
    ax.set_xscale("log")
    shown = table.loc[table["group"].isin(g["group"]), "cycle_life"]
    x_lim = (shown.min() * 0.88, shown.max() * 1.12)  # 반복 셀이 있는 범위만 (빈 왼쪽을 없앤다)
    ticks = [t for t in (300, 400, 500, 700, 1000, 1500, 2000) if x_lim[0] <= t <= x_lim[1]]
    ax.set_xlim(*x_lim)
    ax.set_xticks(ticks, labels=[f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("사이클 수명")
    ax.set_title("같은 조건으로 시험한 셀끼리도 수명이 달라요", pad=34)  # 키 큰 그림이라 부제와 겹치지 않게
    _subtitle(ax, "한 줄 = 같은 배치·같은 표기 · 점 = 셀 · ◆ = 같은 표기가 다른 배치에도 있음 · b3 표기의 -newstructure는 생략")
    handles = [Line2D([], [], marker="o", ls="", color=BATCH_COLORS[b], label=_batch_name(b))
               for b in sorted(g["batch"].unique())]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.06), ncol=3,
              labelcolor=INK_2, fontsize=9)
    return _save(fig, path)


def _try_labels(fig, ax, labels, points, top_band: float) -> tuple[int, list]:
    """라벨을 차례로 놓는다. 점 둘레의 몇 자리 중 다른 점·라벨·축 밖·맨 위 설명 줄과 덜 겹치는 자리를 고른다."""
    spots = [(10, 8, "left", "bottom"), (10, -8, "left", "top"), (-10, 8, "right", "bottom"),
             (-10, -8, "right", "top"), (0, 14, "center", "bottom"), (0, -14, "center", "top")]
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    ceiling = frame.y1 - frame.height * top_band
    dots = ax.transData.transform(points[np.isfinite(points).all(axis=1)])
    placed, made, total = [], [], 0
    for x, y, text in labels:
        best = None
        for dx, dy, ha, va in spots:
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                              fontsize=8.5, color=INK, zorder=6,
                              bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1, "alpha": 0.85})
            box = ann.get_window_extent(renderer).expanded(1.04, 1.08)
            hits = int(((dots[:, 0] > box.x0) & (dots[:, 0] < box.x1) & (dots[:, 1] > box.y0)
                        & (dots[:, 1] < box.y1)).sum())
            outside = box.x0 < frame.x0 or box.x1 > frame.x1 or box.y0 < frame.y0 or box.y1 > ceiling
            score = hits + 100 * outside + 1000 * sum(box.overlaps(o) for o in placed)
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, ann, box)
            else:
                ann.remove()
            if score == 0:
                break
        placed.append(best[2])
        made.append(best[1])
        total += best[0]
    return total, made


def _place_labels(fig, ax, labels, points, top_band: float = 0.08) -> None:
    """왼쪽부터 놓기와 오른쪽부터 놓기를 둘 다 해 보고 덜 겹치는 쪽을 남긴다."""
    first, made = _try_labels(fig, ax, labels, points, top_band)
    if first == 0:
        return
    for ann in made:
        ann.remove()
    second, made_rev = _try_labels(fig, ax, labels[::-1], points, top_band)
    if second <= first:
        return
    for ann in made_rev:
        ann.remove()
    _try_labels(fig, ax, labels, points, top_band)


def fig_frontier(table: pd.DataFrame, front: pd.DataFrame, real_c1: dict[str, float], path: Path) -> Path:
    """측정한 충전 시간 vs 수명. 시간 구간마다 수명 중앙값이 가장 긴 조건(★)을 '관찰된 최고'로 표시."""
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    f = front[front["groups"] > 0]
    for k, r in enumerate(front.itertuples()):
        if k % 2 == 0:
            ax.axvspan(r.lo, r.hi, color=GRID, alpha=0.35, lw=0, zorder=0)
        ax.text((r.lo + r.hi) / 2, 0.985, f"{r.lo:g}~{r.hi:g}분", transform=blended_transform_factory(
            ax.transData, ax.transAxes), ha="center", va="top", fontsize=8, color=MUTED)
    for b, g in table.groupby("batch"):
        ax.scatter(g["measured_min"], g["cycle_life"], s=22, color=BATCH_COLORS.get(b, INK_2), alpha=0.85,
                   edgecolor=SURFACE, linewidth=0.6, zorder=3, label=f"{_batch_name(b)} 셀")
    ax.plot(f["best_minutes"], f["best_life"], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=4)
    ax.scatter(f["best_minutes"], f["best_life"], s=170, marker="*", color=WARN, edgecolor=INK, linewidth=0.7,
               zorder=5, label="구간별 관찰된 최고 (조건의 수명 중앙값)")
    x_max = float(front["hi"].max())
    ax.set_yscale("log")
    ticks = [t for t in (150, 200, 300, 500, 1000, 2000) if table["cycle_life"].min() * 0.9 <= t
             <= table["cycle_life"].max() * 1.2]
    ax.set_yticks(ticks, labels=[f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(table["cycle_life"].min() * 0.85, table["cycle_life"].max() * 1.6)
    ax.set_xlim(float(front["lo"].min()), x_max)
    labels = []
    for r in f.itertuples():
        text = f"{_short(r.best_policy)}\n({r.best_batch}, 셀 {r.best_cells}개)"
        if r.best_group in real_c1:
            text += f"\n실제 C1 {real_c1[r.best_group]:.2f}C"
        labels.append((r.best_minutes, r.best_life, text))
    points = np.vstack([np.column_stack([table["measured_min"], table["cycle_life"]]),
                        np.column_stack([f["best_minutes"], f["best_life"]])])  # 셀 점 + ★
    _place_labels(fig, ax, labels, points)
    ax.set(xlabel="측정한 0→80% 충전 시간 (분, 사이클 2~100 중앙값)", ylabel="사이클 수명")
    ax.set_title("빠르면서 오래 간 조건: 관찰된 최고 (최적화한 결과가 아님)")
    _subtitle(ax, f"★ = {_bin_width()} 구간마다 수명 중앙값이 가장 긴 배치·조건 · 시험한 조건 안에서만 고른 것이라 더 좋은 조건이 있을 수 있어요")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2, fontsize=8.8)
    return _save(fig, path)


# ---------------------------------------------------------------- 숫자 → 문장 도우미

def _bin_width() -> str:
    widths = set(np.round(np.diff(pol.TIME_BINS), 3))
    return f"{widths.pop():g}분" if len(widths) == 1 else "시간"


def _near_fixed(table: pd.DataFrame, batches=("b2", "b3"), half: float = 0.5) -> dict:
    """b2·b3 셀 중 측정한 충전 시간이 중앙값 ± half분 안에 든 비율 (시간이 '거의 고정'인지 숫자로 보이기)."""
    t = table.loc[table["batch"].isin(batches), "measured_min"].dropna()
    if t.empty:
        return {"share": np.nan, "lo": np.nan, "hi": np.nan}
    med = t.median()
    inside = t[(t >= med - half) & (t <= med + half)]
    return {"share": len(inside) / len(t), "lo": inside.min(), "hi": inside.max()}


def _span(lo: float, hi: float, fmt: str = ".2f") -> str:
    """같으면 한 값, 다르면 'lo~hi'."""
    a, b = format(lo, fmt), format(hi, fmt)
    return a if a == b else f"{a}~{b}"


def _pm(sd_log10: float) -> str:
    """log10 수명의 표준편차 → '±x%' (한 표준편차만큼 위로)."""
    return f"±{(10 ** sd_log10 - 1) * 100:.0f}%"


def _rng(v: dict) -> str:
    return f"{v['pct_lo']:+.0f} ~ {v['pct_hi']:+.0f}%"


def _verdict(v: dict) -> str:
    return "범위가 0을 포함하지 않음" if v["clear"] else "범위가 0을 포함 → 방향을 확신할 수 없음"


def _gap_has_zero(e: dict) -> bool:
    """두 전류 모델의 (높은 − 낮은) 차이 90% 범위가 0을 포함하나."""
    return bool(e.get("gap_pct_lo", np.nan) <= 0 <= e.get("gap_pct_hi", np.nan))


def _find(effects: list[dict], name: str, source: str = "실측") -> dict:
    return next(e for e in effects if e["name"] == name and e["source"] == source)


def stitched_cells(processed: Path) -> pd.DataFrame:
    """다음 배치 기간에 이어 시험한 셀: 이어 붙인 첫 사이클과 그 지점의 용량 변화(데이터 품질 문서와 같은 계산)."""
    cells, summary = load_processed(processed)
    cols = ["cell_id", "batch", "next_batch", "seam_cycle", "seam_first_pct"]
    rows = []
    for meta in cells[cells["seam_cycle"].astype(str).str.strip() != ""].to_dict("records"):
        r = check_cell(meta, summary[summary["cell_id"] == meta["cell_id"]], None)
        nxt = re.match(r"b\d+", str(meta.get("merged_from", "")))
        rows.append({"cell_id": r.cell_id, "batch": r.batch, "next_batch": nxt.group(0) if nxt else "",
                     "seam_cycle": r.seam_cycle, "seam_first_pct": r.seam_first_pct})
    return pd.DataFrame(rows, columns=cols)


def _batch_kr(b: str) -> str:
    """'b2' → '2배치'."""
    return f"{b[1:]}배치" if re.fullmatch(r"b\d+", b) else b


def _cell_list(ids: set[str]) -> str:
    """같은 배치에서 번호가 3개 이상 이어지면 'b1c0~b1c4', 아니면 '·'로 잇기."""
    parsed = sorted((m.group(1), int(m.group(2)), i) for i in ids if (m := re.fullmatch(r"(b\d+c)(\d+)", i)))
    if len(parsed) == len(ids) >= 3 and len({p for p, _, _ in parsed}) == 1:
        nums = [n for _, n, _ in parsed]
        if nums == list(range(nums[0], nums[0] + len(nums))):
            return f"{parsed[0][2]}~{parsed[-1][2]}"
    return "·".join(i for *_, i in parsed) if len(parsed) == len(ids) else "·".join(sorted(ids))


def _bins_text(bins: list[tuple[float, float]]) -> str:
    """이어지는 시간 구간은 하나로: [(11.5, 12.5), (12.5, 13.5)] → '11.5~13.5분'."""
    spans: list[list[float]] = []
    for lo, hi in sorted(bins):
        if spans and np.isclose(spans[-1][1], lo):
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])
    return "·".join(f"{lo:g}~{hi:g}분" for lo, hi in spans)


def _stitched_note(table: pd.DataFrame, front: pd.DataFrame, lv: pd.DataFrame,
                   stitched: pd.DataFrame | None) -> list[str]:
    """이어 시험한 셀이 누구이고 어느 결과에 들어가는지 숫자로 적은 한계 항목 (그런 셀이 없으면 빈 목록)."""
    if stitched is None or stitched.empty:
        return []
    st = stitched[stitched["cell_id"].isin(table["cell_id"])]
    out = []
    for batch, g in st.groupby("batch"):
        bt = table[table["batch"] == batch]
        ids = set(g["cell_id"])
        mine = bt[bt["cell_id"].isin(ids)]
        top = set(bt.nlargest(len(ids), "cycle_life")["cell_id"]) == ids    # 그 배치에서 수명이 가장 긴 셀들인가
        groups = list(dict.fromkeys(mine["group"]))
        whole = all(set(bt.loc[bt["group"] == grp, "cell_id"]) <= ids for grp in groups)
        pols = ", " + "·".join(f"`{p}`" for p in dict.fromkeys(mine["policy"])) + " 셀 전부" if whole else ""
        life = mine.set_index("cell_id")["cycle_life"]
        alive = bool((life.reindex(g["cell_id"]).to_numpy() > g["seam_cycle"].to_numpy()).all())
        nxt = "·".join(_batch_kr(b) for b in sorted(set(g["next_batch"]) - {""})) or "다음 배치"
        seam = _span(g["seam_cycle"].min(), g["seam_cycle"].max(), ",.0f")
        text = (f"- {batch}에서 {'수명이 가장 긴 ' if top else ''}셀 {len(ids)}개({_cell_list(ids)}{pols})는 "
                + (f"{_batch_kr(batch)} 기록이 끝날 때 아직 수명이 남아 있어서, " if alive else "")
                + f"{seam}사이클부터 {nxt} 기간에 이어서 시험했어요.")
        bins = [(r.lo, r.hi) for r in front.itertuples() if r.groups > 0 and r.best_group in set(groups)]
        targets = []
        if bins:
            targets.append(f"결과 6의 {_bins_text(bins)} \"관찰된 최고\"")
        if batch == "b1" and "b1" in lv.index:                         # 읽는 법 2는 b1 안의 경향을 말한다
            targets.append("읽는 법 2의 b1 경향")
        where = "와 ".join(targets)                                    # 앞 항목은 늘 '최고"'로 끝나 '와'
        if where and top:
            text += f" 그래서 {where}에 들어간 가장 긴 수명들은 중간에 쉬는 기간을 사이에 두고 이어 간 수명이에요"
        elif where:
            text += f" 그래서 {where}에는 중간에 쉬는 기간을 사이에 두고 이어 간 수명이 들어 있어요"
        elif top:
            text += f" 그래서 {batch}의 가장 긴 수명들은 중간에 쉬는 기간을 사이에 두고 이어 간 수명이에요"
        else:
            text += " 수명이 중간에 쉬는 기간을 사이에 두고 이어져 있어요"
        change = g["seam_first_pct"].dropna()
        link = "[데이터 품질](data_quality.md#이어-붙인-셀-5개의-이음매)"
        text += (f" (이어 붙인 지점의 용량 변화 {_span(change.min(), change.max(), '+.1f')}%, {link})."
                 if len(change) else f" ({link} 참고).")
        out.append(text)
    return out


# ---------------------------------------------------------------- 실행

def build(processed: Path = PROCESSED, docs: Path = DOCS, progress=print) -> dict:
    progress("충전 조건 읽기 · 원시 기록에서 실제 전류 재기")
    table, unparsed = pol.cell_table(processed)
    progress(f"질문별 계산 (부트스트랩 {pol.BOOT:,}번 · 순열 {pol.PERM:,}번)")
    res = pol.analyze(table)
    check, rep, front = res["time_check"], res["replicates"], res["frontier"]

    progress("중간 결과 저장 (data/processed/policies_*.csv)")
    table.to_csv(processed / "policies_cells.csv", index=False)
    res["groups"].to_csv(processed / "policies_groups.csv", index=False)
    front.to_csv(processed / "policies_frontier.csv", index=False)
    flat = [{"model": e["name"], "source": e["source"], "term": t, "cells": e["cells"], "groups": e["groups"],
             "r2": e["r2"], **v} for e in res["effects"] for t, v in e["terms"].items()]
    pd.DataFrame(flat).to_csv(processed / "policies_effects.csv", index=False)

    progress("그래프")
    fig_dir = docs / "figures"
    cross = set(res["across"]["base_policy"])
    off = check[check["reason"].str.contains("약한|센", regex=True)]
    real_c1 = {f"{r.batch} · {r.policy}": r.c1_real for r in off.itertuples()
               if abs(r.c1_real / r.c1 - 1) > pol.RATE_TOL}
    fig_time_check(check, fig_dir / "policies_time_check.png")
    fig_map(res["groups"], fig_dir / "policies_map.png")
    fig_effects(res["effects"], fig_dir / "policies_effects.png")
    fig_replicates(rep["groups"], table, cross, real_c1, fig_dir / "policies_replicates.png")
    fig_frontier(table, front, real_c1, fig_dir / "policies_frontier.png")

    write_doc(docs / "policies.md", markdown(table, unparsed, res, stitched_cells(processed)))
    pooled = _find(res["effects"], "전체 (배치 보정)")
    lvt = res["life_vs_time"].set_index("scope")
    return {
        "cells": len(table), "policies": int(table["policy"].nunique()), "unparsed": unparsed,
        "flagged": int(check["flagged"].sum()),
        "rho_time_all": float(lvt.loc["전체", "rho_time"]),
        "rho_time_b1": float(lvt.loc["b1", "rho_time"]) if "b1" in lvt.index else np.nan,
        "pooled_low_pct": pooled["terms"]["low_rate"]["pct"], "pooled_high_pct": pooled["terms"]["high_rate"]["pct"],
        "pooled_high_worse_share": pooled.get("second_worse_share", np.nan),
        "eta2": rep["all"]["eta2"], "eta2_null_mean": rep["all"]["null_mean"],
        "frontier": front.to_dict("records"),
    }


# ---------------------------------------------------------------- 문서

def markdown(table: pd.DataFrame, unparsed: list[str], res: dict, stitched: pd.DataFrame | None = None) -> str:
    """stitched: stitched_cells()의 결과 (다음 배치 기간에 이어 시험한 셀). 없으면 그 한계 항목을 뺀다."""
    check, lvt, rep, across, front = (res["time_check"], res["life_vs_time"], res["replicates"], res["across"],
                                      res["frontier"])
    lv = lvt.set_index("scope")
    fixed = _near_fixed(table)
    n_pol = table["policy"].nunique()
    ns = table[table["newstructure"]]
    unit = sorted(table.loc[table["unit_missing"], "policy"].unique())
    single = table.loc[table["single_step"], "policy"].nunique()
    raw_counts = table["raw_cycles"].value_counts().sort_index()
    raw_text = " · ".join(f"셀 {v}개는 {int(k)}개 사이클씩" for k, v in raw_counts.items())
    flagged = check[check["flagged"]]
    by_reason = flagged.groupby("reason", sort=False)

    lines = [
        "# 급속충전 조건과 수명: 빨리 충전하면 빨리 늙을까?",
        "",
        f"`python -m battery_lab.policies_report`가 자동으로 만든 문서예요 (만든 날 {date.today().isoformat()}).",
        "문서의 숫자는 모두 [policies.py](../battery_lab/policies.py)와 "
        "[policies_report.py](../battery_lab/policies_report.py)가 계산했어요.",
        "",
        "## 질문",
        "",
        f"이 데이터의 {len(table)}개 셀은 같은 모델의 셀을 같은 30°C 챔버에서, **급속충전 방식을 바꿔 가며** 시험했어요. "
        "다만 배치(시험 시기)마다 시험한 조건 묶음과 시험 절차가 조금씩 달라요(결과 2). "
        f"충전 조건 표기는 {n_pol}가지예요. 그래서 이렇게 물었어요. "
        "(SOC = 충전 상태. 0%는 빈 상태, 100%는 가득 찬 상태예요.)",
        "",
        "1. 0→80% **충전 시간이 짧을수록 수명도 짧을까?**",
        "2. 같은 시간이라도 **낮은 SOC(0~40%)에서 세게** 넣는 것과 **높은 SOC(40~80%)에서 세게** 넣는 것 중 "
        "어느 쪽이 짧은 수명과 더 함께 나타날까?",
        "   (가설: 더 많이 찬 상태(높은 SOC)에서 센 전류를 넣으면 리튬이 흑연 안으로 다 못 들어가고 표면에 쌓이는 "
        "**리튬 석출**이 일어나기 쉽다고 알려져 있어요.)",
        "3. **같은 조건으로 시험한 셀끼리는** 수명이 얼마나 비슷할까? 조건이 수명 차이를 얼마나 설명할까?",
        "4. **빠르면서도 오래 간 조건**은 무엇이었을까? (시험한 조건 중 관찰된 최고)",
        "",
        "## 왜 중요한가 (셀 개발·공정 개발 직무와 연결)",
        "",
        "- **셀 개발:** \"몇 분 만에 80%\" 같은 급속충전 사양과 수명 보증은 서로 부딪쳐요. 같은 시간 안에서 "
        "**어느 SOC 구간의 전류를 줄여야 수명을 지키는지**가 충전 프로토콜(충전 순서·전류 설계)의 핵심이에요.",
        "- **공정 개발:** 같은 조건으로 만든·시험한 셀끼리도 수명이 흔들려요(셀 간 편차. 만들 때의 산포가 원인 후보 중 하나예요). "
        "조건 A가 B보다 낫다고 말하려면 **그 차이가 셀 간 편차보다 커야** 해요. 셀을 조건마다 몇 개씩 만들지(실험 계획, DOE)의 근거가 돼요.",
        "- **데이터 다루기:** 시험 기록의 조건 표기가 실제와 다를 수 있어요. 이번에도 **원시 전류를 직접 재 보고서야** "
        "표기와 다른 셀을 찾았어요.",
        "",
        "## 방법",
        "",
        "- **표기 읽기:** `C1(Q1%)-C2` = SOC 0→Q1%는 C1, Q1→80%는 C2로 충전, 80%부터는 모든 셀이 1C CC-CV(일정한 전류로 채우다가 3.6V에 닿으면 전압을 고정하고 마저 채움)"
        " (C-rate 1C = 1시간에 공칭 용량 1.1Ah를 채우는 전류).",
        f"  - {n_pol}가지 표기를 모두 읽었어요 (읽지 못한 표기 **{len(unparsed)}개**"
        f"{': ' + ', '.join(f'`{u}`' for u in unparsed) if unparsed else ''}).",
        f"  - 3배치(b3) {len(ns)}개 셀의 {ns['policy'].nunique()}가지 표기에는 끝에 `-newstructure`가 붙어 있어요. "
        "따로 표시만 하고, 같은 표기 비교 때는 떼고 봐요. 이 표시는 b3 셀에만 있어서 **배치와 따로 떼어 볼 수 없어요.**",
        f"  - 끝의 C가 빠진 표기 {len(unit)}개({', '.join(f'`{u}`' for u in unit) if unit else '없음'})는 "
        "숫자를 그대로 C-rate로 읽고 표시해 두었어요.",
        f"  - Q1이 80%인 {single}가지는 한 단계 충전(처음부터 80%까지 같은 전류)이에요.",
        "- **계산한 시간:** (Q1 ÷ C1 + (80 − Q1) ÷ C2) ÷ 100 = 표기대로 0→80%에 걸리는 시간(h), ×60 하면 분. "
        "예: `6C(40%)-3C`는 (40÷6 + 40÷3) ÷ 100 = 0.2시간 = 12분. "
        "전압이 상한(3.6V)에 닿아 전류가 줄어드는 정전압 구간은 무시해요.",
        f"- **측정한 시간:** `summary.csv`의 충전 시간(chargetime)을 셀마다 사이클 {pol.EARLY[0]}~{pol.EARLY[1]}에서 "
        "중앙값으로 냈어요. 시험이 멈췄던 사이클의 수백 분짜리 값에 흔들리지 않게 평균 대신 중앙값을 써요.",
        f"- **실제 전류:** 원시 기록(`raw/`)에서 사이클 {pol.EARLY[0]}~{pol.EARLY[1]} 안의 사이클({raw_text})을 골라, "
        "충전 시작 뒤 넣은 용량 ÷ 1.1Ah로 SOC를 계산하고 SOC 40%·80%에 닿은 시각을 쟀어요. "
        f"80% 도달은 {pol.END_SOC:g}%로 판정해요(시행착오 4). "
        "**낮은 SOC 평균 전류** = 0~40% 폭 ÷ 걸린 시간, **높은 SOC 평균 전류** = 40~80% 폭 ÷ 걸린 시간 (C-rate). "
        "**뒤쪽 쏠림** = 높은 SOC 평균 전류 − 낮은 SOC 평균 전류 (양수면 뒤쪽이 더 셈).",
        f"- **수명과 비교:** 수명은 log10으로 바꿔서 봐요 (수명이 {table['cycle_life'].min():,.0f}~{table['cycle_life'].max():,.0f}사이클로 "
        "몇 배씩 퍼져 있어서 비율로 보는 게 자연스러워요).",
        "  - 순위 상관(스피어만 ρ): 값 대신 순위로 계산해서 극단값 몇 개에 덜 흔들려요. −1~+1.",
        "  - 직선 모델: log10 수명 = a + b × 전류 (+ 여러 배치를 합칠 때는 **배치마다 다른 출발점**, 배치 보정). "
        "b를 \"+1C마다 수명 몇 % 변화\"로 바꿔 보여 줘요.",
        f"  - 90% 범위: 같은 조건(배치·표기)의 셀을 한 덩어리로 묶어 다시 뽑는 부트스트랩을 {pol.BOOT:,}번 "
        f"(고정 시드 {pol.SEED}). 같은 조건 셀끼리는 닮아서, 셀을 하나하나 따로 뽑으면 범위가 실제보다 좁게 나와요.",
        f"- **반복 셀:** 같은 배치·같은 표기 셀이 2개 이상인 조건만 모아, 수명 차이 중 \"어느 조건이냐\"로 설명되는 몫(η², 에타 제곱)을 "
        f"계산하고, 조건 이름을 무작위로 {pol.PERM:,}번 섞었을 때의 값과 비교했어요 (섞어도 조건 수가 많으면 η²가 꽤 나와요).",
        f"- **관찰된 최고:** 측정한 충전 시간을 {_bin_width()} 구간({pol.TIME_BINS[0]:g}~{pol.TIME_BINS[-1]:g}분)으로 나누고, "
        "구간마다 **수명 중앙값이 가장 긴 배치·조건**을 골랐어요. 시험한 조건 중에서 고른 것이지 최적화한 결과가 아니에요.",
        "",
        "## 결과",
        "",
        "### 1. 표기대로 충전됐을까? (계산한 시간 vs 측정한 시간)",
        "",
        f"| 배치 | 셀 | 측정 − 계산 (중앙값) | {pol.TIME_TOL:g}분 넘게 다르거나 전류가 표기와 다른 셀 |",
        "|---|---|---|---|",
    ]
    for b, g in check.groupby("batch"):
        lines.append(f"| {_batch_name(b)} | {len(g)} | {g['diff_min'].median() * 60:+.0f}초 | {int(g['flagged'].sum())}개 |")
    lines += [
        "",
        f"*점검: 원시 기록으로 잰 0→80% 시간과 summary의 충전 시간은 셀마다 최대 {check['raw_gap'].abs().max() * 60:.0f}초 차이 나고, "
        f"{pol.RAW_TOL:g}분 넘게 다른 셀은 {int(check['raw_disagree'].sum())}개예요. 두 출처가 서로 맞아요.*",
        "",
        f"대부분은 계산과 측정이 수 초 안에서 맞아요. 다른 셀 **{len(flagged)}개**는 원시 전류를 재 보니 이유가 "
        f"{flagged['reason'].nunique()}가지로 나뉘었어요.",
        "",
        "| 이유 | 셀 | 표기 | 실제로 잰 전류 | 계산 → 측정 (분) | 실제 전류로 다시 계산 (분) |",
        "|---|---|---|---|---|---|",
    ]
    for reason, g in by_reason:
        cur = []
        if g["c1_real"].notna().any() and (abs(g["c1_real"] / g["c1"] - 1) > pol.RATE_TOL).any():
            cur.append(f"C1 {_span(g['c1_real'].min(), g['c1_real'].max())}C")
        if g["c2_real"].notna().any():
            cur.append(f"C2 {_span(g['c2_real'].min(), g['c2_real'].max())}C")
        cur = " · ".join(cur) if cur else "-"
        pols = ", ".join(f"`{p}`" for p in dict.fromkeys(g["policy"]))
        lines.append(f"| {reason} | {', '.join(g['cell_id'])} | {pols} | {cur} | "
                     f"{_span(g['ideal_min'].min(), g['ideal_min'].max())} → "
                     f"{_span(g['measured_min'].min(), g['measured_min'].max())} | "
                     f"{_span(g['real_ideal_min'].min(), g['real_ideal_min'].max())} |")
    weak = flagged[flagged["reason"] == "표기보다 약한 전류"]
    if flagged["reason"].str.startswith("3.6V").any():
        lines += ["", "*3.6V 상한에 닿은 셀은 전압을 3.6V로 붙잡은 채 전류가 계속 줄어들어요(정전압). "
                  "그래서 '실제로 잰 전류'는 그 단계 가운데 구간의 중앙값일 뿐이고, 그 값으로 다시 계산한 시간도 대략적인 값이에요.*"]
    lines += ["", "![계산 vs 측정](figures/policies_time_check.png)", ""]
    if len(weak):
        c_lab, c_real = float(weak["c1"].iloc[0]), float(weak["c1_real"].median())
        guess = abs(c_real - c_lab / NOMINAL_AH) < 0.02
        lines.append(
            f"→ **가장 중요한 발견:** b3의 `{weak['policy'].iloc[0]}` 셀 {len(weak)}개는 실제로 **{c_real:.2f}C**로 충전됐어요. "
            + (f"표기 숫자({c_lab:g})를 공칭 용량 숫자({NOMINAL_AH:g})로 나눈 값({c_lab / NOMINAL_AH:.2f})과 같아서, {c_lab:g}C가 아니라 **{c_lab:g}A로 "
               "설정된 것처럼 보여요** (추측이에요. 공식 설명에서 확인하지는 못했어요). " if guess else "")
            + f"그래서 0→80%에 {weak['measured_min'].median():.1f}분이 걸렸고, 이 셀들의 수명 중앙값은 "
            f"{weak.merge(table, on='cell_id')['cycle_life'].median():,.0f}사이클이에요. "
            "아래 분석에서는 표기 대신 **실제로 잰 전류**를 기본으로 써요.")
    lines += [
        "",
        "### 2. 먼저 볼 함정: 배치마다 다른 실험이에요 (교란)",
        "",
        "세 배치는 시험 시기가 다르고, **시험한 충전 조건의 종류도 달라요.** 그래서 배치 차이와 조건 차이가 섞여 있어요(교란, confounding).",
        "",
        "| 배치 | 셀 | 표기 | 측정한 0→80% 시간 | 수명 중앙값 (범위) | 한 사이클 시간 (원시 기록 중앙값) |",
        "|---|---|---|---|---|---|",
    ]
    for scope, r in lv.iterrows():
        if scope == "전체":
            continue
        lines.append(f"| {_batch_name(scope)} | {int(r['cells'])} | {int(r['policies'])}가지 | {r['time_min']:.1f}~{r['time_max']:.1f}분 | "
                     f"{r['life_median']:,.0f} ({r['life_min']:,.0f}~{r['life_max']:,.0f}) | {r['cycle_minutes']:.0f}분 |")
    lines += [
        "",
        "같은 표기가 여러 배치에 있는 경우만 따로 보면:",
        "",
        "| 표기 (`-newstructure` 뗌) | 배치 | 셀 | 수명 중앙값 (범위) | 실제 C1 | 측정한 시간 |",
        "|---|---|---|---|---|---|",
    ]
    for r in across.itertuples():
        lines.append(f"| `{r.base_policy}` | {r.batch} | {r.cells} | {r.life_median:,.0f} ({r.life_min:,.0f}~{r.life_max:,.0f}) | "
                     f"{r.c1_real:.2f}C | {r.minutes:.2f}분 |")
    ratio_lines = []
    for base, g in across.groupby("base_policy"):
        g = g.set_index("batch")
        same = g[abs(g["c1_real"] / g["c1_label"] - 1) <= pol.RATE_TOL]
        if len(same) >= 2:
            hi_b, lo_b = same["life_median"].idxmax(), same["life_median"].idxmin()
            ratio_lines.append(f"`{base}`: 실제 전류까지 같은 {'·'.join(same.index)}끼리 "
                               f"{hi_b} {same.loc[hi_b, 'life_median']:,.0f} vs {lo_b} {same.loc[lo_b, 'life_median']:,.0f}"
                               f"(×{same.loc[hi_b, 'life_median'] / same.loc[lo_b, 'life_median']:.2f})")
    lines += [
        "",
        "→ " + ("; ".join(ratio_lines) + ". " if ratio_lines else "")
        + "같은 표기·같은 실제 전류인데도 배치에 따라 수명이 크게 다르기도, 거의 같기도 해요. "
        "**배치 차이가 일정하지 않아서 \"배치 효과를 빼면 된다\"고 단순하게 말할 수 없어요.** "
        f"(다만 비교할 수 있는 표기가 {across['base_policy'].nunique()}가지뿐이고 셀도 "
        f"{_span(across['cells'].min(), across['cells'].max(), '.0f')}개씩이라 근거는 적어요.)",
        "",
        "### 3. 충전 시간과 수명",
        "",
        "| 범위 | 셀 | 측정한 시간 범위 | 시간과 log 수명의 순위 상관 ρ |",
        "|---|---|---|---|",
    ]
    for scope, r in lv.iterrows():
        lines.append(f"| {scope if scope == '전체' else _batch_name(scope)} | {int(r['cells'])} | {r['time_min']:.1f}~{r['time_max']:.1f}분 | {r['rho_time']:+.2f} |")
    b1_rho = lv.loc["b1", "rho_time"] if "b1" in lv.index else np.nan
    rho_all_text = f"{lv.loc['전체', 'rho_time']:+.2f}"
    lines += [
        "",
        f"→ 전체로 보면 ρ = {euro(rho_all_text)} **관계가 거의 안 보여요.** 그런데 충전 시간을 실제로 바꿔 가며 시험한 "
        f"b1 안에서는 ρ = {euro(f'**{b1_rho:+.2f}**')}, 충전이 느린 조건일수록 수명이 긴 경향이 "
        f"{'뚜렷해요' if abs(b1_rho) >= 0.5 else '보여요'}. "
        f"b2·b3는 셀 {fixed['share']:.0%}가 측정 시간 {fixed['lo']:.1f}~{fixed['hi']:.1f}분 안에 몰려 있어 시간 차이가 작고, "
        "대신 배치끼리 수명 수준이 달라서 합치면 경향이 묻혀요.",
        "",
        "### 4. 낮은 SOC 전류 vs 높은 SOC 전류",
        "",
        "![충전 조건 지도](figures/policies_map.png)",
        "",
        f"b2·b3처럼 0→80% 시간이 거의 고정이면(셀 {fixed['share']:.0%}가 {fixed['lo']:.1f}~{fixed['hi']:.1f}분) "
        "한쪽을 세게 할수록 다른 쪽은 약해져요(지도의 점선 위를 움직임). "
        "그래서 b2·b3는 **뒤쪽 쏠림** 하나로, 시간이 함께 바뀐 b1과 전체는 두 전류를 함께 넣은 모델로 봤어요.",
        "",
        "| 모델 | 셀 (조건) | 항목 | +1C마다 수명 변화 | 90% 범위 | 판단 | 표기 전류로 다시 하면 | R² |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in res["effects"]:
        if e["source"] != "실측":
            continue
        nominal = next(x for x in res["effects"] if x["name"] == e["name"] and x["source"] == "표기")
        for (term, v), nv in zip(e["terms"].items(), nominal["terms"].values()):
            lines.append(f"| {e['name']} | {e['cells']} ({e['groups']}) | {SEPARATE[term][0]} | **{v['pct']:+.1f}%** | {_rng(v)} | "
                         f"{_verdict(v)} | {nv['pct']:+.1f}% ({_rng(nv)}) | {e['r2']:.2f} |")
    pooled, b1 = _find(res["effects"], "전체 (배치 보정)"), _find(res["effects"], "b1")
    b2 = next((e for e in res["effects"] if e["name"].startswith("b2") and e["source"] == "실측"), None)
    lines += [
        "",
        "*R²: 수명(log) 차이 중 이 모델이 설명하는 몫 (0~1). 전체 모델의 R²에는 배치마다 다른 출발점이 설명한 몫도 들어 있어요.*",
        "",
        "![효과 크기](figures/policies_effects.png)",
        "",
        "두 전류를 함께 넣은 모델에서 \"높은 SOC 전류의 기울기가 낮은 SOC 전류보다 더 나쁜(더 음수인)\" 경우의 비율 "
        "(부트스트랩 다시 뽑기 중):",
        "",
        "| 모델 | 높은 SOC 쪽이 더 나쁘게 나온 비율 | 높은 SOC +1C가 낮은 SOC +1C보다 수명을 더 바꾸는 정도 | 90% 범위 |",
        "|---|---|---|---|",
    ]
    for e in (b1, pooled):
        lines.append(f"| {e['name']} | {e['second_worse_share'] * 100:.0f}% | {e['gap_pct']:+.1f}% | "
                     f"{e['gap_pct_lo']:+.1f} ~ {e['gap_pct_hi']:+.1f}% |")
    b2_cells = table[table["batch"] == "b2"].nsmallest(2, "cycle_life") if "b2" in set(table["batch"]) else table.iloc[:0]
    lines += [""]
    out = res.get("leave_out") or {}
    if b2 is not None and len(b2_cells):
        rho_b2_text = f"{lv.loc['b2', 'rho_shift']:+.2f}"
        text = (f"b2 안의 순위 상관은 뒤쪽 쏠림과 log 수명 사이 ρ = {ieyo(rho_b2_text)}. 직선 모델의 기울기"
                f"({b2['terms']['shift']['pct']:+.1f}%)는 b2에서 수명이 가장 짧은 두 셀("
                + ", ".join(f"`{r.policy}` {r.cycle_life:,.0f}사이클, 뒤쪽 쏠림 {r.shift:+.1f}C" for r in b2_cells.itertuples())
                + ")에 크게 기대요.")
        if out:
            w, p = out["within"]["terms"]["shift"], out["pooled"]
            text += (f" 이 두 셀만 빼고 다시 맞추면(민감도 점검) b2 기울기는 {w['pct']:+.1f}% ({_rng(w)}), "
                     f"전체(배치 보정) 모델의 '높은 SOC +1C가 낮은 SOC +1C보다 수명을 더 바꾸는 정도'는 "
                     f"{p['gap_pct']:+.1f}% ({p['gap_pct_lo']:+.1f} ~ {p['gap_pct_hi']:+.1f}%)"
                     + ("이고, 둘 다 범위가 0을 포함해요." if not w["clear"] and _gap_has_zero(p) else "예요."))
        lines.append(text)
    rep_all = rep["all"]
    lines += [
        "",
        "### 5. 같은 조건 반복 셀: 조건 안 차이 vs 조건 사이 차이",
        "",
        "| 범위 | 셀 (조건) | η² (조건으로 설명되는 몫) | 조건 이름을 섞었을 때 η² 평균 (섞은 결과의 95%가 이 값 이하) | 조건 안 흔들림 | 조건 평균끼리의 흔들림 |",
        "|---|---|---|---|---|---|",
        f"| 전체 (배치·표기 조건) | {rep_all['cells']} ({rep_all['groups']}) | **{rep_all['eta2']:.2f}** | {rep_all['null_mean']:.2f} "
        f"({rep_all['null_p95']:.2f}) | {_pm(rep_all['within_sd'])} | {_pm(rep_all['between_sd'])} |",
    ]
    for b, v in rep["by_batch"].items():
        lines.append(f"| {_batch_name(b)} 안 | {v['cells']} ({v['groups']}) | **{v['eta2']:.2f}** | {v['null_mean']:.2f} ({v['null_p95']:.2f}) | "
                     f"{_pm(v['within_sd'])} | {_pm(v['between_sd'])} |")
    rg = rep["groups"]
    widest = rg.loc[rg["ratio"].idxmax()]
    lines += [
        "",
        f"*흔들림 = log10 수명의 표준편차를 %로 바꾼 값. 같은 셀 수에서 배치만 알 때 η²는 {rep['batch_only']:.2f}예요."
        + "".join(f" {b}는 반복 조건이 {int((rg['batch'] == b).sum())}개(셀 {int(rg.loc[rg['batch'] == b, 'cells'].sum())}개)뿐이라 "
                  "배치 안 η²는 계산하지 않고 전체에만 넣었어요."
                  for b in sorted(set(rg["batch"]) - set(rep["by_batch"])))
        + "*",
        "",
        "![반복 셀](figures/policies_replicates.png)",
        "",
        f"→ 같은 조건 셀의 최대÷최소는 중앙값 ×{rg['ratio'].median():.2f}, 가장 큰 조건은 `{widest['group']}`"
        f"(셀 {widest['cells']}개, {widest['life_min']:,.0f}~{widest['life_max']:,.0f}사이클, ×{widest['ratio']:.2f})예요.",
        "",
        "### 6. 빠르면서 오래 간 조건 (관찰된 최고, 최적화한 결과가 아님)",
        "",
        "| 측정한 충전 시간 | 조건 수 (셀) | 들어 있는 배치 | 수명 중앙값이 가장 긴 조건 | 그 조건의 수명 중앙값 (셀 수) | 구간 안 셀 하나의 최고 | 이 시간까지 본 최고 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in front.itertuples():
        if r.groups == 0:
            lines.append(f"| {r.lo:g}~{r.hi:g}분 | 0 | - | - | - | - | - |")
            continue
        lines.append(f"| {r.lo:g}~{r.hi:g}분 | {r.groups} ({r.cells}) | {r.batches} | `{r.best_group}` | "
                     f"{r.best_life:,.0f} ({r.best_cells}개) | {r.best_cell_life:,.0f} | {r.so_far:,.0f} |")
    lines += [
        "",
        "*이 시간까지 본 최고 = 이 구간과 더 빠른 구간들에서 본 '조건 수명 중앙값' 가운데 가장 긴 값이에요(셀 하나의 값이 아니에요).*",
        "",
        "![관찰된 최고](figures/policies_frontier.png)",
        "",
        "## 읽는 법 (숫자는 코드가 계산)",
        "",
    ]
    lines += _reading(table, check, lv, pooled, b1, b2, rep, front, res["groups"], weak, out)
    lines += [
        "",
        "## 한계",
        "",
        "- **관찰 데이터예요.** 충전 조건을 무작위로 셀에 배정한 실험이 아니라, 배치(시험 시기)마다 다른 조건 묶음을 시험했어요. "
        "그래서 \"이 조건 때문에 수명이 짧다\"가 아니라 \"이 조건과 짧은 수명이 함께 나타났다\"까지만 말할 수 있어요.",
        "- **`-newstructure`와 b3는 떼어 볼 수 없어요.** 이 표시가 붙은 셀은 모두 b3라서, 표시가 뜻하는 차이와 배치 차이를 구분할 수 없어요. "
        "원시 기록에서 b3는 한 사이클 시간이 더 짧다는 것까지는 확인했지만, `newstructure`의 정확한 뜻은 확인하지 못했어요.",
        "- **조건마다 셀이 적어요.** b2는 거의 조건마다 셀 1개라, 조건 안 흔들림보다 작은 차이는 우연과 구별할 수 없어요.",
        f"- **실제 전류는 원시 기록이 남은 일부 사이클(셀마다 {_span(table['raw_cycles'].min(), table['raw_cycles'].max(), '.0f')}개)로만 쟀어요.** "
        "셀이 늙으면서 3.6V 상한에 더 일찍 닿는 변화는 반영하지 않았어요.",
        "- **직선 모델은 단순한 가정이에요.** \"+1C마다 몇 %\"는 시험한 전류 범위 안에서의 평균적인 기울기일 뿐이고, "
        "그 밖으로 늘려 쓰면 안 돼요.",
        "- **원인(리튬 석출 등)은 이 데이터로 직접 확인할 수 없어요.** 수명 숫자만 봤기 때문에, 석출 이야기는 가설로만 남겨요. "
        "급속충전 때의 발열 차이도 교란일 수 있지만, 온도 센서(캔 표면 열전대)는 공식적으로 믿기 어려워서 쓰지 않았어요.",
        *_stitched_note(table, front, lv, stitched),
        "",
        "## 시행착오",
        "",
        "이번 작업에서 실제로 틀렸다가 고친 것들이에요. 형식: 상황 → 처음 생각 → 어떻게 알아챘나 → 원인 → 수정 → 재발 방지.",
        "",
    ]
    lines += _mistakes(weak, check, b2, pooled, out)
    lines += [
        "",
        f"데이터 출처: {PROJECT_URL} ({LICENSE}). 더 해 볼 만한 응용은 [응용 아이디어](ideas.md)에 있어요.",
        "",
    ]
    return "\n".join(lines)


def _reading(table, check, lv, pooled, b1, b2, rep, front, groups, weak, out=None) -> list[str]:
    lo, hi = pooled["terms"]["low_rate"], pooled["terms"]["high_rate"]
    b1lo, b1hi = b1["terms"]["low_rate"], b1["terms"]["high_rate"]
    gap_clear = pooled["gap_pct_lo"] > 0 or pooled["gap_pct_hi"] < 0
    flagged = int(check["flagged"].sum())
    first = (f"1. **표기를 믿기 전에 실제를 재요.** {len(check)}개 중 {flagged}개 셀은 표기와 실제 충전이 달랐어요.")
    if len(weak):
        first += (f" 특히 b3의 `{weak['policy'].iloc[0]}` 셀은 표기보다 약한 전류로 충전됐는데, 표기만 보면 "
                  "\"이 전류로도 오래 간다\"는 잘못된 결론을 낼 수 있었어요.")
    rho_all, rho_b1 = lv.loc["전체", "rho_time"], lv.loc["b1", "rho_time"]
    second = (f"2. **충전이 느린 조건일수록 수명이 긴 경향은 충전 시간을 실제로 바꿔 시험한 b1 안에서 보여요** (ρ = {rho_b1:+.2f}). "
              f"전체로 합치면 ρ = {rho_all:+.2f}{'로 사라져요' if abs(rho_all) < 0.2 else '예요'}. "
              "배치를 섞어 보면 배치 안의 경향이 숨을 수 있어요.")
    both = lo["pct"] < 0 and hi["pct"] < 0 and lo["clear"] and hi["clear"]
    third = (f"3. **{'두 구간 모두 센 전류와 짧은 수명이 함께 나타나요' if both else '두 구간 전류와 수명의 관계'}.** "
             f"전체(배치 보정) 모델에서 +1C마다 낮은 SOC {lo['pct']:+.0f}% ({_rng(lo)}), 높은 SOC {hi['pct']:+.0f}% ({_rng(hi)}). "
             f"b1만 보면 낮은 SOC {b1lo['pct']:+.0f}% ({_rng(b1lo)}), 높은 SOC {b1hi['pct']:+.0f}% ({_rng(b1hi)})예요.")
    share, b1_share = pooled["second_worse_share"], b1["second_worse_share"]
    pooled_says = gap_clear and pooled["gap_pct"] < 0                     # 합친 모델: 높은 SOC 쪽이 뚜렷이 더 나쁨
    b1_agrees = b1["gap_pct"] < 0 and (b1["gap_pct_hi"] < 0)              # b1 안에서도 같은 방향이 뚜렷한가
    rho_b2 = lv.loc["b2", "rho_shift"] if (b2 is not None and "b2" in lv.index) else np.nan
    b2_agrees = np.isfinite(rho_b2) and rho_b2 <= -0.2                     # b2 순위 상관도 같은 방향인가
    if pooled_says and b1_agrees and b2_agrees:
        head = "이 데이터와 맞아요 (관찰 수준에서)"
    elif pooled_says:
        head = "배치를 합친 모델에서는 보이지만, 배치 안에서는 일관되지 않아요"
    else:
        head = "이 데이터로 확인되지 않아요"
    fourth = (f"4. **\"높은 SOC 전류가 더 해롭다\"는 가설: {head}.** "
              f"전체(배치 보정) 모델에서 높은 SOC 쪽이 더 나쁘게 나온 비율은 {share:.0%}, 높은 SOC +1C가 낮은 SOC +1C보다 "
              f"수명을 {pooled['gap_pct']:+.1f}% 더 바꾸고 90% 범위는 {pooled['gap_pct_lo']:+.1f} ~ {pooled['gap_pct_hi']:+.1f}%"
              + ("로 0을 포함하지 않아요." if gap_clear else "로 0을 포함해요.")
              + f" 그런데 b1만 보면 그 비율이 {b1_share:.0%}(차이 {b1['gap_pct']:+.1f}%, 범위 {b1['gap_pct_lo']:+.1f} ~ "
              f"{b1['gap_pct_hi']:+.1f}%)"
              + (("로 오히려 반대 방향이에요" + (" (범위가 0을 포함해 확실하지는 않아요)." if b1["gap_pct_lo"] < 0
                                                  else ".")) if b1["gap_pct"] > 0 else "예요."))
    out = out or {}
    if b2 is not None and np.isfinite(rho_b2):
        fourth += (f" 시간이 거의 고정된 b2에서는 직선 기울기가 뒤쪽 쏠림 +1C마다 {b2['terms']['shift']['pct']:+.1f}%이지만 "
                   f"순위 상관은 ρ = {rho_b2:+.2f}"
                   + ("로 거의 0이라, 수명이 아주 짧은 몇 개 셀이 만든 기울기로 보여요" if abs(rho_b2) < 0.2 else "예요"))
        if out and abs(rho_b2) < 0.2:
            fourth += f" (가장 짧은 {len(out['cells'])}셀을 빼면 {out['within']['terms']['shift']['pct']:+.1f}%)"
        fourth += "."
    leans_on_few = bool(out) and _gap_has_zero(out["pooled"])
    if pooled_says and not (b1_agrees and b2_agrees):
        if leans_on_few:
            fourth += (f" 합친 모델은 배치마다 다른 경향을 기울기 하나로 묶은 결과이고, b2의 {len(out['cells'])}셀을 빼면 차이가 "
                       f"{out['pooled']['gap_pct']:+.1f}% ({out['pooled']['gap_pct_lo']:+.1f} ~ "
                       f"{out['pooled']['gap_pct_hi']:+.1f}%)로 범위가 0을 포함해요. 그래서")
        else:
            fourth += " 합친 모델은 배치마다 다른 경향을 기울기 하나로 묶은 결과예요."
        fourth += " 가설 쪽으로 기우는 관찰은 있지만 확인됐다고 말하기는 일러요."
    elif not pooled_says:
        fourth += " 가설을 반박하지도, 확인하지도 못하는 상태예요."
    fifth = (f"5. **조건이 수명 차이를 크게 설명하지만, 셀 간 편차도 작지 않아요.** 반복 셀에서 η² = {rep['all']['eta2']:.2f} "
             f"(이름을 섞으면 평균 {rep['all']['null_mean']:.2f})예요. 조건 안 흔들림은 "
             + ", ".join(f"{b} {_pm(v['within_sd'])}" for b, v in rep["by_batch"].items())
             + "라서, 셀 1개짜리 조건끼리 그 정도 차이는 우연일 수 있어요.")
    out = [first, second, third, fourth, fifth]
    f = front[(front["groups"] > 0) & (front["best_batch"] == "b3")] if "best_batch" in front else front.iloc[:0]
    if len(f):
        parts = []
        for r in f.itertuples():
            inside = groups[(groups["minutes"] >= r.lo) & (groups["minutes"] < r.hi) & (groups["batch"] != "b3")]
            other = (f"b1·b2 최고 `{inside.loc[inside['life_median'].idxmax(), 'group']}` "
                     f"{inside['life_median'].max():,.0f}" if len(inside) else "b1·b2 조건 없음")
            parts.append(f"{r.lo:g}~{r.hi:g}분: b3 `{_short(r.best_policy)}` {r.best_life:,.0f} vs {other}")
        out.append(
            "6. **관찰된 최고 중 일부 시간 구간은 b3 조건이 차지해요.** " + " / ".join(parts) + ". "
            "조건이 좋아서인지 b3에서 시험해서인지는 구분할 수 없어요. "
            "새 충전 조건을 고른다면 이 목록은 \"다시 시험해 볼 후보\"이지 정답이 아니에요.")
    return out


def _mistakes(weak: pd.DataFrame, check: pd.DataFrame, b2: dict | None = None, pooled: dict | None = None,
              leave_out: dict | None = None) -> list[str]:
    """이번 작업에서 실제로 있었던 일만, 일어난 순서대로 적는다.

    b2·pooled·leave_out(가장 짧은 b2 셀을 뺀 민감도 점검)은 마지막 항목(검증 단계에서 찾은 것)의 숫자에 쓴다.
    """
    # 80% 판정 기준 아래 남는 용량을 마지막 단계 표기 전류로 채우는 시간(초): (남은 %/100) × 3600초 ÷ C-rate
    gap_soc = pol.TARGET_SOC - pol.END_SOC
    last_rate = check["c2"].where(check["q1"] < pol.TARGET_SOC, check["c1"])
    last_secs = gap_soc / 100 * 3600 / last_rate
    items = []
    if len(weak):
        items.append((
            "조건 표기 = 실제 전류라고 짐작함",
            "처음 계획은 표기(C1, C2)만으로 \"낮은 SOC vs 높은 SOC 전류\"를 비교하는 것이었어요.",
            "표기대로 충전됐을 것이다.",
            f"계산한 시간과 측정한 시간을 맞춰 보니 b3 `{weak['policy'].iloc[0]}` 셀 {len(weak)}개만 약 "
            f"{weak['diff_min'].median():.1f}분 길었어요. 같은 표기의 b1·b2 셀은 차이가 없었어요.",
            f"원시 전류를 재 보니 실제로는 {weak['c1_real'].median():.2f}C였어요. 그 밖에도 높은 SOC에서 3.6V 상한에 닿아 "
            "전류가 줄어든 셀과, 표기 끝의 C가 빠진 셀에서 실제 전류가 표기와 달랐어요.",
            "원시 기록으로 실제 평균 전류를 재서 분석의 기본으로 쓰고, 표기 전류로 한 결과는 비교용으로 함께 실었어요.",
            "조건 표기를 변수로 쓰기 전에, 표본 하나가 아니라 전체 셀의 원시 데이터로 확인해요."))
    items.append((
        "80% 뒤 쉬는 시간 재기가 거의 0분과 몇 분으로 갈림 (탐색 단계, 결과에는 안 씀)",
        "배치끼리 시험 순서가 어떻게 다른지 보려고, SOC 80%에 닿은 뒤 다음 충전이 시작될 때까지 쉬는 시간을 재 봤어요.",
        "SOC가 80% 바로 아래 문턱(79.95%)을 넘은 첫 점 = 급속충전이 끝난 점.",
        "같은 셀의 사이클끼리 값이 거의 0분과 몇 분으로 갈렸어요.",
        "문턱을 넘은 첫 점이 아직 급속충전 중일 때가 있어서, 바로 다음 점(여전히 센 전류)을 \"다시 충전 시작\"으로 셌어요.",
        "이 측정은 버리고, 배치 비교에는 한 사이클 전체 시간만 썼어요.",
        "문턱값으로 사건을 찾을 때는 문턱 바로 앞뒤 점의 전류를 같이 출력해 확인해요."))
    items.append((
        "부트스트랩 범위가 비어 나옴 (코드 버그)",
        "전체(배치 보정) 모델의 90% 범위가 숫자 없이 비어 나왔어요.",
        "배치별로 조건 덩어리를 나눠 다시 뽑으면 된다.",
        f"다시 뽑기 {pol.BOOT:,}번 중 실제로 쓰인 것이 한 번도 없었어요. 배치 하나만 쓴 모델은 정상이었어요.",
        "배치별로 잘라 낸 작은 표에서 줄 번호를 가져와서, b2·b3 셀 자리에 b1 셀이 뽑혔어요. 그 결과 배치 표시 열이 모두 0이 돼 "
        "기울기를 계산할 수 없었어요.",
        "줄 번호를 전체 표 기준으로 가져오도록 고쳤고, 여러 배치를 섞어도 범위가 나오는지 확인하는 테스트를 넣었어요.",
        "다시 뽑기 결과에 \"실제로 쓰인 횟수\"를 함께 남기고, 0이면 바로 의심해요."))
    items.append((
        "SOC 80% 도달을 '80.00% 이상'으로 잡음",
        "원시 기록에서 SOC 40%·80%에 닿은 시각으로 구간 평균 전류를 계산했어요.",
        "급속충전은 80%를 넘긴 순간 끝나니, 80.00% 이상인 첫 점을 찾으면 된다.",
        "충전 조건 지도에서 b2 점 하나(b2c5, `3.6C(2%)-4.85C`)가 같은 시간 곡선에서 혼자 멀리 떨어져 있었어요. "
        "높은 SOC 평균 전류가 표기보다 한참 낮게 나왔는데, 그 셀의 summary 충전 시간은 정상이었어요.",
        "그 셀의 한 사이클은 급속충전이 80%를 아주 조금 넘긴 지점에서 끝났어요. 그런데 SOC를 '충전 첫 점 이후 넣은 용량'으로 "
        "계산하면서 첫 점에 이미 쌓여 있던 아주 작은 용량을 빼는 바람에 끝이 80%에 살짝 못 미쳤고, 80%에는 쉬는 시간이 지나 "
        "1C 충전이 시작돼서야 닿았어요. 쉬는 시간이 급속충전 시간에 들어간 거예요.",
        f"'닿음'의 기준을 {pol.END_SOC:g}%로 바꾸고(나머지 {gap_soc:.1f}%p는 센 전류로 1초 안팎, "
        f"셀마다 {_span(last_secs.min(), last_secs.max(), '.1f')}초), "
        f"원시 기록으로 잰 시간과 summary 충전 시간이 {pol.RAW_TOL:g}분 넘게 다른 셀을 세는 점검을 넣었어요"
        f" (고친 뒤 {int(check['raw_disagree'].sum())}개).",
        "그래프에서 혼자 튀는 점은 데이터보다 계산 코드를 먼저 의심하고, 같은 값을 두 출처(원시 기록·summary)로 맞춰 봐요."))
    if b2 is not None and pooled is not None and leave_out:
        w, p = leave_out["within"]["terms"]["shift"], leave_out["pooled"]
        held = not w["clear"] and _gap_has_zero(p)
        items.append((
            "근거 숫자 없이 쓴 주장 (검증 단계에서 발견)",
            "구현을 마친 뒤 다른 에이전트가 문서의 숫자를 모듈 함수 없이 따로 짠 스크립트로 다시 계산하고, 문장과 숫자를 하나씩 대조했어요.",
            f"b2의 직선 기울기(뒤쪽 쏠림 +1C마다 {b2['terms']['shift']['pct']:+.1f}%)는 수명이 가장 짧은 셀 몇 개에 크게 기대고, "
            "합친 모델의 신호도 몇 개 셀에 기댄다. (숫자 없이 문서에 이렇게 단정해 적었어요.)",
            "다시 계산한 숫자는 모두 문서와 맞았어요. 하지만 위 주장을 보여 주는 숫자가 문서 어디에도 없었어요. "
            "또 읽는 법 4번은 그 원인을 '배치 사이 차이(교란)'라고 썼는데, 합친 모델에는 배치 보정(배치마다 다른 출발점)이 이미 들어 있었어요.",
            "주장을 확인하는 계산(그 셀들을 빼고 다시 맞추기)을 하지 않고 문장만 썼어요.",
            f"b2에서 수명이 가장 짧은 {len(leave_out['cells'])}셀({', '.join(leave_out['cells'])})을 빼고 다시 맞추는 민감도 점검을 "
            f"코드에 넣고 결과 4절에 실었어요. 빼면 b2 기울기는 {b2['terms']['shift']['pct']:+.1f}% → {w['pct']:+.1f}% ({_rng(w)}), "
            f"합친 모델의 '높은 SOC +1C가 낮은 SOC +1C보다 수명을 더 바꾸는 정도'는 {pooled['gap_pct']:+.1f}% → {p['gap_pct']:+.1f}% "
            f"({p['gap_pct_lo']:+.1f} ~ {p['gap_pct_hi']:+.1f}%)"
            + ("로 둘 다 범위가 0을 포함해요. 주장 자체는 맞았지만 근거가 빠져 있었던 거예요. " if held else "예요. ")
            + "읽는 법 4번의 원인은 '배치마다 다른 경향을 기울기 하나로 묶은 결과'로 고치고, 단정한 문장은 '~로 보여요'로 낮췄어요. "
            "같은 검증에서 계산식 문장의 단위 실수(÷100이 빠짐, 코드는 맞았음), 근거보다 센 표현(리튬 석출을 단정함 등), "
            "설명이 빠진 표 칸과 각주도 함께 고쳤어요.",
            "\"몇 개 셀에 기댄다\" 같은 주장에는 그걸 보여 주는 숫자(빼고 다시 맞춘 결과 등)를 코드가 계산해 바로 옆에 붙여요. "
            "문서를 만든 사람이 아닌 다른 눈으로 문장과 숫자를 다시 대조해요."))
    out = []
    for n, (title, situation, first, noticed, cause, fix, prevent) in enumerate(items, start=1):
        out += [f"{n}. **{title}**", f"   - 상황: {situation}", f"   - 처음 생각: {first}",
                f"   - 어떻게 알아챘나: {noticed}", f"   - 원인: {cause}", f"   - 수정: {fix}", f"   - 재발 방지: {prevent}"]
    return out


if __name__ == "__main__":
    build()
