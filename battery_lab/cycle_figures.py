"""사이클 수 실험 그래프. 색·글꼴 규칙은 figures.py와 같다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from .figures import AXIS, BLUES, INK, INK_2, MODEL_COLORS, SURFACE, _life_colorbar, _save, _style, _subtitle


def _days(k: int, minutes: float) -> float:
    return k * minutes / 60 / 24


def signal(corr: pd.DataFrame, minutes: float, path: Path) -> Path:
    """사이클 k에 따른 상관 r. 실선 = 곡선 한 개(논문 방식), 점선 = 5사이클 중앙값 곡선."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    styles = {1: ("곡선 한 개 (논문 방식)", "-", "#2a78d6"), 5: ("5사이클 중앙값 곡선", (0, (4, 3)), "#86b6ef")}
    for w, (label, ls, color) in styles.items():
        s = corr[corr["window"] == w].sort_values("late")
        ax.plot(s["late"], s["r"], ls=ls, color=color, lw=1.8, marker="o", ms=4, label=label)
    ax.axhline(-0.8, color=AXIS, lw=1)
    ax.text(21, -0.79, "r = −0.8", color=INK_2, fontsize=8.5, va="bottom")
    ax.invert_yaxis()  # 위로 갈수록 강한 신호
    ax.set(xlabel="k (몇 번째 사이클까지 봤나)", ylabel="log Var(ΔQ_k−10)와 log 수명의 상관 r")
    ax.set_xticks(range(20, 101, 10), labels=[f"{k}\n({_days(k, minutes):.1f}일)" for k in range(20, 101, 10)])
    ax.set_title("신호는 40사이클쯤이면 거의 다 올라와요")
    _subtitle(ax, f"위로 갈수록 강한 신호 · 괄호 = 사이클당 {minutes:.0f}분(측정 중앙값)으로 환산한 시험 기간")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    return _save(fig, path)


def error(summary: pd.DataFrame, minutes: float, path: Path) -> Path:
    """무작위 나누기 100번: k에 따른 오차 중앙값과 10~90% 범위."""
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for kind in ("simple", "full"):
        label, color = MODEL_COLORS[kind]
        s = summary[summary["model"] == kind].sort_values("late")
        ax.fill_between(s["late"], s["p10"], s["p90"], color=color, alpha=0.15, lw=0)
        ax.plot(s["late"], s["median"], color=color, lw=2, marker="o", ms=5, label=label)
    ref = float(summary[(summary["model"] == "simple") & (summary["late"] == 100)]["median"].iloc[0])
    ax.axhline(ref, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(21, ref - 0.4, f"단순 모델 · 100사이클 = {ref:.1f}%", ha="left", va="top", color=INK_2, fontsize=9)
    ax.set_xticks(range(20, 101, 10), labels=[f"{k}\n({_days(k, minutes):.1f}일)" for k in range(20, 101, 10)])
    ax.set(xlabel="k (몇 번째 사이클까지 봤나)", ylabel="평균 % 오차 (MAPE)")
    ax.set_title("사이클을 더 볼수록 정확해지지만, 곧 완만해져요")
    _subtitle(ax, "124개를 41/83으로 무작위로 100번 나눔 · 선 = 중앙값 · 띠 = 10~90% 범위")
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_2)
    return _save(fig, path)


def delta_q_panels(curves_by_k: dict[int, list[tuple[np.ndarray, np.ndarray, float]]], path: Path) -> Path:
    """k = 20·50·100일 때의 ΔQ_k−10(V). 색 = 수명 (진할수록 오래 버틴 셀)."""
    _style()
    lives = [life for rows in curves_by_k.values() for _, _, life in rows]
    norm = LogNorm(vmin=min(lives), vmax=max(lives))
    fig, axes = plt.subplots(1, len(curves_by_k), figsize=(11, 3.8), sharey=True)
    for ax, (k, rows) in zip(axes, curves_by_k.items()):
        for v, d, life in sorted(rows, key=lambda x: -x[2]):
            ax.plot(v, d * 1000, lw=0.7, color=BLUES(norm(life)), alpha=0.95)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set(xlim=(3.5, 2.0), xlabel="전압 (V)")
        ax.set_title(f"k = {k}", fontsize=11, pad=8)
    axes[0].set_ylabel("ΔQ_k−10 (mAh)")
    fig.suptitle("ΔQ의 벌어짐은 사이클이 지날수록 커져요", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.9, "k번째와 10번째 방전 곡선의 차이 · 진할수록 오래 버틴 셀", color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.93, 0.89))
    _life_colorbar(fig, axes, norm)
    return _save(fig, path)


def grid(table: pd.DataFrame, minutes: float, path: Path) -> Path:
    """셀 수 N × 사이클 수 k: 왼쪽 오차 중앙값, 오른쪽 오차 25% 초과 비율."""
    _style()
    ns = sorted(table["n"].unique(), reverse=True)
    ks = sorted(table["late"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    panels = (("median", "실제 오차 중앙값 (%)", "{:.1f}", 10, 45),
              ("over25", "실제 오차 25% 초과 비율", "{:.0%}", 0, 1))
    for ax, (col, title, fmt, lo, hi) in zip(axes, panels):
        m = table.pivot(index="n", columns="late", values=col).loc[ns, ks].to_numpy()
        ax.imshow(np.clip((m - lo) / (hi - lo), 0, 1), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for i in range(len(ns)):
            for j in range(len(ks)):
                shade = (m[i, j] - lo) / (hi - lo)
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=8.5,
                        color=SURFACE if shade > 0.55 else INK)
        ax.set_xticks(range(len(ks)), labels=[str(k) for k in ks])
        ax.set_yticks(range(len(ns)), labels=[f"{n}개" for n in ns])
        ax.set_xlabel("k (몇 번째 사이클까지 봤나)")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=11, pad=8)
    axes[0].set_ylabel("가지고 있던 셀 수")
    fig.suptitle("셀 몇 개 × 몇 사이클이면 될까 (단순 모델)", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.9, "N개를 300번 무작위로 뽑아 1/3로 학습 · 오차는 학습에 안 쓴 나머지 모든 셀로 잼 · 진할수록 나쁨",
             color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, path)
