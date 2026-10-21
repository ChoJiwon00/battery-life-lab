"""README·리포트용 그래프.

색 규칙
- 수명(크기)은 파랑 한 가지 색을 연한 것 → 진한 것으로 (오래 버틸수록 진하게)
- 학습/테스트 구분(종류)은 색맹에도 구분되도록 검증된 세 색(파랑·주황·청록)을 고정 순서로
- 글자는 잉크색, 격자는 아주 옅게
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 그린다
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from .quality import EOL_AH, QD_RANGE, CellReport, capacity_events, delta_q, load_curves  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WARN = "#fab219"      # 상태색: 주의
SERIOUS = "#ec835a"   # 상태색: 심각
KOREAN_FONTS = ("Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR")
LIFE_TICKS = (150, 200, 300, 500, 1000, 2000)
# 파랑 순차 램프 250 → 700 (밝은 배경에서 가장 연한 선도 보이도록 250부터)
BLUES = LinearSegmentedColormap.from_list(
    "life", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
SPLITS = {"train": ("학습", "#2a78d6"), "test": ("테스트", "#eb6834"), "secondary_test": ("2차 테스트", "#1baf7a")}


def _korean_font() -> list[str]:
    """설치된 한글 글꼴 하나 + 아래첨자(₁₀₀) 같은 글자를 채울 matplotlib 기본 글꼴.
    없는 글꼴까지 목록에 넣으면 찾느라 경고가 쏟아진다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in KOREAN_FONTS if f in installed][:1]
    return [*korean, "DejaVu Sans"]


def _life_colorbar(fig, ax, norm) -> None:
    sm = plt.cm.ScalarMappable(norm=norm, cmap=BLUES)
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.035)
    ticks = [t for t in LIFE_TICKS if norm.vmin <= t <= norm.vmax]
    cb.set_ticks(ticks, labels=[f"{t:,}" for t in ticks])
    cb.ax.minorticks_off()
    cb.set_label("사이클 수명", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)


def _style() -> None:
    plt.rcParams.update({
        "font.family": _korean_font(),
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 22,
    })


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _life_norm(reports: list[CellReport]) -> LogNorm:
    life = np.array([r.cycle_life for r in reports if np.isfinite(r.cycle_life)])
    return LogNorm(vmin=life.min(), vmax=life.max())


def capacity_fade(summary: pd.DataFrame, reports: list[CellReport], path: Path) -> Path:
    _style()
    norm = _life_norm(reports)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for r in sorted(reports, key=lambda r: -r.cycle_life):  # 짧은 수명(연한 선)을 위에
        s = summary[summary["cell_id"] == r.cell_id]
        qd = s["QD"].where((s["QD"] > QD_RANGE[0]) & (s["QD"] < QD_RANGE[1]))
        ax.plot(s["cycle"], qd, lw=0.8, color=BLUES(norm(r.cycle_life)), alpha=0.95)
    ax.axhline(EOL_AH, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(2280, EOL_AH + 0.004, "수명 끝 0.88Ah (공칭의 80%)", ha="right", va="bottom", color=INK_2, fontsize=9,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2, "alpha": 0.9}, zorder=5)
    ax.set(xlim=(0, 2300), ylim=(0.86, 1.10), xlabel="사이클", ylabel="방전 용량 (Ah)")
    ax.set_title(f"{len(reports)}개 셀의 방전 용량 변화")
    ratio = norm.vmax / norm.vmin
    _subtitle(ax, f"같은 모델의 셀, 같은 30°C인데 충전 조건에 따라 수명이 약 {ratio:.0f}배까지 달라요 · 진할수록 오래 버틴 셀")
    _life_colorbar(fig, ax, norm)
    return _save(fig, path)


def delta_q_curves(folder: Path, reports: list[CellReport], path: Path) -> Path:
    _style()
    norm = _life_norm(reports)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for r in sorted(reports, key=lambda r: -r.cycle_life):
        curves = load_curves(folder, r.cell_id)
        d = delta_q(curves)
        if d is None:
            continue
        v = curves.get("Vdlin", np.linspace(3.5, 2.0, len(d)))
        ax.plot(v, d * 1000, lw=0.8, color=BLUES(norm(r.cycle_life)), alpha=0.95)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set(xlim=(2.0, 3.5), xlabel="전압 (V)", ylabel="ΔQ₁₀₀₋₁₀ (mAh)")
    ax.invert_xaxis()  # 방전은 높은 전압 → 낮은 전압
    ax.set_title("100번째와 10번째 방전 곡선의 차이 ΔQ₁₀₀₋₁₀(V)")
    _subtitle(ax, "용량은 거의 그대로인데 곡선 모양은 셀마다 다르게 변해요 · 연한 선(짧은 수명)일수록 크게 벌어져요")
    _life_colorbar(fig, ax, norm)
    return _save(fig, path)


def logvar_vs_life(reports: list[CellReport], feature: np.ndarray, rho: float, path: Path) -> Path:
    _style()
    life = np.array([r.cycle_life for r in reports])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ok = np.isfinite(feature) & np.isfinite(life)
    for key, (label, color) in SPLITS.items():
        m = ok & np.array([r.split == key for r in reports])
        ax.scatter(feature[m], life[m], s=34, color=color, edgecolor=SURFACE, linewidth=1.2, label=f"{label} ({m.sum()})",
                   zorder=3)
    slope, intercept = np.polyfit(feature[ok], np.log10(life[ok]), 1)
    xs = np.linspace(feature[ok].min(), feature[ok].max(), 50)
    ax.plot(xs, 10 ** (slope * xs + intercept), color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_yscale("log")
    ax.set_yticks([100, 200, 500, 1000, 2000], labels=["100", "200", "500", "1,000", "2,000"])
    ax.set(xlabel="log₁₀ Var(ΔQ₁₀₀₋₁₀)", ylabel="사이클 수명")
    ax.set_title("초반 100사이클의 신호 하나로 수명이 갈려요")
    _subtitle(ax, f"ΔQ 분산이 클수록 수명이 짧아요 · 상관계수 r = {rho:.2f} (로그 수명 기준, {ok.sum()}개 셀)")
    leg = ax.legend(frameon=False, loc="upper right", labelcolor=INK_2)
    for h in leg.legend_handles:
        h.set_sizes([40])
    return _save(fig, path)


def _capacity(summary: pd.DataFrame, cell_id: str) -> tuple[np.ndarray, np.ndarray]:
    s = summary[summary["cell_id"] == cell_id].sort_values("cycle")
    s = s[~((s["QD"] == 0) & (s["IR"] == 0))]  # b1의 1번째 사이클 빈 자리
    qd = s["QD"].where((s["QD"] > QD_RANGE[0]) & (s["QD"] < QD_RANGE[1])).to_numpy()
    return s["cycle"].to_numpy(), qd


def quality_example(summary: pd.DataFrame, reports: list[CellReport], path: Path) -> Path:
    """왼쪽: 회복 튐이 있는 셀(확대), 오른쪽: 순간 하락이 가장 많은 셀. 둘은 원인이 달라 따로 센다."""
    _style()
    rec = max(reports, key=lambda r: len(r.recovery_cycles))
    dip = max(reports, key=lambda r: len(r.dip_cycles))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})

    cyc, qd = _capacity(summary, rec.cell_id)
    if rec.recovery_cycles:
        k = rec.recovery_cycles[0]
        keep = (cyc >= k - 60) & (cyc <= k + 60)
        left.plot(cyc[keep], qd[keep], lw=1.1, color="#2a78d6", marker="o", ms=2.2)
        i = int(np.searchsorted(cyc, k))
        left.scatter([cyc[i]], [qd[i]], s=90, marker="^", color=WARN, edgecolor=INK, linewidth=0.6, zorder=4)
        left.annotate(f"▲ 회복 튐 (사이클 {k})", (cyc[i], qd[i]), xytext=(12, 6), textcoords="offset points",
                      color=INK_2, fontsize=9)
    left.set(xlabel="사이클", ylabel="방전 용량 (Ah)")
    left.set_title(f"회복 튐: {rec.cell_id} ({len(rec.recovery_cycles)}곳)", fontsize=11, pad=8)

    cyc, qd = _capacity(summary, dip.cell_id)
    _, dips = capacity_events(qd)
    right.plot(cyc, qd, lw=1.0, color="#2a78d6", zorder=2)
    right.scatter(cyc[dips], qd[dips], s=16, marker="v", color=SERIOUS, edgecolor="none", zorder=3,
                  label=f"순간 하락: 1% 넘게 꺼졌다 돌아옴 ({len(dip.dip_cycles)}번)")
    right.axhline(EOL_AH, color=INK_2, lw=1, ls=(0, (4, 3)))
    right.set(xlabel="사이클")
    right.set_title(f"순간 하락: {dip.cell_id} ({dip.policy}, 수명 {dip.cycle_life:,.0f})", fontsize=11, pad=8)
    right.legend(frameon=False, loc="lower left", labelcolor=INK_2)

    fig.suptitle("품질 점검 예시: 회복 튐과 순간 하락은 원인이 달라 따로 세요", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, "회복 튐 = 시험이 멈췄다 재개된 흔적 · 순간 하락 = 측정 이상. 둘을 섞어 세던 첫 기준은 시행착오 기록에 남겼어요",
             color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)
