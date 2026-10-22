"""ΔQ 특징이 왜 통하나 (해석): 계산 → 그래프 → docs/interpret.md.

    python -m battery_lab.interpret_report

문서의 숫자는 모두 이 파일의 build()가 계산한다. 손으로 고치지 않는다.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .mdtext import write_doc
from . import interpret as it
from .features import REACH
from .figures import AXIS, BLUES, INK, INK_2, MUTED, SPLITS, _save, _style
from .korean import ieyo
from .model import NAMES
from .sources import PROCESSED, ROOT

DOCS = ROOT / "docs"
SIGNAL_TOP = 3.2   # V. 이보다 높은 전압(주 봉우리 위쪽)에서 신호가 얼마나 약한지 따로 본다
TRAIN_FLAT = 1.10  # 학습 RMSE가 최솟값의 110% 안인 전압 구간 = 학습 오차만으로는 전압을 가르기 어려운 구간

# 논문 수치는 원문에서 직접 확인한 것만 쓴다 (troubleshooting.md 16번).
# Attia, Severson, Witmer (2021), "Statistical learning for accurate and interpretable battery lifetime prediction"
ATTIA_URL = "https://arxiv.org/abs/2101.01885"
ATTIA_V = 2.959                                                     # 예시 셀의 Qn − Q2가 가장 깊은 전압이라 골랐다고 적혀 있다
ATTIA_VS_VAR = {"train": 8.7, "test": -13.8, "secondary_test": 7.1}  # 2.959V 모델 오차가 단순 모델(분산)보다 몇 % 높은가
ATTIA_TRAIN_BEST_V = 2.911                                          # log10 단일 전압 모델, 학습 오차가 가장 작은 전압
ATTIA_TRAIN_BEST = {"train": 108, "test": 118, "secondary_test": 214}  # 그 전압의 RMSE (사이클)
ATTIA_TABLE_V, ATTIA_TABLE_TRAIN = 2.913, 109                       # 같은 모델이 논문 요약 표에는 이렇게 적혀 있다 (본문과 조금 다름)
SPLIT_NAMES = {"train": "학습", "test": "테스트", "secondary_test": "2차 테스트"}
GROUP_COLORS = (BLUES(0.0), BLUES(0.5), BLUES(1.0))  # 짧은 수명 = 연한 파랑 → 긴 수명 = 진한 파랑 (figures.py 규칙)


# ---------------------------------------------------------------- 계산

def analyze(data: dict) -> dict:
    """collect() 결과에서 문서에 쓸 숫자를 모두 계산한다."""
    cells, V, D = data["cells"], data["voltage"], data["dq"]
    life, split = cells["cycle_life"].to_numpy(), cells["split"].to_numpy()
    y = np.log10(life)
    spec = it.correlation_spectrum(V, D, life)
    strong = {k: it.strongest(spec, f"r_{k}") for k in it.TRANSFORMS}
    bands = {k: it.band(V, spec[f"r_{k}"].to_numpy()) for k in it.TRANSFORMS}
    logq = it.transform(D, "log")
    boot = it.boot_peak_voltage(V, logq, life)
    top = V >= SIGNAL_TOP
    above = float(np.nanmax(np.abs(spec.loc[top, "r_log"]))) if top.any() else np.nan

    uni = {k: it.univariate_errors(V, D, life, split, k) for k in it.TRANSFORMS}
    models = {"best_log": it.pick(uni["log"], "train"), "attia": it.pick(uni["log"], voltage=ATTIA_V),
              "best_raw": it.pick(uni["raw"], "train")}
    peek = {k: it.pick(uni["log"], k) for k in ("test", "secondary_test")}  # 참고용: 평가 셀을 보고 고른 값
    var = it.variance_model(D, life, split)
    logvar = np.log10(np.var(D, axis=1))
    j = strong["log"]["index"]

    neg10 = -data["dqdv_early"]
    mean10 = neg10.mean(axis=0)
    win = (V >= it.PEAK_WINDOW[0]) & (V <= it.PEAK_WINDOW[1])
    peak_v = float(V[win][np.argmax(mean10[win])])
    peak_h = float(mean10[win].max())
    peaks = it.peak_metrics(V, data["dqdv_early"], data["dqdv_late"])
    counts = np.array([it.count_peaks(V, row) for row in data["dqdv_early"]])

    groups = it.life_groups(life)
    gq = it.group_curves(D, groups)
    depth = {g: it.depth_and_recovery(V, c["mean"]) for g, c in gq.items()}
    flows = {g: it.moved(V, c["mean"]) for g, c in gq.items()}
    d_ir = it.ir_change(data["summary"], cells["cell_id"])
    ir_early = it.ir_at(data["summary"], cells["cell_id"], it.EARLY)
    with np.errstate(divide="ignore"):
        inv_train = 1 / uni["log"]["rmse_train"].to_numpy(dtype=np.float64)  # band()는 '큰 값' 기준이라 뒤집어 넣는다
    train_flat = it.band(V, inv_train, 1 / TRAIN_FLAT)

    per_cell = cells[["cell_id", "batch", "policy", "split", "cycle_life", "late_used", "early_used"]].copy()
    per_cell["group"] = groups
    per_cell["dq_at_signal_mah"] = D[:, j] * 1000
    per_cell["logvar"] = logvar
    per_cell = pd.concat([per_cell, peaks], axis=1)
    per_cell["ir_change_mohm"] = d_ir
    per_cell["peaks_cycle10"] = counts

    group_rows = []
    for g in sorted(gq):
        m = groups == g
        group_rows.append({
            "group": g, "name": it.GROUP_NAMES[g], "cells": int(m.sum()),
            "life_min": float(life[m].min()), "life_max": float(life[m].max()),
            "dq_signal_median_mah": float(np.median(D[m, j]) * 1000),
            "deepest_v": depth[g]["voltage"], "deepest_mah": depth[g]["deepest_mah"],
            "end_mah": depth[g]["end_mah"], "recovered": depth[g]["recovered"],
            "shift_median_mv": float(peaks.loc[m, "shift_mv"].median()),
            "height_change_median_pct": float(peaks.loc[m, "height_change_pct"].median()),
            "ir_change_median_mohm": float(np.nanmedian(d_ir[m])),
        })
    return {
        "cells": len(cells), "skipped": data["skipped"], "voltage": V, "spec": spec, "uni": uni,
        "substituted": int(((cells["late_used"] != it.LATE) | (cells["early_used"] != it.EARLY)).sum()),
        "strong": strong, "bands": bands, "boot_p5": float(np.percentile(boot, 5)),
        "boot_p50": float(np.median(boot)), "boot_p95": float(np.percentile(boot, 95)), "above_top_r": above,
        "models": models, "peek": peek, "var": var,
        "r_logvar_signal": it.corr(logvar, logq[:, j]),
        "mean_dqdv10": mean10, "dqdv10_p10": np.percentile(neg10, 10, axis=0),
        "dqdv10_p90": np.percentile(neg10, 90, axis=0),
        "peak_v": peak_v, "peak_h": peak_h, "mean_peaks": it.count_peaks(V, -mean10),
        "one_peak_cells": int((counts == 1).sum()),
        "signal_height_pct": float(mean10[j] / peak_h * 100),
        "shift_median_mv": float(peaks["shift_mv"].median()),
        "height_change_median_pct": float(peaks["height_change_pct"].median()),
        "r_shift_life": it.corr(peaks["shift_mv"], y), "r_height_life": it.corr(peaks["height_change_pct"], y),
        "grid_mv": float(abs(V[1] - V[0]) * 1000),
        "groups": pd.DataFrame(group_rows), "group_curves": gq, "flows": flows,
        "all_depth": it.depth_and_recovery(V, D.mean(axis=0)),
        "ir_median_mohm": float(np.nanmedian(d_ir)), "ir_cells": int(np.isfinite(d_ir).sum()),
        "ir_early_median_mohm": float(np.nanmedian(ir_early)), "train_flat": train_flat,
        "r_ir_signal": it.corr(d_ir, logq[:, j]), "r_ir_life": it.corr(d_ir, y),
        "per_cell": per_cell,
        "short_loss_v": float(V[np.argmin(flows[0])]),
        "r_height_life_full": it.corr(it.peak_metrics(V, data["dqdv_early"], data["dqdv_late"],
                                                      (float(V.min()), float(V.max())))["height_change_pct"], y),
        "dqdv_integral_mah": float(-np.trapezoid(data["dqdv_early"].mean(axis=0) - data["dqdv_late"].mean(axis=0), V) * 1000),
        "full_range_peak_cells": [c for c, v in zip(cells["cell_id"], V[np.argmax(neg10, axis=1)])
                                  if not it.PEAK_WINDOW[0] <= v <= it.PEAK_WINDOW[1]],
    }


# ---------------------------------------------------------------- 그래프

def plt_subplots(rows: int, figsize: tuple[float, float]):
    """세로로 쌓은 그래프 rows개. 가로축(전압)을 같이 쓴다 (세로축을 둘 겹치지 않는다)."""
    import matplotlib.pyplot as plt

    return plt.subplots(rows, 1, figsize=figsize, sharex=True)


def _voltage_axis(ax) -> None:
    ax.set_xlim(3.5, 2.0)  # 방전은 높은 전압 → 낮은 전압
    ax.set_xlabel("전압 (V)")


def spectrum_figure(a: dict, path: Path) -> Path:
    """위: 전압별 상관 세기, 가운데: 전압별 단일 모델 오차, 아래: 10번째 사이클 평균 −dQ/dV."""
    _style()
    V, spec = a["voltage"], a["spec"]
    hi, lo = a["bands"]["log"]
    fig, axes = plt_subplots(3, figsize=(8.6, 9.6))
    for ax in axes:
        ax.axvspan(lo, hi, color="#d9e7f9", alpha=0.6, lw=0, zorder=0)

    ax = axes[0]
    ax.plot(V, np.abs(spec["r_log"]), color="#1c5cab", lw=1.6)
    ax.plot(V, np.abs(spec["r_raw"]), color=MUTED, lw=1.2)
    s = a["strong"]["log"]
    ax.annotate(f"log₁₀|ΔQ| : 가장 강한 곳 {s['voltage']:.3f} V, |r| = {abs(s['r']):.3f}", (s["voltage"], abs(s["r"])),
                xytext=(2.95, 0.42), color="#1c5cab", fontsize=9, va="center",
                arrowprops={"arrowstyle": "-", "color": "#1c5cab", "lw": 0.8})
    j = int(np.argmin(np.abs(V - 2.5)))
    ax.text(2.5, abs(spec["r_raw"].iloc[j]) - 0.05, "ΔQ 그대로", color=MUTED, fontsize=9, ha="center", va="top")
    ax.axhline(abs(a["var"]["r"]), color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(3.45, abs(a["var"]["r"]) + 0.02, f"단순 모델 |r| = {abs(a['var']['r']):.3f}", color=INK_2, fontsize=8.5,
            va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("log 수명과의 상관 |r|")
    ax.set_title("① 전압 칸마다 ΔQ₁₀₀₋₁₀와 수명의 상관 세기", fontsize=11, pad=8)

    ax = axes[1]
    t = a["uni"]["log"]
    for key, (label, color) in SPLITS.items():
        ax.plot(t["voltage"], t[f"rmse_{key}"], color=color, lw=1.3, label=label)
    b = a["models"]["best_log"]
    ax.axvline(b["voltage"], color=INK_2, lw=1, ls=(0, (2, 2)))
    ax.text(b["voltage"] - 0.02, 385, f"학습 셀로 고른 전압\n{b['voltage']:.3f} V", color=INK_2, fontsize=8.5, ha="left",
            va="top")
    ax.set_ylim(80, 400)
    ax.set_ylabel("RMSE (사이클)")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2, ncol=3, fontsize=8.5)
    ax.set_title("② 전압 칸 하나의 log₁₀|ΔQ|로만 만든 모델의 오차 (학습 41개로 맞춤)", fontsize=11, pad=8)

    ax = axes[2]
    ax.fill_between(V, a["dqdv10_p10"], a["dqdv10_p90"], color="#b7d3f6", alpha=0.7, lw=0)
    ax.plot(V, a["mean_dqdv10"], color="#1c5cab", lw=1.6)
    ax.annotate(f"평균 봉우리 {a['peak_v']:.3f} V", (a["peak_v"], a["peak_h"]), xytext=(3.42, a["peak_h"] * 0.8),
                color=INK_2, fontsize=9, arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    ax.text((hi + lo) / 2, a["peak_h"] * 0.55, f"신호가 모인 구간 (파란 띠)\n{hi:.2f}~{lo:.2f} V", color="#1c5cab",
            fontsize=9, ha="center", va="center")  # 띠 안 가운데: 띠 밖에 두면 다른 구간을 가리키는 것처럼 보인다
    ax.set_ylabel("−dQ/dV (Ah/V)")
    ax.set_title("③ 10번째 사이클 방전의 미분 용량 (선 = 평균, 띠 = 셀 10~90%)", fontsize=11, pad=8)
    _voltage_axis(ax)

    fig.suptitle("수명 신호는 4C 방전 봉우리의 아래쪽 전압에 모여 있어요", x=0.01, y=0.995, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.962, f"셀 {a['cells']}개 · 파란 띠 = log₁₀|ΔQ|의 |r|이 최댓값의 95% 이상인 구간 · "
             "가로축은 방전 순서대로 높은 전압 → 낮은 전압", color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _save(fig, path)


def groups_figure(a: dict, path: Path) -> Path:
    """위: 수명 무리별 ΔQ(V) 평균과 25~75% 띠, 아래: 어느 전압에서 용량이 줄고 늘었나."""
    _style()
    V, gq, g = a["voltage"], a["group_curves"], a["groups"]
    fig, axes = plt_subplots(2, figsize=(8.6, 7.4))
    ax = axes[0]
    for row in g.itertuples():
        c = gq[row.group]
        color = GROUP_COLORS[row.group]
        if row.group != 1:
            ax.fill_between(V, c["p25"] * 1000, c["p75"] * 1000, color=color, alpha=0.22, lw=0)
        ax.plot(V, c["mean"] * 1000, color=color, lw=1.8 if row.group != 1 else 1.1,
                label=f"{row.name} ({row.life_min:,.0f}~{row.life_max:,.0f}사이클, {row.cells}개)")
    ax.axhline(0, color=AXIS, lw=1)
    short = g.iloc[0]
    ax.annotate(f"짧은 수명 무리가 가장 깊은 곳: {short.deepest_v:.2f} V, {short.deepest_mah:.0f} mAh",
                (short.deepest_v, short.deepest_mah), xytext=(short.deepest_v, short.deepest_mah * 1.2), color=INK_2,
                fontsize=9, ha="center", va="center", arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    ax.set_ylim(short.deepest_mah * 1.3, 3)
    ax.set_ylabel("ΔQ₁₀₀₋₁₀ (mAh)")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2, fontsize=8.5)
    ax.set_title("① 수명 무리별 ΔQ₁₀₀₋₁₀(V): 선 = 평균, 띠 = 25~75% (짧은·긴 수명만)", fontsize=11, pad=8)

    ax = axes[1]
    for row in g.itertuples():
        ax.plot(V, a["flows"][row.group], color=GROUP_COLORS[row.group], lw=1.6 if row.group != 1 else 1.0)
    ax.axhline(0, color=AXIS, lw=1)
    low = float(np.min([f.min() for f in a["flows"].values()]))
    high = float(np.max([f.max() for f in a["flows"].values()]))
    ax.text(2.95, low * 0.5, "0보다 아래 = 100번째 사이클에서 이 전압 칸에 나온 용량이 줄었어요\n"
            "0보다 위 = 이 전압 칸에 나온 용량이 늘었어요", color=INK_2, fontsize=9, va="center", linespacing=1.6)
    ax.set_ylim(low * 1.1, high * 2.5)
    ax.set_ylabel("Δ(−dQ/dV) (Ah/V)")
    ax.set_title(f"② 어느 전압에서 용량이 줄고 늘었나 (①을 전압으로 미분, 약 {it.SMOOTH * a['grid_mv']:.0f}mV로 다듬음)",
                 fontsize=11, pad=8)
    _voltage_axis(ax)

    fig.suptitle("수명이 짧은 셀일수록 봉우리 근처에서 더 많이 잃어요", x=0.01, y=0.995, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.944, "수명 순서로 셀을 세 무리로 나눔 · 연한 파랑 = 짧은 수명, 진한 파랑 = 긴 수명",
             color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save(fig, path)


# ---------------------------------------------------------------- 문서

def _err(row: dict, key: str) -> str:
    return f"{row[f'rmse_{key}']:.0f}사이클 ({row[f'mape_{key}']:.1f}%)"


def _pct_diff(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f}%"


def _r(x: float, fmt: str = "+.2f") -> str:
    """상관계수 글자. 계산할 수 없으면(값이 모두 같음 등) 그렇다고 적는다."""
    return format(x, fmt) if np.isfinite(x) else "계산 불가"


def _strength(*rs: float) -> str:
    """상관의 세기를 말로. 가장 강한 것 기준 (|r| < 0.3 약함, < 0.6 중간)."""
    top = max((abs(r) for r in rs if np.isfinite(r)), default=np.nan)
    if not np.isfinite(top):
        return "계산할 수 없어요"
    return "약해요" if top < 0.3 else ("중간 정도예요" if top < 0.6 else "강해요")


def _lower(a: float, b: float) -> str:
    """'a가 b보다 낮/높'의 앞 글자. 숫자가 바뀌어도 문장이 틀리지 않도록 비교도 코드가 한다."""
    return "낮" if a < b else "높"


def markdown(a: dict, mistakes: list[str]) -> str:
    s, sr = a["strong"]["log"], a["strong"]["raw"]
    hi, lo = a["bands"]["log"]
    hir, lor = a["bands"]["raw"]
    m, var = a["models"], a["var"]
    b, at = m["best_log"], m["attia"]
    g = a["groups"]
    short, long_ = g.iloc[0], g.iloc[-1]
    ratio = short.dq_signal_median_mah / long_.dq_signal_median_mah
    below_peak = (a["peak_v"] - s["voltage"]) * 1000
    lines = [
        "# ΔQ 특징이 왜 통하나: 어느 전압이 수명을 알려 줄까?",
        "",
        f"`python -m battery_lab.interpret_report`가 자동으로 만든 문서예요 (만든 날 {date.today().isoformat()}).",
        "",
        "## 질문",
        "",
        f"{NAMES['simple']}은 ΔQ₁₀₀₋₁₀(V) 곡선(100번째와 10번째 방전 곡선의 차이, 전압 칸 1,000개)의 **분산 하나**로 수명을 맞혀요"
        f" (논문의 variance 모델과 같은 형태). 로그 수명과의 상관 r은 {ieyo(format(var['r'], '.3f'))}."
        " 상관 r은 두 값이 직선으로 함께 움직이는 정도로 −1~+1 사이 값이고, −1에 가까울수록 '한쪽이 크면 다른 쪽은 작다'는 관계가 뚜렷해요."
        " 그런데 분산은 1,000개 값을 한 숫자로 뭉치면서 '어느 전압에서' 달라졌는지는 버려요.",
        "",
        "1. 1,000개 전압 칸 중 **어디에** 수명 신호가 모여 있을까?",
        "2. 그 전압은 방전 곡선의 어디(평탄 구간·봉우리)이고, 전기화학적으로 **무엇을 뜻할 수 있을까?**",
        "3. 수명이 짧은 셀과 긴 셀의 ΔQ 곡선은 **어떻게 다를까?**",
        "",
        "## 왜 중요한가",
        "",
        "- **셀 개발:** \"모델이 맞혔다\"에서 멈추면 다음 실험을 설계할 수 없어요. 신호가 어느 전압 구간에서 나오는지 알면,"
        " 그 구간을 저율(천천히 충방전) 진단으로 정밀하게 다시 재서 원인(리튬 손실인지 재료 손상인지)을 가르는 실험을 계획할 수 있어요.",
        "- **공정 개발·품질:** 곡선 전체가 아니라 특정 전압 구간의 용량 변화만으로도 수명 신호가 잡힌다면,"
        " 초기 사이클 검사를 더 단순하게 만들 실마리가 돼요. (이 데이터는 공정 데이터가 아니라서 가능성만 말할 수 있어요.)",
        "- **면접:** \"ΔQ 분산이 왜 통하나요?\"라는 질문에 \"통계적으로 그렇다\"가 아니라 \"방전 곡선의 이 부분이 이렇게 바뀌기 때문\"이라고"
        " 데이터로 답할 수 있어요.",
        "",
        "## 방법",
        "",
        f"- **데이터:** 셀 {a['cells']}개. ΔQ₁₀₀₋₁₀(V) = 100번째 − 10번째 사이클의 방전 곡선 Q(V)"
        f" ([interpret.py](../battery_lab/interpret.py)). 곡선은 행 번호가 아니라 사이클 번호로 찾고, 깨끗하지 않으면"
        f" {it.LATE}번째는 앞쪽({it.LATE - REACH}~{it.LATE}번째)에서만, {it.EARLY}번째는 ±{REACH}사이클 안에서 가까운 곡선을 써요"
        f" (이번엔 대신 쓴 셀 {a['substituted']}개).",
        "- **상관 스펙트럼:** 전압 칸마다 셀들의 ΔQ(V)와 log10(수명)의 상관 r을 구해요. ΔQ를 그대로 쓴 것과 log10|ΔQ|(절댓값의 로그)를"
        " 쓴 것 두 가지예요. 단순 모델도 로그를 쓰고, 수명과의 관계가 직선보다 곡선에 가까워서 로그를 함께 봐요.",
        f"- **흔들림 확인:** 셀 {a['cells']}개를 복원 추출(같은 셀이 여러 번 뽑힐 수 있게)로 {it.BOOT:,}번 다시 뽑아,"
        f" 상관이 가장 강한 전압이 얼마나 움직이는지 봐요 (난수 고정 seed {it.SEED}).",
        "- **전압별 단일 특징 모델:** 전압 칸 하나의 값만으로 log10(수명)에 직선을 맞춰요. 논문과 같은 나누기(학습 41 · 테스트 43 ·"
        " 2차 테스트 40)를 쓰고, **전압 고르기도 학습 셀로만** 해요. 평가 셀 오차를 보고 전압을 고르면 답을 보고 고르는 셈이에요.",
        "- **dQ/dV(미분 용량):** 데이터에 들어 있는 10번째 사이클의 dQ/dV를 셀마다 모아 평균을 그려요 (봉우리 이동·높이 변화는 100번째와 비교). 방전에서는 dQ/dV가 음수라"
        f" −dQ/dV로 뒤집어 봉우리가 위로 오게 했어요. 봉우리는 {it.PEAK_WINDOW[0]}~{it.PEAK_WINDOW[1]} V 안에서 찾아요.",
        "- **수명 무리:** 수명 순서로 셀을 세 무리(짧은·중간·긴 수명)로 나눠 ΔQ(V)의 평균과 25~75% 범위를 비교해요.",
        "- **ΔQ의 미분:** ΔQ(V)는 3.5V부터 그 전압까지 나온 용량 변화를 모두 더한 **누적 값**이에요. 어느 전압에서 변화가 생겼는지 보려고"
        f" ΔQ 곡선을 약 {it.SMOOTH * a['grid_mv']:.0f}mV 폭으로 다듬은 뒤 전압으로 미분했어요.",
        f"- **내부저항:** 요약표의 내부저항(IR)으로 10번째 → 100번째 사이클 변화를 계산해요 (각 사이클 ±{it.IR_REACH}의 중앙값).",
        "- 물리 시뮬레이션(전기화학 모델)은 돌리지 않았어요. 아래의 원인 해석은 모두 데이터와 함께 나타나는 것을 바탕으로 한 **가설**이에요.",
        "",
        "## 결과",
        "",
        "### 1. 신호는 봉우리 아래쪽 넓은 전압 구간에 모여 있어요",
        "",
        "| 항목 | ΔQ 그대로 | log10\\|ΔQ\\| |",
        "|---|---|---|",
        f"| 상관이 가장 강한 전압 | {sr['voltage']:.3f} V | {s['voltage']:.3f} V |",
        f"| 그때의 상관 r | {sr['r']:+.3f} | {s['r']:+.3f} |",
        f"| \\|r\\|이 최댓값의 95% 이상인 구간 | {hir:.2f} ~ {lor:.2f} V | {hi:.2f} ~ {lo:.2f} V |",
        f"| 셀 다시 뽑기 {it.BOOT:,}번: 가장 강한 전압의 90% 범위 | | {a['boot_p95']:.2f} ~ {a['boot_p5']:.2f} V (중앙값 {a['boot_p50']:.3f} V) |",
        f"| {SIGNAL_TOP} V보다 높은 전압에서 가장 강한 \\|r\\| | | {a['above_top_r']:.3f} |",
        "",
        f"*비교: 단순 모델(log10 Var ΔQ)의 상관 r = {var['r']:.3f}. ΔQ가 음수(용량이 줄어듦)라서, ΔQ 그대로는 양의 상관"
        " (덜 줄어든 셀일수록 오래 가는 경향), log10|ΔQ|는 음의 상관(많이 줄어든 셀일수록 일찍 끝나는 경향)이에요.*",
        "",
        "![전압별 상관과 dQ/dV](figures/interpret_spectrum.png)",
        "",
        "### 2. 전압 칸 하나로 만든 모델 vs 단순 모델",
        "",
        "| 모델 | 학습 | 테스트 | 2차 테스트 |",
        "|---|---|---|---|",
        f"| {NAMES['simple']}: log10 Var(ΔQ) (기준) | {_err(var, 'train')} | {_err(var, 'test')} | {_err(var, 'secondary_test')} |",
        f"| 전압 하나: log10\\|ΔQ({b['voltage']:.3f} V)\\| (학습 셀로 고름) | {_err(b, 'train')} | {_err(b, 'test')} | {_err(b, 'secondary_test')} |",
        f"| 전압 하나: log10\\|ΔQ({at['voltage']:.3f} V)\\| (Attia 외가 쓴 {ATTIA_V} V) | {_err(at, 'train')} | {_err(at, 'test')} | {_err(at, 'secondary_test')} |",
        f"| 전압 하나: ΔQ({m['best_raw']['voltage']:.3f} V) 그대로 (학습 셀로 고름) | {_err(m['best_raw'], 'train')} | {_err(m['best_raw'], 'test')} | {_err(m['best_raw'], 'secondary_test')} |",
        "",
        "- 오차는 RMSE(평균 제곱근 오차. 예측이 평균적으로 몇 사이클 빗나갔는지)와 괄호 안 평균 % 오차(MAPE)예요."
        f" 단순 모델의 테스트 / 2차 테스트 RMSE {var['rmse_test']:.0f} / "
        f"{var['rmse_secondary_test']:.0f}사이클은 [데이터 크기 실험](data_size.md)의 기준점과 같은 값이에요.",
        f"- **비교 (원문 확인):** Attia 외(2021, [{ATTIA_URL}]({ATTIA_URL}))는 ΔQ₁₀₀₋₁₀를 {ATTIA_V} V 한 점에서만 읽은 모델이"
        f" 단순 모델과 비슷하다고 보고했어요 (단순 모델 대비 오차가 학습 {ATTIA_VS_VAR['train']:+.1f}%, 테스트 {ATTIA_VS_VAR['test']:+.1f}%,"
        f" 2차 테스트 {ATTIA_VS_VAR['secondary_test']:+.1f}%). 이 전압은 예시 셀 하나의 Qn − Q2(2번째 사이클 기준 용량 차이)가 가장 깊은"
        f" 전압이라 골랐다고 적혀 있어요. 우리 계산으로는 같은 전압에서 단순 모델 대비 {_pct_diff(at['rmse_train'], var['rmse_train'])},"
        f" {_pct_diff(at['rmse_test'], var['rmse_test'])}, {_pct_diff(at['rmse_secondary_test'], var['rmse_secondary_test'])}예요."
        f" 우리 {a['cells']}개 셀의 평균 ΔQ₁₀₀₋₁₀가 가장 깊은 전압은 {a['all_depth']['voltage']:.3f} V예요.",
        f"- 같은 논문에서 학습 오차로 고른 전압은 {ATTIA_TRAIN_BEST_V} V였고 RMSE는 학습 {ATTIA_TRAIN_BEST['train']} · 테스트"
        f" {ATTIA_TRAIN_BEST['test']} · 2차 테스트 {ATTIA_TRAIN_BEST['secondary_test']}사이클이에요. 우리는 {b['voltage']:.3f} V에서"
        f" {b['rmse_train']:.0f} · {b['rmse_test']:.0f} · {b['rmse_secondary_test']:.0f}사이클이에요. 논문은 단일 특징에도 Elastic Net(제약을 건"
        " 선형 회귀)과 표준화를 썼고 우리는 보통 직선 맞추기라, 숫자가 똑같지는 않아요."
        f" (논문 요약 표에는 같은 모델이 {ATTIA_TABLE_V} V, 학습 {ATTIA_TABLE_TRAIN}사이클로 적혀 있어 본문과 조금 달라요. 여기서는 본문 값을 썼어요.)",
        f"- *참고 (답을 보고 고른 값):* 테스트 셀 오차가 가장 작은 전압은 {a['peek']['test']['voltage']:.3f} V"
        f" ({a['peek']['test']['rmse_test']:.0f}사이클), 2차 테스트는 {a['peek']['secondary_test']['voltage']:.3f} V"
        f" ({a['peek']['secondary_test']['rmse_secondary_test']:.0f}사이클)예요. 평가 셀마다 '좋은 전압'이 달라서, 이런 숫자는 성능으로 쓰면 안 돼요.",
        "",
        "### 3. 그 전압은 방전 곡선의 어디일까 (dQ/dV)",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 10번째 사이클 평균 −dQ/dV의 봉우리 | {a['peak_v']:.3f} V, {a['peak_h']:.2f} Ah/V |",
        f"| 평균 곡선의 봉우리 개수 ({it.COUNT_WINDOW[0]}~{it.COUNT_WINDOW[1]} V, 주 봉우리의 {it.PROMINENCE * 100:.0f}% 이상) | {a['mean_peaks']}개 |",
        f"| 봉우리가 하나만 보이는 셀 | {a['one_peak_cells']}개 / {a['cells']}개 |",
        f"| 신호가 가장 강한 전압({s['voltage']:.3f} V)의 위치 | 봉우리보다 {below_peak:.0f} mV 아래, 높이는 봉우리의 {a['signal_height_pct']:.0f}% |",
        f"| 봉우리 위치 이동 (10 → 100번째, 중앙값) | {a['shift_median_mv']:+.1f} mV (전압 칸 간격 {a['grid_mv']:.1f} mV) |",
        f"| 봉우리 높이 변화 (중앙값) | {a['height_change_median_pct']:+.1f}% |",
        f"| 수명과의 상관: 봉우리 이동 / 높이 변화 | r = {_r(a['r_shift_life'])} / {_r(a['r_height_life'])} |",
        "",
        "### 4. 짧은 수명 vs 긴 수명",
        "",
        f"| 무리 | 셀 | 수명 | ΔQ({s['voltage']:.3f} V) 중앙값 | 평균 ΔQ가 가장 깊은 곳 | 2.0V 끝의 평균 ΔQ | 끝까지 되찾은 비율 | 봉우리 이동 중앙값 | 봉우리 높이 변화 중앙값 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in g.itertuples():
        lines.append(f"| {r.name} | {r.cells} | {r.life_min:,.0f}~{r.life_max:,.0f} | {r.dq_signal_median_mah:.1f} mAh | "
                     f"{r.deepest_v:.2f} V, {r.deepest_mah:.1f} mAh | {r.end_mah:.1f} mAh | {r.recovered * 100:.0f}% | "
                     f"{r.shift_median_mv:+.1f} mV | {r.height_change_median_pct:+.1f}% |")
    ad = a["all_depth"]
    lines += [
        "",
        "![수명 무리별 ΔQ](figures/interpret_groups.png)",
        "",
        "### 5. 내부저항으로 본 확인",
        "",
        f"- 10번째 → 100번째 사이클의 내부저항 변화는 중앙값 {a['ir_median_mohm']:+.3f} mΩ({a['ir_cells']}개 셀)으로 아주 작아요."
        f" 10번째 사이클 내부저항(중앙값 {a['ir_early_median_mohm']:.1f} mΩ)의"
        f" {abs(a['ir_median_mohm']) / a['ir_early_median_mohm'] * 100:.1f}%예요.",
        f"- 내부저항 변화와 log10|ΔQ({s['voltage']:.3f} V)|의 상관(r = {_r(a['r_ir_signal'])})과 로그 수명과의 상관(r = {_r(a['r_ir_life'])})은 둘 다 {_strength(a['r_ir_signal'], a['r_ir_life'])}.",
        "",
        "## 읽는 법 (숫자는 코드가 계산)",
        "",
        f"1. **신호의 자리:** 상관은 {s['voltage']:.3f} V에서 가장 강하고(r = {s['r']:+.3f}), {hi:.2f}~{lo:.2f} V의 넓은 구간이 거의"
        f" 같은 세기예요. 반대로 {SIGNAL_TOP} V보다 높은 전압에서는 가장 강한 곳도 |r| = {a['above_top_r']:.2f} 정도예요. 셀을 다시 뽑아도"
        f" 가장 강한 전압은 90%가 {a['boot_p95']:.2f}~{a['boot_p5']:.2f} V 안에 있어요. 신호는 '한 점'이 아니라 **봉우리 아래쪽 구간 전체**에 퍼져 있어요.",
        f"2. **전압 하나로도 단순 모델과 비슷하게 맞혀요.** 학습 셀로 고른 {b['voltage']:.3f} V 한 점의 모델은 테스트"
        f" {b['rmse_test']:.0f}사이클로 단순 모델({var['rmse_test']:.0f})보다 {_lower(b['rmse_test'], var['rmse_test'])}고, 2차 테스트는"
        f" {b['rmse_secondary_test']:.0f}사이클로 단순 모델({var['rmse_secondary_test']:.0f})보다"
        f" {_lower(b['rmse_secondary_test'], var['rmse_secondary_test'])}아요."
        f" 로그 없이 ΔQ 그대로 쓰면 테스트 {m['best_raw']['rmse_test']:.0f}사이클로 크게 나빠져요. ΔQ와 로그 수명의 관계가 직선보다"
        f" 곡선에 가깝기 때문으로 보여요 (가장 강한 |r|도 그대로 {abs(sr['r']):.3f}, 로그 {abs(s['r']):.3f}).",
        f"3. **분산이 통하는 이유 (관찰):** 세 무리의 평균 ΔQ 곡선은 모양(봉우리에서 꺼지고, 그 아래에서 조금 회복)이 비슷하고 **골의 깊이가 달라요.**"
        f" 그래서 분산은 사실상 '골이 얼마나 깊은가'를 재는 셈이에요. 실제로 셀마다 log 분산과 log10|ΔQ({s['voltage']:.3f} V)|의 상관이"
        f" 아주 높아요 (r = {_r(a['r_logvar_signal'], '.3f')}). 수명이 짧은 무리의 ΔQ({s['voltage']:.3f} V) 중앙값은 긴 수명 무리의 약 {ratio:.1f}배예요."
        " Attia 외(2021)도 IQR 같은 '퍼짐' 특징이 사실상 ΔQ의 3.2V 근처 '어깨'(급하게 꺼지는 부분)의 크기를 재는 셈이라고 해석했어요.",
        f"4. **ΔQ는 누적 값이에요.** {s['voltage']:.3f} V의 ΔQ는 '그 전압에서 생긴 변화'가 아니라 3.5V부터 그 전압까지 생긴 변화의 합이에요."
        f" '수명 무리별 ΔQ' 그림의 ②를 보면 100번째 사이클은 **봉우리({a['peak_v']:.2f} V) 근처에서 나오던 용량이 줄고**(짧은 수명 무리에서 가장 크게 줄어든 곳"
        f" {a['short_loss_v']:.2f} V), **약 {ad['voltage']:.2f} V보다 낮은 전압에서는 조금 늘어요.** 전체 평균으로 가장 깊은 곳이"
        f" {ad['voltage']:.2f} V, {ad['deepest_mah']:.1f} mAh이고, 2.0V 끝에서는 {ad['end_mah']:.1f} mAh라서 {ad['recovered'] * 100:.0f}%를 되찾아요."
        f" 상관이 가장 강한 전압({s['voltage']:.3f} V)이 '줄어든 구간과 늘어난 구간의 경계'({ad['voltage']:.2f} V)에서"
        f" {abs(ad['voltage'] - s['voltage']) * 1000:.0f} mV 거리로 가까운 것도 이와 관련 있어 보여요. 경계 근처의 ΔQ는 위쪽에서 잃은 양을"
        " 가장 많이 모아 담고 있으니까요 (가설).",
        f"5. **봉우리 꼭대기 한 점은 신호가 약해요.** 봉우리 위치 이동(r = {_r(a['r_shift_life'])})과 높이 변화(r = {_r(a['r_height_life'])})는"
        f" ΔQ(|r| = {abs(s['r']):.3f})보다 수명과의 관계가 훨씬 약해요. 봉우리 이동은 중앙값 {a['shift_median_mv']:+.1f} mV로 전압 칸 간격"
        f"({a['grid_mv']:.1f} mV)과 비슷할 만큼 작아서 재기 어렵고, 넓은 구간의 용량을 더한 ΔQ가 이런 작은 변화를 더 안정적으로 잡는 것으로"
        " 보여요 (가설).",
        "",
        "### 전기화학으로는 무엇을 뜻할 수 있을까 (가설)",
        "",
        "- **이 셀의 방전 곡선:** LFP 양극은 전압이 거의 평평해서, LFP/흑연 셀 곡선의 굴곡은 주로 흑연 음극이 리튬을 단계적으로 내놓는"
        f" 과정(staging)에서 온다고 흔히 설명해요. Attia 외(2021)도 ΔQ의 3.2V 근처 급한 '어깨'를 흑연의 가장 큰 평탄 구간 변화와 연결했어요."
        f" 우리 데이터의 4C 방전에서는 이 단계들이 하나의 봉우리({a['peak_v']:.3f} V)로 뭉쳐 보여요.",
        "- **가설 A · 분극(전압 손실) 증가:** 같은 양의 리튬이 나오긴 하지만 곡선 전체가 아래로 밀려 **더 낮은 전압에서** 나온다면, 저항 증가·이온 이동이"
        " 느려지는 것(확산 한계)·전극 부위별 반응 차이(불균일)가 커졌을 수 있어요. '수명 무리별 ΔQ' 그림의 ②에서 봉우리 근처에서 줄어든 용량의 일부가 낮은 전압에서"
        " 다시 나오는 모양은 이 가설과 어긋나지 않아요.",
        "- **가설 B · 진짜 용량 손실:** 2.0V까지 가도 되찾지 못한 부분은 사용할 수 있는 리튬이 줄었거나(리튬 손실, LLI) 리튬을 담는 재료가"
        " 줄어든(활물질 손실, LAM) 몫일 수 있어요. 다만 분극이 커져 2.0V에 더 일찍 닿아도 되찾지 못한 몫이 생기므로, 이 부분이 모두 진짜"
        " 손실이라고 할 수는 없어요. 저율 곡선이라면 흔히 리튬 손실은 두 전극이 맞물리는 위치가 어긋나 **곡선 한쪽 끝의 단계가"
        " 짧아지는** 모양으로, 활물질 손실은 그 전극이 맡은 **평탄 구간들이 함께 줄어드는** 모양으로 읽어요.",
        f"- **내부저항은 가설 A를 직접 뒷받침하지 않아요.** 측정된 내부저항 변화는 작고 ΔQ와의 상관도 약해요 (r = {_r(a['r_ir_signal'])})."
        " 다만 이 저항은 80% SOC에서 짧은 펄스로 잰 값이라([배경 지식](background.md)), 방전 끝(낮은 SOC)의 분극과는 다를 수 있어서"
        " 가설 A를 지우지도 못해요.",
        "- **왜 4C 데이터로는 가를 수 없나:** 빠른 방전에서는 저항·발열·전극 안의 불균일 때문에 봉우리가 뭉개지고 곡선 전체가 아래로 밀려요."
        f" Attia 외(2021)도 그림 1 설명에서 4C 응답이 불균일 때문에 매끈해져서 dQ/dV·dV/dQ 진단을 적용할 수 없다고 적었어요."
        " 그래서 '용량이 줄어서 생긴 ΔQ'와 '전압이 밀려서 생긴 ΔQ'가 같은 모양으로 섞여요. 이 둘을 가르려면 같은 셀을 천천히(저율로)"
        " 잰 진단 곡선이나, 전극별 반응을 계산하는 물리 모델이 필요해요 ([배경 지식](background.md)의 dQ/dV 절, [응용 아이디어](ideas.md) 6번).",
        "",
        "## 한계",
        "",
        "- **관찰 데이터예요.** ΔQ와 수명이 '함께 나타난다'는 것까지만 말할 수 있어요. 충전 조건이 ΔQ와 수명 둘 다를 바꾸는 공통 원인일 수 있어요.",
        "- **4C 방전 하나뿐이라 리튬 손실과 활물질 손실, 분극을 가르지 못해요.** 저율 진단 데이터나 물리 시뮬레이션은 쓰지 않았어요.",
        "- **전압 1,000칸 중 하나를 고르는 건 여러 번 시도한 것과 같아요.** 학습 셀로 골랐지만, 학습 오차가"
        f" {a['train_flat'][0]:.2f}~{a['train_flat'][1]:.2f} V에서 최솟값의 {(TRAIN_FLAT - 1) * 100:.0f}% 안으로 거의 평평해서"
        f" 맞추는 방법이 조금만 달라도 고른 전압이 바뀌어요 (같은 학습 셀로 우리 보통 직선은 {b['voltage']:.3f} V, 논문 Elastic Net은"
        f" {ATTIA_TRAIN_BEST_V} V). 셀 다시 뽑기에서도 가장 강한 전압이 움직여서, {it.BOOT:,}번 중 90%가"
        f" {a['boot_p95']:.2f}~{a['boot_p5']:.2f} V에 흩어져 있었어요."
        " 그래서 '정확히 몇 V'보다 '봉우리 아래 구간'으로 읽어야 해요.",
        "- **dQ/dV 배열은 데이터에 들어 있는 값을 그대로 썼어요.** 이 값을 전압으로 더해도 Qdlin의 용량 차이와 정확히 같지는 않아서"
        " (데이터 쪽에서 매끄럽게 다듬은 것으로 보여요), 용량이 어디서 줄었는지는 Qdlin에서 직접 계산한 ΔQ를 미분해 봤어요.",
        "- Qdlin은 방전의 정전류 구간만 담아요. 2.0V에 닿은 뒤의 용량은 ΔQ에 들어 있지 않아요.",
        "- 셀 겉면 온도는 신뢰도가 낮아 쓰지 않았어요. 발열 차이가 곡선 모양에 주는 영향은 따로 떼어 보지 못했어요.",
        "- 한 종류의 셀(LFP/흑연 18650), 한 온도(30°C)라서 다른 셀로 그대로 넓히기 어려워요.",
        "",
        "## 시행착오",
        "",
    ]
    lines += [f"- {x}" for x in mistakes] if mistakes else ["- 이번 작업에서 따로 기록할 실수는 없었어요."]
    lines += ["", "더 해 볼 만한 응용은 [응용 아이디어](ideas.md)에 정리해 두었어요.", ""]
    return "\n".join(lines)


def mistakes(a: dict) -> list[str]:
    """이번 작업에서 실제로 있었던 실수. 문서의 '시행착오' 절에 들어가고, 숫자는 다시 계산한 값을 쓴다."""
    odd = ", ".join(a["full_range_peak_cells"]) or "없음"
    return [
        "**dQ/dV 봉우리를 2.0V 끝에서 찾음:** 처음엔 −dQ/dV가 가장 큰 곳을 전압 전체(3.5~2.0V)에서 찾았는데, 셀마다 봉우리 위치 범위를"
        f" 훑어보니 가장 낮은 값이 2.0V였어요. 그렇게 찾으면 봉우리가 {it.PEAK_WINDOW[0]}~{it.PEAK_WINDOW[1]} V 밖으로 나오는 셀: {odd}."
        " 방전 끝(2.0V)에서 dQ/dV 값이 튀는 셀이 있어서예요. 그 상태로 계산한 봉우리 높이 변화와 수명의 상관"
        f"(r = {_r(a['r_height_life_full'])})은 '관계없음'처럼 보였는데, 봉우리는 {it.PEAK_WINDOW[0]}~{it.PEAK_WINDOW[1]} V 안에서만"
        f" 찾도록 바꾸자 상관이 달라졌어요 (r = {_r(a['r_height_life'])}). → 봉우리·최댓값을 찾을 때는 먼저 셀 전체의 결과 범위를 확인해요.",
        "**dQ/dV 배열을 ΔQ의 미분과 같다고 짐작함:** 처음엔 '용량이 어느 전압에서 줄었나'를 데이터의 dQ/dV 배열 차이(100번째 − 10번째)로"
        " 그리려 했어요. 확인 삼아 그 차이의 셀 평균을 전압으로 더해 보니"
        f" {a['dqdv_integral_mah']:.1f} mAh로, Qdlin에서 바로 구한 2.0V의 평균 ΔQ({a['all_depth']['end_mah']:.1f} mAh)와 맞지 않았어요."
        " 그래서 '수명 무리별 ΔQ' 그림의 ②는 Qdlin의 ΔQ를 직접 미분해 만들고, dQ/dV 배열은 10번째 사이클의 봉우리 위치를 보는 데만 썼어요."
        " → 미리 계산된 값은 다른 값과 서로 맞는지 더해 보고 써요.",
        "**근거가 되지 못하는 숫자를 이유로 댐 (검증 단계에서 고침):** 첫 원고는 '학습 셀 41개로 고른 전압은 흔들린다'의 예로"
        f" 우리 {a['models']['best_log']['voltage']:.3f} V와 논문 {ATTIA_TRAIN_BEST_V} V를 들었어요. 그런데 두 값은 같은 학습 셀에"
        " 다른 방법(보통 직선 vs Elastic Net)을 쓴 결과라, 셀 수 때문에 흔들린다는 근거가 못 돼요. 학습 오차가"
        f" {a['train_flat'][0]:.2f}~{a['train_flat'][1]:.2f} V에서 거의 평평하다는 계산과 셀 다시 뽑기 결과로 근거를 바꿨어요."
        " 같은 검증에서 '직선이 아니라서예요'처럼 단정한 원인 문장도 '~로 보여요'로 낮췄어요."
        " → 숫자를 이유로 들 때는 그 숫자가 정말 그 이유를 보여 주는지 확인해요.",
    ]


def build(processed: Path = PROCESSED, docs: Path = DOCS, progress=print) -> dict:
    progress("곡선 모으기 (ΔQ₁₀₀₋₁₀, dQ/dV)")
    data = it.collect(processed)
    progress(f"전압별 상관·단일 모델·흔들림 ({it.BOOT:,}번)")
    a = analyze(data)

    spec = a["spec"].copy()
    for kind in it.TRANSFORMS:
        t = a["uni"][kind]
        for key in it.SPLIT_KEYS:
            spec[f"rmse_{kind}_{key}"] = t[f"rmse_{key}"].to_numpy()
    spec["neg_dqdv10_mean"] = a["mean_dqdv10"]
    for g, f in a["flows"].items():
        spec[f"group{g}_dq_mean_mah"] = a["group_curves"][g]["mean"] * 1000
        spec[f"group{g}_moved"] = f
    spec.to_csv(processed / "interpret_spectrum.csv", index=False, encoding="utf-8")
    a["per_cell"].to_csv(processed / "interpret_cells.csv", index=False, encoding="utf-8")
    a["groups"].to_csv(processed / "interpret_groups.csv", index=False, encoding="utf-8")

    progress("그래프")
    fig_dir = docs / "figures"
    spectrum_figure(a, fig_dir / "interpret_spectrum.png")
    groups_figure(a, fig_dir / "interpret_groups.png")
    write_doc(docs / "interpret.md", markdown(a, mistakes(a)))
    return {
        "cells": a["cells"], "skipped": a["skipped"],
        "strongest_v_log": a["strong"]["log"]["voltage"], "strongest_r_log": a["strong"]["log"]["r"],
        "strongest_v_raw": a["strong"]["raw"]["voltage"], "strongest_r_raw": a["strong"]["raw"]["r"],
        "band_log": a["bands"]["log"], "boot_90": (a["boot_p95"], a["boot_p5"]),
        "variance_r": a["var"]["r"],
        "variance_rmse": {k: a["var"][f"rmse_{k}"] for k in it.SPLIT_KEYS},
        "single_v": a["models"]["best_log"]["voltage"],
        "single_rmse": {k: a["models"]["best_log"][f"rmse_{k}"] for k in it.SPLIT_KEYS},
        "attia_v_rmse": {k: a["models"]["attia"][f"rmse_{k}"] for k in it.SPLIT_KEYS},
        "peak_v": a["peak_v"], "deepest_v": a["all_depth"]["voltage"], "recovered": a["all_depth"]["recovered"],
        "r_logvar_signal": a["r_logvar_signal"], "r_ir_signal": a["r_ir_signal"],
        "groups": a["groups"],
    }


if __name__ == "__main__":
    build()
