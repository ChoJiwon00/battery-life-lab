"""5사이클 선별(분류) 실험 전체를 돌려 docs/screening.md와 그래프를 만든다.

    .venv\\Scripts\\python.exe -m battery_lab.screening_report

문서의 숫자는 모두 이 파일이 계산한다. 논문 숫자는 직접 열어 확인한 출처(PAPER)에 있는 것만 쓴다.
"""

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mdtext import write_doc
from . import screening as sc
from .cycles import minutes_per_cycle
from .figures import AXIS, BLUES, INK, INK_2, MUTED, SURFACE, _save, _style, _subtitle
from .korean import ieotda
from .sources import PROCESSED, ROOT

DOCS = ROOT / "docs"

# 논문 숫자는 기억이 아니라 직접 연 출처로 확인한 것만 쓴다 (troubleshooting.md 16번과 같은 원칙).
PAPER = {
    "pdf": "https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf",  # 저자 연구실이 올린 논문 PDF
    "si": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41560-019-0356-8/MediaObjects/"
          "41560_2019_356_MOESM1_ESM.pdf",  # 보충 자료 (Supplementary Information)
    "threshold": 550,                       # 본문 Table 2, 보충 자료 Supplementary Note 4
    "test_error": 4.9,                      # 초록·결론: 첫 5사이클, 두 그룹 분류의 테스트 오차
    "variance": {"train": 82.1, "test": 78.6, "secondary_test": 97.5},  # Table 2 정확도(%)
    "full": {"train": 97.4, "test": 92.7, "secondary_test": 97.5},
    "variance_test_all": 88.8, "full_test_all": 95.1,                     # 본문: 테스트 정확도
    "cells": {"train": 39, "test": 41, "secondary_test": 40},             # 보충 자료 Table 4·6
    "full_selected": "온도 적분, 충전 시간, min ΔQ₅₋₄, var ΔQ₅₋₄",          # 보충 자료 Table 5 (18개 중 4개)
}
NEAR = 0.95  # 기준 수명의 95% 이상이면 '기준 바로 아래'로 본다
SPLIT_NAMES = {"train": "학습", "test": "테스트", "secondary_test": "2차 테스트", "test_all": "테스트 합계"}
KIND_COLORS = {"variance": "#2a78d6", "full": "#eb6834", "cell": "#1baf7a"}  # figures.py의 검증된 세 색, 고정 순서
BATCH_COLORS = {"b1": "#2a78d6", "b2": "#eb6834", "b3": "#1baf7a"}
FEATURE_NAMES = {"logvar": "log 분산 ΔQ", "logmin": "ΔQ 최솟값 크기(log)", "q2": "2번째 용량", "qmax_q2": "용량 증가분",
                 "ir2": "2번째 내부저항", "ir_min": "내부저항 최솟값", "ir_diff": "내부저항 변화",
                 "chargetime": "충전 시간", "tavg": "평균 온도", "tmax": "최고 온도", "tmin": "최저 온도"}


# ---------------------------------------------------------------- 실행

def build(processed: Path = PROCESSED, docs: Path = DOCS, repeats: int = sc.REPEATS, shuffles: int = 2000,
          progress=print) -> dict:
    """전체 실험 → data/processed/screening_*.csv, docs/figures/screening_*.png, docs/screening.md. 핵심 숫자를 돌려준다.

    repeats: 무작위 나누기 횟수, shuffles: 같은 조건 안 짝 비교에서 수명을 섞어 보는 횟수 (테스트에서는 줄여 쓴다).
    """
    progress("특징 계산 (5·10·20·50·100사이클)")
    feats = sc.build_features(processed)
    pd.concat([f.assign(cycles=n) for n, f in feats.items()]).to_csv(processed / "screening_features.csv", index=False)
    f5 = feats[5]

    progress("논문 나누기")
    paper = {n: sc.paper_split(f) for n, f in feats.items()}
    cut, op, op_p = sc.operating_point(f5)
    coef = sc.coefficients(sc.OPERATING_KIND, f5[f5["split"] == "train"],
                           sc.labels(f5.loc[f5["split"] == "train", "cycle_life"]))
    paper[5][1].assign(op_cut=cut).to_csv(processed / "screening_paper.csv", index=False)

    progress(f"무작위 나누기 {repeats}번 (기준 {sc.THRESHOLD}사이클)")
    runs, curves = sc.random_splits(feats, repeats=repeats, progress=progress)
    progress(f"무작위 나누기 {repeats}번 (기준 {sc.ALT_THRESHOLD}사이클, 5사이클만)")
    runs_alt, _ = sc.random_splits({5: f5}, threshold=sc.ALT_THRESHOLD, repeats=repeats, cut_kind=None)
    pd.concat([runs.assign(threshold=sc.THRESHOLD), runs_alt.assign(threshold=sc.ALT_THRESHOLD)]) \
        .to_csv(processed / "screening_runs.csv", index=False)
    summ, summ_alt, trade = sc.summarize_runs(runs), sc.summarize_runs(runs_alt), sc.summarize_tradeoff(curves)
    trade.to_csv(processed / "screening_tradeoff.csv", index=False)

    progress("같은 충전 조건 안에서 짝 비교")
    within = sc.within_condition(feats, repeats=shuffles)
    within.to_csv(processed / "screening_within.csv", index=False)
    corr = {n: float(np.corrcoef(f["logvar"], np.log10(f["cycle_life"]))[0, 1]) for n, f in feats.items()}
    minutes = minutes_per_cycle(processed)

    progress("그래프")
    fig = docs / "figures"
    signal_figure(f5, fig / "screening_signal.png")
    cycles_figure(summ, minutes, repeats, fig / "screening_cycles.png")
    tradeoff_figure(trade, summ, repeats, fig / "screening_tradeoff.png")
    confusion_figure(op, cut, fig / "screening_confusion.png")

    old = first_cut_rule_replay(f5, op_p)
    ctx = {"feats": feats, "paper": paper, "cut": cut, "op": op, "op_p": op_p, "coef": coef, "summ": summ, "old": old,
           "summ_alt": summ_alt, "trade": trade, "within": within, "corr": corr, "minutes": minutes,
           "repeats": repeats, "shuffles": shuffles, "subs": sc.substitutions(feats)}
    write_doc(docs / "screening.md", markdown(ctx))
    return key_numbers(ctx)


def first_cut_rule_replay(f: pd.DataFrame, p: np.ndarray, max_miss: float = 0.10) -> dict:
    """시행착오 기록용: 처음 썼다가 버린 운영점 규칙('학습 셀 놓침 10% 이하인 가장 높은 기준값')을 다시 계산한다."""
    y = sc.labels(f["cycle_life"])
    tr, te = (f["split"] == "train").to_numpy(), (f["split"] == "test").to_numpy()
    oof = sc.oof_proba(sc.OPERATING_KIND, f[tr], y[tr])
    cut = float(max(c for c in sc.CUTS if np.mean(oof[y[tr] == 1] < c) <= max_miss + 1e-12))
    comp = f[tr].assign(short=y[tr]).groupby("batch")["short"].agg(["size", "sum"])
    return {"cut": cut, "short": int(y[te].sum()), "missed": sc.confusion(y[te], p[te], cut)["missed"],
            "missed_default": sc.confusion(y[te], p[te], 0.5)["missed"],
            "train": {b: (int(r["size"]), int(r["sum"])) for b, r in comp.iterrows()}}


def _row(summ: pd.DataFrame, n: int, kind: str) -> pd.Series:
    return summ[(summ["cycles"] == n) & (summ["kind"] == kind)].iloc[0]


def _paper_row(paper: dict, n: int, kind: str, split: str) -> pd.Series:
    m = paper[n][0]
    return m[(m["kind"] == kind) & (m["split"] == split)].iloc[0]


def key_numbers(ctx: dict) -> dict:
    s = ctx["summ"]
    return {
        "threshold": sc.THRESHOLD,
        "short_cells": int(sc.labels(ctx["feats"][5]["cycle_life"]).sum()),
        "short_in_b2": int(sc.batch_table(ctx["feats"][5]).set_index("batch").loc["b2", "short"]),
        "paper_split_5": {k: {sp: float(_paper_row(ctx["paper"], 5, k, sp)["accuracy"])
                              for sp in (*sc.SPLIT_ORDER, "test_all")} for k in sc.KINDS},
        "random_5": {k: float(_row(s, 5, k)["accuracy_median"]) for k in (*sc.KINDS, "majority", "batch")},
        "random_by_cycles": {n: {k: float(_row(s, n, k)["accuracy_median"]) for k in sc.KINDS} for n in sc.CYCLES},
        "operating_cut_paper": ctx["cut"],
        "operating_random": {c: float(_row(s, 5, f"{sc.OPERATING_KIND}@op")[f"{c}_median"])
                             for c in ("cut", "miss_rate", "discard_rate", "accuracy")},
        "within_concordance": dict(zip(ctx["within"]["cycles"], ctx["within"]["concordance"])),
        "logvar_corr": ctx["corr"],
        "minutes_per_cycle": ctx["minutes"],
    }


# ---------------------------------------------------------------- 그래프

def signal_figure(f: pd.DataFrame, path: Path) -> Path:
    """5사이클 신호(log 분산 ΔQ₅₋₄)와 수명. 배치마다 색을 달리해 '단수명 = 거의 2배치'를 보여 준다."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    table = sc.batch_table(f).set_index("batch")
    for b, color in BATCH_COLORS.items():
        g = f[f["batch"] == b]
        ax.scatter(g["logvar"], g["cycle_life"], s=34, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   label=f"{b[1]}배치 ({len(g)}개 중 단수명 {int(table.loc[b, 'short'])}개)")
    ax.axhline(sc.THRESHOLD, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(f["logvar"].max(), sc.THRESHOLD * 1.03, f"기준 {sc.THRESHOLD}사이클 (이보다 짧으면 단수명)", ha="right",
            va="bottom", color=INK_2, fontsize=9)
    ax.set_yscale("log")
    ax.set_yticks([150, 200, 300, 500, 1000, 2000], labels=["150", "200", "300", "500", "1,000", "2,000"])
    ax.minorticks_off()
    ax.set(xlabel="log₁₀ Var(ΔQ₅₋₄)  (5번째 − 4번째 방전 곡선 차이의 분산)", ylabel="사이클 수명")
    short = int(table["short"].sum())
    r = float(np.corrcoef(f["logvar"], np.log10(f["cycle_life"]))[0, 1])
    ax.set_title("5사이클 신호와 수명: 단수명 셀은 거의 한 배치에 몰려 있어요")
    _subtitle(ax, f"단수명 셀 {short}개 중 {int(table.loc['b2', 'short'])}개가 2배치 · 로그 수명과의 상관 r = {r:.2f} "
                  f"({len(f)}개 셀)")
    leg = ax.legend(frameon=False, loc="upper right", labelcolor=INK_2)
    for h in leg.legend_handles:
        h.set_sizes([40])
    return _save(fig, path)


def cycles_figure(summ: pd.DataFrame, minutes: float, repeats: int, path: Path) -> Path:
    """사이클 수 n마다 무작위 나누기 정확도 (중앙값과 10~90%), 비교 기준은 점선."""
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = np.arange(len(sc.CYCLES))
    for offset, kind in zip((-0.16, 0.0, 0.16), sc.KINDS):
        rows = [_row(summ, n, kind) for n in sc.CYCLES]
        med = np.array([r["accuracy_median"] for r in rows])
        lo = med - np.array([r["accuracy_p10"] for r in rows])
        hi = np.array([r["accuracy_p90"] for r in rows]) - med
        ax.errorbar(xs + offset, med, yerr=[lo, hi], fmt="o", ms=6.5, color=KIND_COLORS[kind], elinewidth=1.5,
                    capsize=3, label=sc.NAMES[kind], zorder=3)
        ax.plot(xs + offset, med, color=KIND_COLORS[kind], lw=1, alpha=0.5)
    for kind, va in (("batch", "bottom"), ("majority", "bottom")):
        v = float(_row(summ, sc.CYCLES[0], kind)["accuracy_median"])
        ax.axhline(v, color=INK_2 if kind == "batch" else MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(xs[-1] + 0.32, v + 0.4, f"{sc.NAMES[kind]} {v:.1f}%", ha="right", va=va, fontsize=8.5,
                color=INK_2 if kind == "batch" else MUTED)
    labels = [f"{n}사이클\n(약 {n * minutes / 60:.0f}시간)" if n * minutes < 48 * 60 else
              f"{n}사이클\n(약 {n * minutes / 1440:.1f}일)" for n in sc.CYCLES]
    ax.set_xticks(xs, labels=labels)
    ax.set_xlim(-0.5, xs[-1] + 0.45)
    ax.set_ylim(55, 101)
    ax.set_ylabel("정확도 (%)")
    ax.set_title(f"몇 사이클을 보면 얼마나 맞힐까 ({sc.THRESHOLD}사이클 기준)")
    _subtitle(ax, f"무작위 나누기 {repeats}번 (학습 41 · 평가 83) · 점 = 중앙값 · 막대 = 10~90% · 점선 = 셀 측정 없이 찍기")
    ax.legend(frameon=False, loc="lower left", labelcolor=INK_2, fontsize=8.5, ncol=3)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def tradeoff_figure(trade: pd.DataFrame, summ: pd.DataFrame, repeats: int, path: Path) -> Path:
    """기준값(P(단수명) 이상이면 뺀다)에 따른 놓친 단수명·버린 장수명 비율."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = trade["cut"].to_numpy()
    lines = (("miss", "놓친 단수명 셀 (출하됨)", "#eb6834"), ("discard", "버린 장수명 셀 (멀쩡한데 뺌)", "#2a78d6"))
    for key, label, color in lines:
        ax.fill_between(x, trade[f"{key}_p10"], trade[f"{key}_p90"], color=color, alpha=0.15, lw=0)
        ax.plot(x, trade[f"{key}_median"], color=color, lw=2, label=label)
    op = _row(summ, 5, f"{sc.OPERATING_KIND}@op")
    marks = ((0.5, "기본 0.5", 96), (float(op["cut_median"]), f"운영점 {op['cut_median']:.2f}\n(학습 셀로 정한 값)", 60))
    for c, name, y_text in marks:
        ax.axvline(c, color=INK_2, lw=1, ls=(0, (4, 3)))
        ax.text(c + 0.012, y_text, name, ha="left", va="top", fontsize=8.5, color=INK_2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.9})
    ax.set(xlim=(0, 1), ylim=(0, 100), xlabel="기준값: 단수명일 확률이 이 값 이상이면 '불량'으로 빼요",
           ylabel="비율 (%)")
    ax.set_title("기준값을 낮추면 덜 놓치지만 더 버려요")
    _subtitle(ax, f"{sc.NAMES[sc.OPERATING_KIND]} · 5사이클 · 무작위 나누기 {repeats}번의 평가 셀 · 선 = 중앙값 · 띠 = 10~90%")
    # 범례를 그래프 안에 두면 0.5 점선과 10~90% 띠를 가려서, 그래프 아래로 뺀다
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, labelcolor=INK_2)
    return _save(fig, path)


def confusion_figure(op: pd.DataFrame, cut: float, path: Path) -> Path:
    """논문 나누기의 테스트·2차 테스트를 운영점 기준값으로 판정한 결과 (혼동 행렬)."""
    _style()
    parts = [op[(op["point"] == "operating") & (op["split"] == s)].iloc[0] for s in ("test", "secondary_test")]
    top = max(max(p["caught"], p["missed"], p["discarded"], p["kept"]) for p in parts)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), gridspec_kw={"wspace": 0.55})
    for ax, part, split in zip(axes, parts, ("test", "secondary_test")):
        grid = np.array([[part["caught"], part["missed"]], [part["discarded"], part["kept"]]], float)
        names = np.array([["잡아냄", "놓침"], ["버림", "통과"]])
        ax.imshow(grid / top, cmap=BLUES, vmin=-0.15, vmax=1.0)
        for i in range(2):
            for j in range(2):
                dark = grid[i, j] / top > 0.55
                ax.text(j, i, f"{names[i, j]}\n{int(grid[i, j])}개", ha="center", va="center", fontsize=12,
                        color=SURFACE if dark else INK, fontweight="bold")
        ax.set_xticks([0, 1], labels=["불량으로 뺌", "출하"])
        ax.set_xlabel("판정", color=INK_2)
        ax.set_yticks([0, 1], labels=[f"실제 단수명\n({int(part['short'])}개)",
                                      f"실제 장수명\n({int(part['cells'] - part['short'])}개)"])
        ax.tick_params(length=0, colors=INK_2)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{SPLIT_NAMES[split]} {int(part['cells'])}개 · 정확도 {part['accuracy']:.1f}%", fontsize=11, pad=8)
    fig.suptitle(f"논문 나누기에서 기준값 {cut:.2f}로 판정한 결과", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.885, f"{sc.NAMES[sc.OPERATING_KIND]} · 5사이클 · 기준값은 학습 셀 41개만 보고 정했어요 "
                          f"(놓침 1개 = 버림 {sc.MISS_COST}개 손해로 가정)", color=INK_2, fontsize=9.5)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.14, top=0.74)
    return _save(fig, path)


# ---------------------------------------------------------------- 문서

def _rng(r: pd.Series, col: str, digits: int = 1) -> str:
    return f"{r[f'{col}_median']:.{digits}f}% ({r[f'{col}_p10']:.{digits}f}~{r[f'{col}_p90']:.{digits}f})"


def _time(n: int, minutes: float) -> str:
    return f"약 {n * minutes / 60:.0f}시간" if n * minutes < 48 * 60 else f"약 {n * minutes / 1440:.1f}일"


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def _fmt_feature(name: str, v: float) -> str:
    if name.startswith("ir"):
        return f"{v * 1000:.2f}"   # Ω → mΩ
    if name in ("q2", "qmax_q2"):
        return f"{v:.4f}"
    return f"{v:.2f}"


def markdown(ctx: dict) -> str:
    feats, paper, summ, minutes = ctx["feats"], ctx["paper"], ctx["summ"], ctx["minutes"]
    f5 = feats[5]
    table = sc.batch_table(f5)
    tb = table.set_index("batch")
    short_total, short_b2 = int(table["short"].sum()), int(tb.loc["b2", "short"])
    y = sc.labels(f5["cycle_life"])
    batch_errors = int(sum(min(r.short, r.cells - r.short) for r in table.itertuples()))
    sec = f5[f5["split"] == "secondary_test"]
    sec_long = float((sc.labels(sec["cycle_life"]) == 0).mean() * 100)
    r5 = {k: _row(summ, 5, k) for k in (*sc.KINDS, "majority", "batch")}
    op_r = _row(summ, 5, f"{sc.OPERATING_KIND}@op")
    base_r = r5[sc.OPERATING_KIND]
    best5 = max(sc.KINDS, key=lambda k: r5[k]["accuracy_median"])
    acc = {(n, k): float(_row(summ, n, k)["accuracy_median"]) for n in sc.CYCLES for k in sc.KINDS}
    worst_n = min(sc.CYCLES, key=lambda n: acc[(n, "variance")])
    corr = ctx["corr"]
    w = ctx["within"].set_index("cycles")
    shared = sc.same_policy_across_batches(f5)
    subs = ctx["subs"]
    used = [k for k, v in ctx["coef"].items() if v != 0]
    med = f5.groupby("batch")[used].median()

    lines = [
        "# 5사이클로 수명이 짧을 셀 골라내기 (분류)",
        "",
        f"`.venv\\Scripts\\python.exe -m battery_lab.screening_report`가 자동으로 만든 문서예요 (만든 날 {date.today().isoformat()}).",
        "",
        "## 질문",
        "",
        f"첫 **5사이클**만 보고 \"수명이 {sc.THRESHOLD}사이클도 안 될 셀\"(단수명 셀)을 골라낼 수 있을까요?",
        "그리고 10·20·50·100사이클까지 기다리면 얼마나 더 잘 맞힐까요?",
        "",
        f"논문(Severson 외, 2019)은 첫 5사이클로 셀을 두 그룹으로 나눠 테스트 오차를 **{PAPER['test_error']}%로** 보고했어요.",
        f"두 그룹을 가르는 기준 수명은 [저자가 올린 논문 PDF]({PAPER['pdf']})의 Table 2와",
        f"[보충 자료]({PAPER['si']})의 Supplementary Note 4에서 직접 확인했어요 → **{PAPER['threshold']}사이클**.",
        "",
        "## 왜 중요한가 (셀 개발·공정 개발 직무와 연결)",
        "",
        f"- 이 데이터에서 사이클 하나는 중앙값 **{minutes:.0f}분**이에요(원시 기록이 있는 100사이클 이내 사이클로 측정). "
        f"5사이클은 {_time(5, minutes)}, 100사이클은 {_time(100, minutes)}이에요. 수명이 긴 셀은 끝까지 시험하는 데 몇 달이 걸려요.",
        "- **공정·품질:** 셀을 만든 뒤 화성(첫 충방전으로 전극 표면 막을 만드는 공정)·에이징(보관하며 안정시키는 공정) 직후의",
        "  아주 짧은 데이터로 '오래 못 갈 셀'을 미리 빼면, 출하 뒤 불량과 긴 검사 시간을 줄일 수 있어요.",
        "  이때 핵심은 정확도 하나가 아니라 **놓친 불량(출하된 단수명 셀)과 버린 정상 셀 사이의 균형**이에요.",
        "- **셀 개발:** 새 소재·충전 조건을 평가할 때 몇 시간 만에 '가망 없는 후보'를 거를 수 있다면 시험 기간을 크게 줄일 수 있어요."
        " 이 데이터에서 정말 그게 되는지가 이 문서의 질문이에요.",
        "- **주의할 점:** 이 데이터는 공장 셀 선별 데이터가 아니라 **충전 조건을 일부러 바꿔 가며 시험한 연구 데이터**예요."
        " 그래서 '셀 하나하나의 차이'를 읽은 것인지, '어느 시험 묶음·조건이었는지'를 읽은 것인지 따로 확인해야 해요(결과 1·6).",
        "",
        "## 방법",
        "",
        f"- **나누는 기준:** 수명 < {sc.THRESHOLD}사이클이면 단수명, 아니면 장수명. 124개 중 단수명은 **{short_total}개**예요."
        f" 기준을 {sc.ALT_THRESHOLD}사이클로 바꾼 결과도 함께 봤어요(논문 보충 자료도 {sc.ALT_THRESHOLD}을 함께 봤어요).",
        "- **특징 (n사이클까지의 데이터만, n보다 뒤는 절대 안 써요)** — [screening.py](../battery_lab/screening.py)",
        "",
        "| 특징 | 뜻 |",
        "|---|---|",
        "| log 분산 ΔQ | log₁₀ Var(ΔQ(V)). ΔQ = n번째 − 기준 사이클의 방전 곡선(전압마다 방전 용량). 기준은 n ≤ 10이면 바로 앞 사이클(5사이클이면 ΔQ₅₋₄, 논문과 같음), n > 10이면 10번째 |",
        "| log \\|최솟값 ΔQ\\| | ΔQ 곡선의 가장 낮은 값의 크기 (log) |",
        "| 2번째 용량 · 용량 증가분 | 2번째 사이클 방전 용량, 2~n사이클 최대 용량 − 2번째 용량 (1배치는 1번째 사이클 기록이 없어 모두 2번째부터) |",
        "| 내부저항 3개 | 2번째 사이클 값, 2~n 중 최솟값, n번째 − 2번째 |",
        "| 충전 시간 | 2~5사이클 평균. 이 데이터에서는 **충전 조건(실험 설정)을 거의 그대로 담아요** |",
        "| 온도 3개 | 2~n사이클 캔 표면 온도의 평균·최고·최저. 공식 설명상 **믿기 어려운 측정**(센서가 떨어진 셀이 있음)이에요 |",
        "",
        f"  두 곡선 중 하나라도 깨끗하지 않으면(품질 점검 기준) 두 곡선을 함께 1~2사이클 **앞으로** 옮겨 써요."
        + (f" 이번 데이터에서는 그럴 일이 없었어요(옮긴 곳 0곳, 비어 있는 특징 0개)." if subs.empty else
           f" 옮기거나 비어 있던 곳: {len(subs)}곳 (`data/processed/screening_features.csv`)."),
        "- **모델 (로지스틱 회귀 = '단수명일 확률'을 내는 직선 모델, 특징은 표준화 = 평균 0·표준편차 1로 맞춤)**",
        f"  - **{sc.NAMES['variance']}:** 논문의 'variance classifier'와 같은 형태. 약한 L2 규제(계수가 너무 커지지 않게 누름).",
        f"  - **{sc.NAMES['cell']}:** ΔQ 2개 + 용량 2개 + 내부저항 3개. L1 규제(쓸모없는 특징의 계수를 0으로 만듦), 세기는 학습 셀 안 5겹 교차 검증으로 골라요.",
        f"  - **{sc.NAMES['full']}:** 위 7개 + 충전 시간 + 온도 3개. 논문의 'full classifier'(L1 규제, 후보 18개 중 4개 선택)를 참고했지만"
        " 특징이 똑같지는 않아요. 특히 논문이 가장 크게 쓴 '온도 적분(시간에 대해 온도를 더한 값)'은 사이클별 시간 기록이 없어 평균 온도로 대신했어요.",
        "- **비교 기준 (셀을 측정하지 않고 찍기)**",
        f"  - **{sc.NAMES['majority']}:** 학습 셀에서 더 많은 쪽으로 모두 판정",
        f"  - **{sc.NAMES['batch']}:** 배치(1·2·3배치 = 시험 시기가 다른 세 묶음)마다 학습 셀에서 더 많은 쪽으로 판정. 셀은 전혀 보지 않아요.",
        "- **평가**",
        "  1. 논문 나누기: 학습 41 · 테스트 43 · 2차 테스트 40 (논문 보충 자료는 39 · 41 · 40개를 썼다고 적었어요. 4개 차이의 이유는 확인하지 못했어요."
        " 보충 자료 Table 4·6의 칸을 직접 더하면 학습 셀이 38개로 나와서, 논문 안에서도 셀 수가 조금 어긋나요).",
        f"  2. 무작위 나누기: 124개를 학습 41 / 평가 83으로 {ctx['repeats']}번 나눠요. 단수명 비율은 매번 같게 맞추고, "
        "**반복마다 같은 나누기를 모든 사이클 수·모델에 써서** 차이가 나누기 운이 아니라 사이클 수·모델 차이에서 오게 했어요 (고정 시드 2026).",
        f"- **운영점(실제로 쓸 기준값) 정하기:** 기본은 '단수명일 확률 ≥ 0.5면 뺀다'예요. 공정에서는 놓친 불량이 더 비싸서, "
        f"**놓친 단수명 1개 = 버린 장수명 {sc.MISS_COST}개 손해**라고 가정했어요(우리가 정한 값). 학습 셀만으로(5겹 교차 검증 확률) "
        "손해가 가장 작은 기준값을 고르고, 평가 셀에는 그 값을 그대로 써요.",
        f"  운영점 모델은 **{sc.NAMES[sc.OPERATING_KIND]}** 모델로 정했어요. 공장에서는 모든 셀을 같은 조건으로 충전하니 "
        "'충전 조건 차이'를 담은 충전 시간 특징은 쓸 수 없고, 온도는 믿기 어려운 측정이라서예요.",
        "",
        "## 결과",
        "",
        f"### 1. 먼저 볼 것: 단수명 셀 {short_total}개 중 {short_b2}개가 2배치예요",
        "",
        "| 배치 | 셀 | 단수명 셀 | 수명 최소 | 수명 중앙값 | 수명 최대 |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        lines.append(f"| {r.batch} | {r.cells} | **{r.short}** | {r.life_min:,.0f} | {r.life_median:,.0f} | {r.life_max:,.0f} |")
    lines += [
        "",
        f"→ {sc.THRESHOLD}사이클 기준에서는 '단수명인가'가 거의 '2배치인가'와 같아요. 셀을 전혀 측정하지 않고 "
        f"\"2배치면 단수명\"이라고만 해도 124개 중 **{batch_errors}개만 틀려요** ({(1 - batch_errors / len(f5)) * 100:.1f}%).",
        "배치는 시험 시기뿐 아니라 시험한 충전 조건의 종류도 달라서, 이 차이가 셀 때문인지 시험 환경·조건 때문인지 이 데이터만으로는 가를 수 없어요"
        " ([충전 조건 실험](policies.md)에서도 같은 함정을 다뤘어요).",
        "",
    ]
    if not shared.empty:
        lines += ["충전 조건 이름이 똑같은데 배치만 다른 셀들:", "", "| 충전 조건 | 배치 | 셀 | 수명 범위 |", "|---|---|---|---|"]
        for r in shared.itertuples():
            rng = f"{r.life_min:,.0f}" if r.life_min == r.life_max else f"{r.life_min:,.0f} ~ {r.life_max:,.0f}"
            lines.append(f"| `{r.policy}` | {r.batch} | {r.cells} | {rng} |")
        lines += ["", "→ 같은 조건 이름에서도 2배치 쪽 수명이 같거나 짧게 관찰돼요. 원인(시험 환경, 셀 생산 묶음 등)은 이 데이터로는 알 수 없는 가설이에요.", ""]
    lines += [
        "![5사이클 신호와 수명](figures/screening_signal.png)",
        "",
        "### 2. 논문 나누기 (5사이클, 기준값 0.5)",
        "",
        "| 모델 | 학습 | 테스트 | 2차 테스트 | 테스트 합계 |",
        "|---|---|---|---|---|",
    ]
    for kind in sc.KINDS:
        cells = [_pct(_paper_row(paper, 5, kind, sp)["accuracy"]) for sp in (*sc.SPLIT_ORDER, "test_all")]
        lines.append(f"| {sc.NAMES[kind]} | " + " | ".join(cells) + " |")
    long_share = []
    for sp in (*sc.SPLIT_ORDER, "test_all"):
        r = _paper_row(paper, 5, "variance", sp)
        long_share.append(_pct((r["cells"] - r["short"]) / r["cells"] * 100))
    lines.append("| 참고: 모두 장수명이라고 찍기 | " + " | ".join(long_share) + " |")
    lines += [
        f"| 논문 variance classifier (셀 {PAPER['cells']['train']}·{PAPER['cells']['test']}·{PAPER['cells']['secondary_test']}개) | "
        + " | ".join(_pct(PAPER["variance"][sp]) for sp in sc.SPLIT_ORDER) + f" | {_pct(PAPER['variance_test_all'])} |",
        f"| 논문 full classifier | " + " | ".join(_pct(PAPER["full"][sp]) for sp in sc.SPLIT_ORDER)
        + f" | {_pct(PAPER['full_test_all'])} |",
        "",
        f"*논문 숫자 출처: [논문 PDF]({PAPER['pdf']}) Table 2와 본문, 셀 수는 [보충 자료]({PAPER['si']}) Table 4·6. "
        f"'테스트 합계' 칸의 {PAPER['variance_test_all']}%·{PAPER['full_test_all']}%는 본문이 'test classification accuracy'라고만 쓴 값이에요. "
        f"full classifier는 보충 자료 Table 6에서 테스트·2차 테스트를 합쳐 세면 81개 중 4개가 틀려 {PAPER['full_test_all']}%가 되므로, "
        f"논문의 테스트 오차 {PAPER['test_error']}%(= 100 − {PAPER['full_test_all']})는 두 테스트를 합친 값으로 보여요.*",
        "",
    ]
    v_test, f_all = _paper_row(paper, 5, "variance", "test")["accuracy"], _paper_row(paper, 5, "full", "test_all")["accuracy"]
    lines += [
        f"→ 우리 '{sc.NAMES['variance']}'의 테스트 정확도는 {v_test:.1f}%(논문 {PAPER['variance']['test']}%)예요. "
        f"'{sc.NAMES['full']}'의 테스트 합계 정확도는 {f_all:.1f}%로, 오차로는 {100 - f_all:.1f}%예요. "
        f"셀 수(41·43·40 vs 39·41·40)와 특징(온도 적분 대신 평균 온도 등)이 달라서 **논문의 {PAPER['test_error']}%를 재현했다고 말하지 않아요.**",
        f"→ 2차 테스트(3배치)는 40개 중 {int(sec['cycle_life'].lt(sc.THRESHOLD).sum())}개만 단수명이라, "
        f"**모두 장수명이라고 찍어도 {sec_long:.1f}%예요.** 논문의 2차 테스트 정확도 {PAPER['full']['secondary_test']}%도 이 값과 같고, "
        "보충 자료 Table 4·6의 혼동 행렬을 보면 논문의 두 모델도 2차 테스트 40개를 모두 장수명으로 판정했어요.",
        "",
        "운영점 모델이 학습 셀에서 고른 특징 (표준화한 계수, +면 단수명 쪽, 0이면 안 씀):",
        "",
        "| 특징 | " + " | ".join(FEATURE_NAMES[k] for k in ctx["coef"].index) + " |",
        "|---|" + "---|" * len(ctx["coef"]),
        "| 계수 | " + " | ".join(f"{v:+.2f}" if v != 0 else "0" for v in ctx["coef"]) + " |",
        "",
        f"*논문 full classifier가 고른 특징 4개: {PAPER['full_selected']} (보충 자료 Table 5).*",
        "",
        "계수가 0이 아닌 특징의 배치별 중앙값 (5사이클):",
        "",
        "| 배치 | " + " | ".join(FEATURE_NAMES[k] for k in used) + " |",
        "|---|" + "---|" * len(used),
        *[f"| {b} | " + " | ".join(_fmt_feature(k, v) for k, v in row.items()) + " |" for b, row in med.iterrows()],
        "",
        "*내부저항은 mΩ(밀리옴), 용량은 Ah. 특징마다 배치끼리 수준이 다르면, 모델이 셀이 아니라 배치 차이를 배웠을 수 있어요.*",
        "",
        f"### 3. 무작위 나누기: 몇 사이클이면 될까 ({ctx['repeats']}번)",
        "",
        "| 사이클 | 시험 기간 | " + " | ".join(sc.NAMES[k] for k in sc.KINDS) + " | ΔQ 분산과 로그 수명의 상관 r |",
        "|---|---|" + "---|" * (len(sc.KINDS) + 1),
    ]
    for n in sc.CYCLES:
        cells = [_rng(_row(summ, n, k), "accuracy") for k in sc.KINDS]
        lines.append(f"| {n} | {_time(n, minutes)} | " + " | ".join(cells) + f" | {corr[n]:.2f} |")
    lines += [
        f"| 비교 | 셀 측정 없음 | {sc.NAMES['majority']} {_rng(r5['majority'], 'accuracy')} | "
        f"{sc.NAMES['batch']} {_rng(r5['batch'], 'accuracy')} | | |",
        "",
        f"*정확도 중앙값 (10~90% 범위). 10~90% 범위 = 무작위 나누기 {ctx['repeats']}번의 결과를 작은 것부터 줄 세워 "
        "아래 10%와 위 10%를 뺀 가운데 80% 구간이에요. ΔQ의 기준: 5·10사이클은 바로 앞 사이클, 20·50·100사이클은 10번째 사이클.*",
        "",
        "![사이클 수별 정확도](figures/screening_cycles.png)",
        "",
        "### 4. 놓친 불량 vs 버린 정상 셀",
        "",
        f"{sc.NAMES[sc.OPERATING_KIND]}, 5사이클, 무작위 나누기 {ctx['repeats']}번의 평가 셀 83개:",
        "",
        "| 기준값 | 놓친 단수명 셀 | 버린 장수명 셀 | 정확도 |",
        "|---|---|---|---|",
        f"| 기본 0.5 | {_rng(base_r, 'miss_rate')} | {_rng(base_r, 'discard_rate')} | {_rng(base_r, 'accuracy')} |",
        f"| 운영점 (반복마다 학습 셀로 정함, 중앙값 {op_r['cut_median']:.2f}, 10~90% {op_r['cut_p10']:.2f}~{op_r['cut_p90']:.2f}) | "
        f"{_rng(op_r, 'miss_rate')} | {_rng(op_r, 'discard_rate')} | {_rng(op_r, 'accuracy')} |",
        "",
        "![기준값에 따른 놓침과 버림](figures/screening_tradeoff.png)",
        "",
    ]
    op = ctx["op"]
    cut_text = f"**{ctx['cut']:.2f}**"
    lines += [
        f"논문 나누기에서는 학습 셀 41개로 정한 기준값이 {ieotda(cut_text)}:",
        "",
        "| 기준값 | 나누기 | 셀 | 잡은 단수명 | 놓친 단수명 | 버린 장수명 | 통과한 장수명 | 정확도 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for point in ("default", "operating"):
        for sp in ("test", "secondary_test", "test_all"):
            r = op[(op["point"] == point) & (op["split"] == sp)].iloc[0]
            lines.append(f"| {r['cut']:.2f} | {SPLIT_NAMES[sp]} | {r['cells']} | {r['caught']} | {r['missed']} | "
                         f"{r['discarded']} | {r['kept']} | {r['accuracy']:.1f}% |")
    probs = paper[5][1].assign(p=ctx["op_p"])
    held = probs[probs["split"] != "train"]
    changed = int(((held["p"] >= ctx["cut"]) != (held["p"] >= 0.5)).sum())
    extreme = int(((held["p"] < 0.1) | (held["p"] >= 0.9)).sum())
    note = (f"→ 기준값을 0.5에서 {ctx['cut']:.2f}로 바꾸니 테스트·2차 테스트 {len(held)}개 중 "
            + ("**판정이 하나도 바뀌지 않았어요.** " if changed == 0 else f"**{changed}개의 판정이 바뀌었어요.** "))
    if extreme >= 0.8 * len(held):
        note += (f"이 나누기에서는 모델 확률이 0이나 1 가까이에 몰려 있어요({len(held)}개 중 {extreme}개가 0.1 미만이거나 0.9 이상). "
                 "확률이 이렇게 극단적이면 '확신'이 아니라 학습 셀이 너무 쉽게 갈렸다는 신호일 수 있어요.")
    lines += [
        "",
        note,
        "",
        "![운영점에서의 혼동 행렬](figures/screening_confusion.png)",
        "",
    ]
    wrong = probs[(probs["split"] != "train") & ((probs["p"] >= ctx["cut"]) != (probs["short"] == 1))]
    if len(wrong):
        lines += [f"기준값 {ctx['cut']:.2f}에서 틀린 셀 (테스트·2차 테스트):", "",
                  "| 셀 | 배치 | 충전 조건 | 수명 | 단수명일 확률 | 판정 |", "|---|---|---|---|---|---|"]
        for r in wrong.sort_values("cycle_life").itertuples():
            verdict = "놓침 (단수명인데 출하)" if r.short == 1 else "버림 (장수명인데 뺌)"
            lines.append(f"| {r.cell_id} | {r.batch} | `{r.policy}` | {r.cycle_life:,.0f} | {r.p:.2f} | {verdict} |")
        missed = wrong[wrong["short"] == 1]
        near = missed[missed["cycle_life"] >= NEAR * sc.THRESHOLD]
        if len(missed) and len(near):
            text = (f"→ 놓친 단수명 {len(missed)}개 중 {len(near)}개는 수명이 기준의 {NEAR * 100:.0f}% 이상"
                    f"({', '.join(f'{v:,.0f}' for v in near['cycle_life'])}사이클)으로 기준 바로 아래예요. "
                    "기준선 근처 셀은 가르기 어려울 수 있어요.")
            # 확률이 기준값 근처가 아니라 0에 가까우면 '망설임'이 아니라 '확신하고 틀림'이에요.
            # 그 셀들이 단수명이 몰린 배치가 아니라면, 배치마다 다른 특징 수준을 읽었을 가능성(읽는 법 2)을 함께 적어요.
            short_batch = probs.groupby("batch")["short"].mean().idxmax()
            if (near["p"] < 0.1).all() and short_batch not in set(near["batch"]):
                who = "이 셀" if len(near) == 1 else f"이 {len(near)}개"
                text += (f" 다만 {who}의 단수명일 확률은 {'모두 ' if len(near) > 1 else ''}0.1 미만"
                         f"({', '.join(f'{v:.2f}' for v in near['p'])})이라, 모델이 기준선에서 망설였다기보다 "
                         f"배치마다 다른 특징 수준을 보고 {'·'.join(sorted(set(near['batch'])))} 셀을 장수명 쪽으로 "
                         "판정했을 수 있어요(가설, 읽는 법 2).")
            lines += ["", text]
        elif len(missed):
            lines += ["", f"→ 놓친 단수명 {len(missed)}개 중 수명이 기준의 {NEAR * 100:.0f}% 이상인(기준 바로 아래) 셀은 없어요."]
        lines.append("")
    ra = {k: _row(ctx["summ_alt"], 5, k) for k in (*sc.KINDS, "majority", "batch")}
    alt_y = sc.labels(f5["cycle_life"], sc.ALT_THRESHOLD)
    alt_tb = sc.batch_table(f5, sc.ALT_THRESHOLD).set_index("batch")
    lines += [
        f"### 5. 기준을 {sc.ALT_THRESHOLD}사이클로 바꾸면 (5사이클, 무작위 나누기 {ctx['repeats']}번)",
        "",
        f"{sc.ALT_THRESHOLD}사이클 기준에서는 단수명이 {int(alt_y.sum())}개이고 배치별로 "
        + " · ".join(f"{b} {int(alt_tb.loc[b, 'short'])}/{int(alt_tb.loc[b, 'cells'])}" for b in alt_tb.index)
        + "개라, 배치만으로는 덜 갈려요.",
        "",
        "| 모델 | 정확도 | 놓친 단수명 셀 | 버린 장수명 셀 |",
        "|---|---|---|---|",
    ]
    for kind in (*sc.KINDS, "batch", "majority"):
        r = ra[kind]
        lines.append(f"| {sc.NAMES[kind]} | {_rng(r, 'accuracy')} | {_rng(r, 'miss_rate')} | {_rng(r, 'discard_rate')} |")
    lines += [
        "",
        "### 6. 같은 충전 조건 안에서도 통할까 (공장 선별에 가장 가까운 질문)",
        "",
        f"같은 배치·같은 충전 조건 셀이 2개 이상인 {int(w['groups'].iloc[0])}개 조건({int(w['cells'].iloc[0])}개 셀)에서 "
        f"셀을 두 개씩 짝지어({int(w['pairs'].iloc[0])}쌍) **ΔQ 분산이 큰 셀이 먼저 수명이 다했는지** 맞힌 비율이에요. "
        "배치와 충전 조건이 같으니, 남은 차이는 셀 자체(제조 편차 등), 챔버 안 위치(온도 차이), 측정 잡음 같은 데서 올 수 있어요"
        f"([보충 자료]({PAPER['si']}) Supplementary Fig. 3은 셀 평균 온도가 충전 조건·배치뿐 아니라 챔버 안 위치에 따라서도 달라진다고 적었어요). "
        "50%면 동전 던지기예요.",
        "",
        "| 사이클 | 맞힌 비율 | 수명을 조건 안에서 섞었을 때 (5~95%) |",
        "|---|---|---|",
    ]
    for n in sc.CYCLES:
        r = w.loc[n]
        mark = "**" if r["concordance"] > r["shuffled_p95"] else ""
        lines.append(f"| {n} | {mark}{r['concordance']:.1f}%{mark} | {r['shuffled_p5']:.1f} ~ {r['shuffled_p95']:.1f}% |")
    lines += ["", "*'수명을 조건 안에서 섞었을 때' = 같은 배치·같은 충전 조건 안에서 셀의 수명만 무작위로 섞고 맞힌 비율을 다시 세는 일을 "
              f"{ctx['shuffles']:,}번 해서, 그 값들의 가운데 90%(5~95%)를 적은 거예요. ΔQ가 수명과 상관없을 때 우연히 나오는 범위예요. "
              "굵은 글씨 = 섞었을 때의 95% 지점보다 높음. 짝들은 셀을 공유해서 서로 독립이 아니에요.*", ""]

    # ---------------------------------------------------------------- 읽는 법
    alt_gap = float(ra["full"]["accuracy_median"] - ra["batch"]["accuracy_median"])
    op_miss_drop = float(base_r["miss_rate_median"] - op_r["miss_rate_median"])
    op_discard_up = float(op_r["discard_rate_median"] - base_r["discard_rate_median"])
    beats = [n for n in sc.CYCLES if w.loc[n, "concordance"] > w.loc[n, "shuffled_p95"]]
    beat_batch = max(r5[k]["accuracy_median"] for k in sc.KINDS) >= r5["batch"]["accuracy_median"]
    old = ctx["old"]
    train_short = sum(k for _, k in old["train"].values())
    train_long = sum(n for n, _ in old["train"].values()) - train_short
    batch_sec = float(_paper_row(paper, 5, "batch", "secondary_test")["accuracy"])
    # 700사이클 기준에서 놓침을 줄인 것이 '셀 신호' 덕분인지, 충전 시간(= 충전 조건)·온도 덕분인지 구분해서 쓴다
    cell_miss, batch_miss = float(ra["cell"]["miss_rate_median"]), float(ra["batch"]["miss_rate_median"])
    cell_vs_batch = ("와 비슷해요" if abs(cell_miss - batch_miss) < 3 else
                     "보다 낮아요" if cell_miss < batch_miss else "보다 높아요")
    alt_cell_note = f"{sc.NAMES['cell']} 모델의 놓친 단수명 셀은 {cell_miss:.1f}%로 배치만 보고 찍기({batch_miss:.1f}%){cell_vs_batch}. "
    if float(ra["full"]["miss_rate_median"]) < min(cell_miss, batch_miss):
        alt_cell_note += ("놓침이 가장 적은 전체 모델은 충전 시간(= 충전 조건)과 온도를 함께 쓰므로, 이 차이도 "
                          "'셀 하나하나를 읽은 덕분'이라기보다 '충전 조건을 안 덕분'일 수 있어요(가설).")
    lines += [
        "## 읽는 법 (숫자는 코드가 계산)",
        "",
        f"1. **5사이클 데이터만으로도 '모두 많은 쪽으로 찍기'보다 훨씬 잘 맞혀요.** 무작위 나누기 정확도 중앙값이 "
        + ", ".join(f"{sc.NAMES[k]} {r5[k]['accuracy_median']:.1f}%" for k in sc.KINDS)
        + f"로, 모두 많은 쪽으로 찍기({r5['majority']['accuracy_median']:.1f}%)보다 높아요. 다만 이 정확도가 무엇을 읽은 결과인지는 2번을 함께 봐야 해요.",
        (f"2. **하지만 '배치만 보고 찍기'({r5['batch']['accuracy_median']:.1f}%)를 넘지 못해요.** " if not beat_batch else
         f"2. **'배치만 보고 찍기'({r5['batch']['accuracy_median']:.1f}%)와 비슷하거나 높지만,** ")
        + f"{sc.THRESHOLD}사이클 기준의 단수명 셀은 거의 다 2배치라서, 높은 정확도가 '셀 하나하나를 읽었다'는 증거가 되지 못해요. "
        "위 배치별 중앙값처럼 5사이클 특징도 배치마다 수준이 달라서, 모델이 2배치를 알아보는 데 썼을 수 있어요(가설).",
        (f"3. **사이클을 더 본다고 꼭 나아지지 않아요.** {sc.NAMES['variance']}는 {worst_n}사이클에서 가장 나빴어요"
         f"({acc[(worst_n, 'variance')]:.1f}%, 상관 r = {corr[worst_n]:.2f}). " if worst_n != sc.CYCLES[0] else
         f"3. **사이클을 더 볼수록 대체로 나아져요.** {sc.NAMES['variance']}는 5사이클에서 가장 나빴어요"
         f"({acc[(worst_n, 'variance')]:.1f}%). ")
        + f"[논문 보충 자료]({PAPER['si']})는 초반에는 '바로 이웃한 사이클끼리' 비교해야 잘 맞는다고 보고하며, "
        "챔버 온도가 몇 사이클에 걸쳐 오르내리기 때문이라는 가설을 냈어요. 우리는 5·10사이클은 이웃 사이클끼리, "
        "20사이클부터는 10번째 사이클과 비교했어요. "
        + (f"그래서 {worst_n}사이클의 하락은 '사이클 수'보다 '어느 사이클과 비교했나'의 영향일 수 있어요(가설). "
           if worst_n > 10 else "")
        + "[사이클 수 실험](cycles.md)도 함께 보세요.",
        f"4. **기준값은 공정의 선택이에요.** 기준값을 0.5에서 운영점(중앙값 {op_r['cut_median']:.2f})으로 낮추면 놓친 단수명 셀은 "
        f"중앙값 {base_r['miss_rate_median']:.1f}% → {op_r['miss_rate_median']:.1f}%로 {op_miss_drop:.1f}%p 줄고, "
        f"버린 장수명 셀은 {base_r['discard_rate_median']:.1f}% → {op_r['discard_rate_median']:.1f}%로 {op_discard_up:.1f}%p 늘어요. "
        f"하지만 학습 셀 41개로 정한 기준값은 나누기마다 {op_r['cut_p10']:.2f}~{op_r['cut_p90']:.2f}로 흔들리고, "
        f"평가 셀에서 놓친 비율의 90% 끝은 {op_r['miss_rate_p90']:.1f}%예요. 실제 라인이라면 더 많은 셀로 기준값을 다시 확인해야 해요.",
        f"5. **기준을 {sc.ALT_THRESHOLD}사이클로 바꾸면** 배치만 보고 찍기가 {ra['batch']['accuracy_median']:.1f}%로 떨어지고, "
        f"{sc.NAMES['full']}는 {ra['full']['accuracy_median']:.1f}%로 "
        + ("정확도가 같아요. " if abs(alt_gap) < 0.05 else f"{abs(alt_gap):.1f}%p {'높아요' if alt_gap > 0 else '낮아요'}. ")
        + f"대신 실수의 종류가 달라요: 놓친 단수명 셀은 배치만 보고 찍기 {ra['batch']['miss_rate_median']:.1f}% vs 전체 모델 "
        f"{ra['full']['miss_rate_median']:.1f}%, 버린 장수명 셀은 {ra['batch']['discard_rate_median']:.1f}% vs "
        f"{ra['full']['discard_rate_median']:.1f}%예요. "
        + alt_cell_note,
    ]
    if beats:
        lines.append(f"6. **같은 조건 안에서도 신호가 관찰돼요:** "
                     + ", ".join(f"{n}사이클 {w.loc[n, 'concordance']:.1f}%" for n in beats)
                     + "가 '수명을 섞었을 때'의 95%보다 높아요. 나머지 사이클 수는 운으로도 나올 수 있는 범위예요. "
                       "짝이 서로 독립이 아니고 조건마다 셀이 2~8개뿐이라, 이 결과는 '가능성이 보인다' 정도로만 읽어야 해요.")
    else:
        lines.append("6. **같은 조건 안에서는 신호가 뚜렷하지 않아요:** 어느 사이클 수에서도 '수명을 섞었을 때'의 범위를 넘지 못했어요.")
    lines += [
        "",
        "## 한계",
        "",
        f"- **배치와 단수명이 거의 겹쳐요.** {sc.THRESHOLD}사이클 기준에서는 모델이 셀을 읽었는지 배치를 읽었는지 가를 수 없어요. "
        "공장 셀 선별(같은 조건, 같은 시험 환경)에 바로 옮길 수 있는 결과가 아니에요.",
        "- **관찰 데이터예요.** 특징과 수명이 함께 나타난다는 것이지, 특징이 수명을 정한다는 뜻이 아니에요.",
        "- **ΔQ₅₋₄는 아주 작은 차이예요.** LFP는 전압이 평평한 구간이 길어서, 전압이 조금만 흔들려도 같은 전압에서의 용량 차이(ΔQ)가 크게 바뀌어요."
        " 온도·측정 잡음에 민감할 수 있어요(가설).",
        "- **특징이 논문과 같지 않아요.** 논문의 'full classifier' 후보 18개를 다 만들지 않았고, 온도 적분은 평균 온도로 대신했어요. 셀 수도 달라요.",
        "- **운영점의 비용 비율(놓침 1 = 버림 5)은 우리가 정한 가정이에요.** 실제 비용(리콜·보증 비용 vs 셀 원가)에 따라 달라져요.",
        f"- **평가 셀이 적어요.** 논문 나누기의 2차 테스트에는 단수명 셀이 {int(sec['cycle_life'].lt(sc.THRESHOLD).sum())}개뿐이라, "
        "그 나누기의 놓침 비율은 거의 의미가 없어요. 그래서 무작위 나누기를 함께 봤어요.",
        "- 같은 124개 셀을 다시 나누는 것이라, 정말로 새 셀로 확인한 것과 같지는 않아요.",
        "",
        "## 시행착오",
        "",
        "1. **이 과제를 '셀 품질 선별'로만 생각했어요.** 데이터를 처음 훑으며 배치별로 단수명 셀을 세어 보니 "
        f"{short_total}개 중 {short_b2}개가 2배치였어요. 이대로면 정확도가 높아도 셀을 읽었다는 근거가 안 돼요. "
        f"→ 수정: 모델을 만들기 전에 '배치만 보고 찍기' 비교 기준, {sc.ALT_THRESHOLD}사이클 기준, 같은 조건 안 짝 비교를 계획에 넣었어요. "
        "→ 재발 방지: 라벨(정답)을 만들면 먼저 배치·조건별로 세어 본다.",
        "2. **운영점을 '학습 셀에서 놓침 10% 이하인 가장 높은 기준값'으로 정했다가 실패했어요.** 논문 나누기의 학습 셀은 "
        + " · ".join(f"{b} {n}개 중 단수명 {k}개" for b, (n, k) in old["train"].items())
        + f"라서 학습 셀 안에서는 거의 완벽하게 갈렸고, 조건을 겨우 만족하는 끝값 {old['cut']:.2f}가 골라졌어요. "
        f"이 값을 테스트에 쓰니 놓친 단수명이 {old['short']}개 중 {old['missed']}개(0.5일 때는 {old['missed_default']}개)로 오히려 늘었어요. "
        f"→ 수정: '놓침 1개 = 버림 {sc.MISS_COST}개' 손해가 가장 작은 기준값들의 **가운데 값**을 고르도록 바꿨어요. "
        "→ 재발 방지: 학습 셀에서 정한 기준값은 끝값이 아니라 여유 있는 값을 쓰고, 무작위 나누기로 흔들림을 함께 본다.",
        "3. **셀별 확률 표에서 '배치 이름' 열이 덮어써졌어요.** 비교 기준 이름(`batch`)을 확률 열 이름으로 그대로 써서, "
        "틀린 셀 목록을 뽑을 때 배치 칸에 0·1이 찍혔어요. → 수정: 확률 열 이름 앞에 `p_`를 붙였어요.",
        "4. **논문 나누기에서 비교 기준의 2차 테스트 성적이 엉뚱하게 나왔어요.** 3배치는 학습에 한 개도 없어서 '배치만 보고 찍기'는 "
        f"학습 셀 전체의 다수(단수명 {train_short} vs 장수명 {train_long})로 판정했고, 2차 테스트 정확도가 {batch_sec:.1f}%로 찍혔어요. "
        "규칙이 할 수 없는 일을 숫자로 만든 것이라, "
        "논문 나누기 표에서는 비교 기준을 빼고 '모두 장수명이라고 찍기'의 비율을 참고로 넣었어요. "
        "→ 재발 방지: 비교 기준도 '이 나누기에서 쓸 수 있는가'를 먼저 확인한다.",
        "5. **시험 실행 결과가 엉뚱한 폴더에 저장됐어요.** 반복 횟수를 줄여 build()를 먼저 돌려 보면서, 출력 폴더를 "
        "Git Bash 형식 경로(`/c/Users/...`)로 넘겼어요. 그런데 만든 문서가 예상한 폴더에 없었어요. "
        "Windows용 Python은 이 경로를 `C:\\c\\Users\\...`로 읽어서 `C:\\c` 아래에 새 폴더를 만들었어요. "
        "→ 수정: 그 폴더에서 결과물을 찾아 검토하고, 안에 든 시험 출력 파일은 확인한 뒤 지웠어요"
        "(C 드라이브 바로 아래라 빈 폴더 삭제는 안전 검사가 막았어요). "
        "나중에 다른 검증 에이전트가 같은 실수를 한 번 더 했어요([시행착오 기록](troubleshooting.md) 37번). "
        "→ 재발 방지: Windows용 Python에 넘기는 경로는 `C:\\...` 형식으로 쓰고, 실행 뒤 결과 파일이 예상한 곳에 생겼는지 먼저 확인한다.",
        "6. **같은 코드를 두 번 돌렸더니 표의 숫자가 달라졌어요** (검증 단계에서 발견). 20사이클 '전체' 모델의 정확도 중앙값이 "
        "두 실행에서 0.6%p 달랐어요. L1 규제 로지스틱 회귀에 쓴 saga 풀이법은 학습 중에 데이터 순서를 무작위로 섞는데, "
        "시드(random_state, 무작위의 출발점)를 넘기지 않아서 실행마다 섞는 순서가 달랐어요. "
        "→ 수정: 모든 L1 모델에 시드를 넘기고, 같은 입력이면 같은 숫자가 나오는지 테스트로 확인해요. "
        "→ 재발 방지: 무작위가 들어가는 곳에는 모두 시드를 넘기고, 문서를 만든 뒤 한 번 더 돌려 같은지 비교한다.",
        "7. **해석 문장이 숫자보다 앞서 나갔어요** (검증 단계에서 발견). 처음 문서의 읽는 법 5번은 "
        "\"배치 안에도 단수명 셀이 섞이면 셀을 측정하는 쪽이 놓침을 줄여요\"라고 썼어요. 문장과 숫자를 하나씩 대조해 보니 "
        f"{sc.ALT_THRESHOLD}사이클 기준에서 {sc.NAMES['cell']} 모델의 놓친 단수명 셀은 {cell_miss:.1f}%로 "
        f"배치만 보고 찍기({batch_miss:.1f}%){cell_vs_batch}. "
        + (f"놓침이 적었던 건 충전 시간(= 충전 조건)과 온도도 쓰는 전체 모델({float(ra['full']['miss_rate_median']):.1f}%)이었어요. "
           if float(ra["full"]["miss_rate_median"]) < min(cell_miss, batch_miss) else "")
        + "같은 검증에서 '5사이클 신호는 분명히 있어요'(읽는 법 1번), '조건이 같으니 차이는 셀 자체(제조 편차 등)나 측정 잡음에서 와요'(6절)처럼 "
        "근거보다 센 표현도 나왔어요. "
        "→ 수정: 읽는 법 5번은 숫자에 따라 문장이 바뀌게 하고, 단정한 곳은 '(가설)'·'~올 수 있어요'로 낮추고 근거(보충 자료 등)를 붙였어요. "
        "→ 재발 방지: '셀을 읽었다'는 해석은 비교 기준(배치만 보고 찍기)과 숫자로 견준 뒤에만 쓰고, 그 숫자를 코드가 계산해 문장 옆에 붙인다.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    numbers = build()
    print({k: v for k, v in numbers.items() if not isinstance(v, dict)})
