"""사이클 수 실험 전체를 돌려 docs/cycles.md와 그래프를 만든다.

    python -m battery_lab cycles
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .mdtext import write_doc
from . import cycle_figures, cycles
from .features import build_many, find_row
from .korean import euro, ieyo
from .model import NAMES
from .quality import load_curves, load_processed
from .sources import PROCESSED, ROOT

DOCS = ROOT / "docs"
PANELS = (20, 50, 100)


def _panel_curves(folder: Path, cells: pd.DataFrame) -> dict[int, list]:
    out = {k: [] for k in PANELS}
    for meta in cells.to_dict("records"):
        curves = load_curves(folder, meta["cell_id"])
        early = find_row(curves, 10)
        for k in PANELS:
            late = find_row(curves, k, forward=False)  # k번째 이후 곡선은 쓰지 않는다
            if early and late:
                d = curves["Qdlin"][late[0]].astype(np.float64) - curves["Qdlin"][early[0]]
                out[k].append((curves["Vdlin"], d, float(meta["cycle_life"])))
    return out


def build(processed: Path = PROCESSED, docs: Path = DOCS, split_repeats: int = cycles.SPLIT_REPEATS,
          grid_repeats: int = cycles.GRID_REPEATS, progress=print) -> dict:
    cells, summary = load_processed(processed)
    progress("사이클 수별 특징")
    feats = build_many(processed, cycles.LATES)
    cycles.check(feats)
    subs = cycles.substitutions(feats)
    minutes = cycles.minutes_per_cycle(processed)
    progress("신호 세기 (곡선 한 개 / 5사이클 중앙값)")
    curve = cycles.signal_curve(processed)
    dips = cycles.local_dips(curve)
    checks = {d["late"]: cycles.dip_checks(processed, d, summary) for d in dips}
    progress("논문 나누기")
    paper = cycles.paper_split(feats)
    progress(f"무작위 나누기 {split_repeats}번")
    splits = cycles.random_splits(feats, repeats=split_repeats)
    split_sum = cycles.summarize_splits(splits)
    progress(f"셀 수 × 사이클 수 지도 ({grid_repeats}번씩)")
    grid_runs = cycles.grid(feats, repeats=grid_repeats)
    grid_sum = cycles.summarize_grid(grid_runs)
    for name, df in (("cycles_signal", curve), ("cycles_paper", paper), ("cycles_splits", splits),
                     ("cycles_grid", grid_runs), ("cycles_substitutions", subs)):
        df.to_csv(processed / f"{name}.csv", index=False)

    fig_dir = docs / "figures"
    cycle_figures.signal(curve, minutes, fig_dir / "cycles_signal.png")
    cycle_figures.error(split_sum, minutes, fig_dir / "cycles_error.png")
    cycle_figures.delta_q_panels(_panel_curves(processed, cells), fig_dir / "cycles_delta_q.png")
    cycle_figures.grid(grid_sum, minutes, fig_dir / "cycles_grid.png")
    text = markdown(curve, paper, split_sum, grid_sum, subs, minutes, dips, checks, split_repeats, grid_repeats,
                    splits=splits, total=len(cells))
    write_doc(docs / "cycles.md", text)
    return {"minutes": minutes, "splits": split_sum, "grid": grid_sum, "dips": dips, "checks": checks}


def _days(k: int, minutes: float) -> str:
    return f"{k * minutes / 60 / 24:.1f}일"


def _bump_overlap(dips: list[dict], kept: list[int], bumps: list[int], err: pd.Series) -> str:
    """오차 그래프의 '혹'(바로 앞보다 오차가 커진 k)과 신호 꺾임이 실제로 겹치는지 계산해서 쓴다.

    처음엔 "위 표의 꺾임과 같은 현상이에요"를 고정 문장으로 썼는데, 꺾임 4곳 중 2곳에서는 오차가 오히려 줄었다.
    err = 무작위 나누기 오차 중앙값 (k → %). 오차는 10사이클 간격으로만 재서, 앞 점이 없는 꺾임은 비교하지 않는다.
    """
    lates = sorted(int(k) for k in err.index)
    prev = dict(zip(lates[1:], lates[:-1]))
    dip_ks = [d["late"] for d in dips]
    graded = [k for k in dip_ks if k in prev]
    both = sorted(k for k in graded if k in bumps)
    fell = sorted(k for k in graded if err[k] < err[prev[k]])
    off = sorted(k for k in dip_ks if k not in prev)
    alone = [k for k in bumps if k not in dip_ks]

    def ks(xs):
        return ", ".join(map(str, xs))

    out = "- 10사이클 간격의 오차 그래프(무작위 나누기 중앙값, 단순 모델)에서 바로 앞보다 오차가 커진 곳은"
    out += f" **{ks(bumps)}사이클**이에요." if bumps else " 없어요."
    if both:
        out += f" 위 표의 꺾임 {len(dips)}곳 중 {len(both)}곳({ks(both)}사이클)과 겹쳐요."
        still = [k for k in both if k in kept]
        if len(still) == len(both):
            out += f" 겹치는 곳은{' 모두' if len(both) > 1 else ''} 5사이클 중앙값 곡선으로도 남은 꺾임이에요."
        elif still:
            out += f" 그중 5사이클 중앙값 곡선으로도 남은 꺾임은 {ks(still)}사이클이에요."
    elif bumps:
        out += f" 위 표의 꺾임 {len(dips)}곳과는 겹치지 않아요."
    if fell:
        out += (f" {ks(fell)}사이클에서는 오차가 오히려 줄었어요 ("
                + ", ".join(f"{err[prev[k]]:.1f}% → {err[k]:.1f}%" for k in fell) + ").")
    if off:
        out += f" {ks(off)}사이클은 10사이클 간격 오차 그래프에 바로 앞 점이 없어서 비교하지 못했어요."
    if alone:
        out += f" 꺾임이 아닌 {ks(alone)}사이클에서도 오차가 커졌어요."
    if both:
        out += " 겹친다는 것까지만 확인했고, 같은 원인인지는 몰라요."
    return out


def markdown(curve, paper, s, g, subs, minutes, dips, checks, split_repeats, grid_repeats,
             splits=None, total=124) -> str:
    one = curve[curve["window"] == 1].set_index("late")["r"]
    five = curve[curve["window"] == 5].set_index("late")["r"]
    med = {kind: s[s["model"] == kind].set_index("late") for kind in ("simple", "full")}
    ref = float(med["simple"].loc[100, "median"])
    lines = [
        "# 사이클 수 실험: 몇 사이클이면 충분할까?",
        "",
        f"`python -m battery_lab cycles`가 자동으로 만든 문서예요 (만든 날 {date.today().isoformat()}).",
        "",
        "## 질문",
        "",
        "논문은 **100번째와 10번째 사이클**의 방전 곡선 차이(ΔQ₁₀₀₋₁₀)로 수명을 예측했어요.",
        "20·30·…·90사이클만 보고도 예측할 수 있을까요? [데이터 크기 실험](data_size.md)이 \"셀 몇 개\"였다면 이번엔 \"며칠\"이에요.",
        "",
        "## 왜 중요한가",
        "",
        f"- 이 데이터에서 사이클 하나는 중앙값 **{minutes:.0f}분**이에요(원시 데이터가 있는 100사이클 이내 사이클들로 측정). "
        f"100사이클이면 약 **{_days(100, minutes)}**이 걸려요.",   # _days()는 늘 '일'로 끝나서 '이'
        "- **셀 개발:** 새 소재·설계를 평가할 때 며칠 만에 \"오래 갈 셀인지\" 감을 잡을 수 있으면 개발 속도가 빨라져요.",
        "- **공정 개발:** 출하 전 검사·에이징을 얼마나 길게 해야 하는지 정하는 근거가 돼요.",
        "",
        "## 방법",
        "",
        "- ΔQ₁₀₀₋₁₀ 대신 **ΔQ_k−10**(k번째 − 10번째 사이클의 방전 곡선)으로 특징을 다시 만들어요. k = 20, 30, …, 100.",
        "  용량 특징(2번째 사이클 용량, 초반 용량 증가분)도 2~k사이클 안에서만 계산해요. **k번째 이후의 데이터는 쓰지 않아요.**",
        f"- 모델은 [데이터 크기 실험](data_size.md)과 같은 두 가지예요: {NAMES['simple']}, {NAMES['full']}.",
        f"- 평가는 세 가지로 해요.",
        "  1. 논문과 같은 나누기 (학습 41 · 테스트 43 · 2차 테스트 40)",
        f"  2. 124개를 41 / 83으로 무작위로 {split_repeats}번 나누기. 반복마다 **같은 나누기를 모든 k에 써서** 사이클 수 효과만 비교해요.",
        f"  3. 셀 수 N(124·70·35·10) × k 지도: N개를 {grid_repeats}번 뽑아 1/3로 학습하고, 학습에 안 쓴 나머지 모든 셀로 오차를 재요. 단순 모델만 써요.",
        f"- 시험 기간은 사이클당 {minutes:.0f}분으로 환산했어요. 셀마다·충전 조건마다 실제 시간은 달라요.",
    ]
    if len(subs):
        lines += ["- 곡선이 깨끗하지 않으면(품질 점검 기준) 가까운 깨끗한 곡선을 대신 썼어요. "
                  "**k번째는 k−1, k−2처럼 앞쪽으로만** 찾고(k번째 이후 데이터를 쓰지 않도록), 10번째는 ±2사이클 안에서 찾아요:",
                  "", "| k | 셀 | 대신 쓴 사이클 |", "|---|---|---|"]
        lines += [f"| {r.late} | {r.cell_id} | {r.late_used} |" for r in subs.itertuples()]
    lines += [
        "",
        "## 결과 1 · 신호는 언제 올라오나",
        "",
        "| k | " + " | ".join(str(k) for k in range(20, 101, 10)) + " |",
        "|---|" + "---|" * 9,
        "| 시험 기간 | " + " | ".join(_days(k, minutes) for k in range(20, 101, 10)) + " |",
        "| 상관 r (곡선 한 개) | " + " | ".join(f"{one[k]:.3f}" for k in range(20, 101, 10)) + " |",
        "| 상관 r (5사이클 중앙값) | " + " | ".join(f"{five[k]:.3f}" for k in range(20, 101, 10)) + " |",
        "",
        "*상관 r = 두 값이 함께 움직이는 정도(−1~+1). −1에 가까울수록 \"ΔQ 분산이 크면 수명이 짧다\"는 관계가 뚜렷해요. "
        "5사이클 중앙값 = k번째 곡선 하나 대신 k−4~k번째 곡선 5개의 중앙값, 10번째 대신 8~12번째 곡선 5개의 중앙값으로 ΔQ를 만든 것"
        "(k번째 이후 데이터는 쓰지 않아요).*",
        "",
        "![신호 세기](figures/cycles_signal.png)",
        "",
        "![ΔQ 곡선의 변화](figures/cycles_delta_q.png)",
        "",
        "## 결과 2 · 예측 오차는 어떻게 줄어드나",
        "",
        f"무작위 나누기 {split_repeats}번의 평균 % 오차(MAPE):",
        "",
        "| k | 시험 기간 | " + " | ".join(f"{NAMES[k]} 중앙값 (10~90%)" for k in ("simple", "full")) + " |",
        "|---|---|---|---|",
    ]
    for k in cycles.LATES:
        cells = [f"{med[kind].loc[k, 'median']:.1f}% ({med[kind].loc[k, 'p10']:.1f}~{med[kind].loc[k, 'p90']:.1f})"
                 for kind in ("simple", "full")]
        lines.append(f"| {k} | {_days(k, minutes)} | " + " | ".join(cells) + " |")
    pt = paper[paper["split"] != "train"].pivot_table(index="late", columns=["model", "split"], values="mape")
    lines += [
        "",
        "![사이클 수별 오차](figures/cycles_error.png)",
        "",
        "논문과 같은 나누기로 잰 값 (MAPE):",
        "",
        "| k | 단순 · 테스트 | 단순 · 2차 테스트 | 6개 특징 · 테스트 | 6개 특징 · 2차 테스트 |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {k} | {pt.loc[k, ('simple', 'test')]:.1f}% | {pt.loc[k, ('simple', 'secondary_test')]:.1f}% | "
              f"{pt.loc[k, ('full', 'test')]:.1f}% | {pt.loc[k, ('full', 'secondary_test')]:.1f}% |" for k in cycles.LATES]
    lines += ["", "## 결과 3 · 셀 몇 개 × 몇 사이클이면 될까", "", "실제 오차 중앙값 (단순 모델):", "",
              "| 셀 수 | " + " | ".join(str(k) for k in cycles.LATES) + " |", "|---|" + "---|" * len(cycles.LATES)]
    gm = g.pivot(index="n", columns="late", values="median")
    go = g.pivot(index="n", columns="late", values="over25")
    for n in sorted(gm.index, reverse=True):
        lines.append(f"| {n}개 | " + " | ".join(f"{gm.loc[n, k]:.1f}%" for k in cycles.LATES) + " |")
    lines += ["", "실제 오차가 25%를 넘는 비율:", "",
              "| 셀 수 | " + " | ".join(str(k) for k in cycles.LATES) + " |", "|---|" + "---|" * len(cycles.LATES)]
    for n in sorted(go.index, reverse=True):
        lines.append(f"| {n}개 | " + " | ".join(f"{go.loc[n, k]:.0%}" for k in cycles.LATES) + " |")
    lines += ["", "![셀 수 × 사이클 수 지도](figures/cycles_grid.png)", "", "## 읽는 법", ""]

    first_strong = next((k for k in range(20, 101, 10) if one[k] <= -0.8), None)
    if first_strong:
        lines.append(f"1. **신호는 {first_strong}사이클(약 {_days(first_strong, minutes)})쯤이면 r이 −0.8보다 강해져요.** "
                     f"20사이클에서는 r = {euro(f'{one[20]:.2f}')} 약하고, 100사이클은 {ieyo(f'{one[100]:.3f}')}.")

    def stable_from(kind, ok):
        """조건을 '그 k부터 100까지 계속' 만족하는 가장 짧은 k. 처음 만족한 k만 보면 바로 뒤에서 다시 벗어날 수 있다."""
        return next((k for k in cycles.LATES
                     if all(ok(med[kind].loc[j, "median"]) for j in cycles.LATES if j >= k)), None)

    def first_then_back(kind, ok, stable):
        """stable보다 앞에서 처음 만족했다가 다시 벗어난 경우: (처음 만족한 k, 다시 벗어난 k)."""
        first = next((k for k in cycles.LATES if ok(med[kind].loc[k, "median"])), None)
        if first is None or (stable is not None and first >= stable):
            return None
        back = next(k for k in cycles.LATES if k > first and not ok(med[kind].loc[k, "median"]))
        return first, back

    def near(v):
        return v <= ref + 2

    def match(v):
        return v <= ref

    simple_close = stable_from("simple", near)
    if simple_close and simple_close < 100:
        line = (f"2. **단순 모델은 {simple_close}사이클부터 100사이클 성능의 2%p 안에 계속 머물러요** "
                f"({med['simple'].loc[simple_close, 'median']:.1f}% vs {ref:.1f}%).")
    else:
        line = "2. **단순 모델은 100사이클보다 짧게 보면서 100사이클 성능의 2%p 안을 끝까지 유지하는 지점이 없어요.**"
    jump = first_then_back("simple", near, simple_close)
    if jump:
        line += (f" {jump[0]}사이클도 2%p 안({med['simple'].loc[jump[0], 'median']:.1f}%)이지만 "
                 f"{jump[1]}사이클에서 {med['simple'].loc[jump[1], 'median']:.1f}%로 다시 벗어나요.")
    lines.append(line)
    full_match = stable_from("full", match)
    if full_match:
        line = (f"3. **6개 특징 모델은 {full_match}사이클(약 {_days(full_match, minutes)})부터 "
                f"단순 모델의 100사이클 성능({ref:.1f}%) 이하를 계속 유지해요** "
                f"(무작위 나누기 중앙값 {med['full'].loc[full_match, 'median']:.1f}%).")
        early = first_then_back("full", match, full_match)
        if early:
            line += (f" {early[0]}사이클도 중앙값은 {med['full'].loc[early[0], 'median']:.1f}%이지만 "
                     f"{early[1]}사이클에서 {med['full'].loc[early[1], 'median']:.1f}%로 다시 나빠져서 기준으로 삼지 않았어요.")
        lines.append(line)
        if splits is not None:
            a = splits[(splits["model"] == "full") & (splits["late"] == full_match)].set_index("repeat")["mape"]
            b = splits[(splits["model"] == "simple") & (splits["late"] == 100)].set_index("repeat")["mape"]
            share = float((a <= b.reindex(a.index)).mean())
            pf = (pt.loc[full_match, ("full", "test")], pt.loc[full_match, ("full", "secondary_test")])
            ps = (pt.loc[100, ("simple", "test")], pt.loc[100, ("simple", "secondary_test")])
            worse = [name for name, x, y in (("테스트", pf[0], ps[0]), ("2차 테스트", pf[1], ps[1])) if x > y]
            lines.append(f"   - 하지만 **같은 나누기끼리 짝지어 보면** {full_match}사이클 6개 특징 모델이 100사이클 단순 모델보다 "
                         f"좋거나 같은 경우는 {share:.0%}예요. 논문 나누기에서는 테스트 {pf[0]:.1f}% · 2차 테스트 {pf[1]:.1f}%로, "
                         f"100사이클 단순 모델({ps[0]:.1f}% · {ps[1]:.1f}%)보다 "
                         + (f"{'·'.join(worse)}에서 나빠요." if worse else "모두 좋거나 같아요."))
            verdict = ("시험 기간을 줄일 **가능성**은 보이지만, 확실하다고 말하기엔 일러요"
                       if share < 0.7 or worse else "시험 기간을 줄일 수 있어 보여요")
            lines.append(f"   - 그래서 약 {_days(100, minutes)} → {_days(full_match, minutes)}로 {verdict}.")
    small = min(gm.index)
    lines.append(f"4. **셀 수 부족은 사이클로 메울 수 없어요.** 셀 {small}개면 100사이클까지 봐도 실제 오차가 25%를 넘는 비율이 "
                 f"{go.loc[small, 100]:.0%}예요. 반대로 셀이 35개 이상이면 {min(k for k in cycles.LATES if go.loc[35, k] <= 0.05)}사이클부터 "
                 "그 비율이 5% 아래로 내려가요.")
    bumps = cycles.error_bumps(s)
    paper_simple = pt[("simple", "test")]
    paper_bumps = [int(k) for k, a, b in zip(paper_simple.index[1:], paper_simple.values[:-1], paper_simple.values[1:]) if b > a]
    if dips:
        still = [d for d in dips if d["late"] in five.index and d["prev"] in five.index and d["next"] in five.index
                 and five[d["late"]] > five[d["prev"]] and five[d["late"]] > five[d["next"]]]
        temps = [abs(checks[d["late"]]["corr_all"]) for d in dips]
        lines += ["", "### 오르락내리락하는 지점 (원인 미확인)", "",
                  f"신호는 사이클이 늘수록 강해지지만 한 방향으로만 가지는 않아요. 5사이클 간격으로 보면 양옆보다 약해지는 지점이 "
                  f"**{len(dips)}곳** 있어요. 두 가지 가설을 데이터로 확인했어요.", "",
                  "| k | r | 양옆 r | 꺾인 정도 | 5사이클 중앙값 곡선의 r | 특징 벗어남 ↔ 온도 벗어남 상관 |",
                  "|---|---|---|---|---|---|"]
        for d in dips:
            c = checks[d["late"]]
            w5 = f"{five[d['late']]:.3f}" if d["late"] in five.index else "-"
            lines.append(f"| {d['late']} | {d['r']:.3f} | {d['r_prev']:.3f} · {d['r_next']:.3f} | {d['drop']:.3f} | {w5} | "
                         f"{c['corr_all']:+.2f} |")
        lines += ["", "*꺾인 정도 = 이 k의 r − 양옆 r의 평균. r이 음수라서 이 값이 클수록 양옆보다 약해졌다는 뜻이에요. "
                      "특징 벗어남 ↔ 온도 벗어남 상관 = 셀마다 이 k의 특징이 ±10사이클 평균에서 벗어난 정도와, "
                      "같은 방식으로 잰 셀 겉면 온도가 벗어난 정도의 상관이에요.*"]
        fewer = curve[curve["cells"] < total]
        if len(fewer):
            lines += ["", "*깨끗한 곡선을 찾지 못한 셀이 있어서 셀 " + str(total) + "개보다 적게 계산한 점: "
                      + ", ".join(f"k={r.late}({'곡선 한 개' if r.window == 1 else '5사이클 중앙값'}, {r.cells}개)"
                                  for r in fewer.itertuples()) + ".*"]
        gone = [d["late"] for d in dips if d not in still]
        kept = [d["late"] for d in still]
        mixed = [d["late"] for d in dips
                 if len({np.sign(v) for v in checks[d["late"]]["by_batch"].values()}) > 1]
        lines += ["",
                  f"- **곡선 한 개의 잡음 때문인가? 일부만 맞아요.** 5사이클 중앙값 곡선으로 바꾸면 "
                  f"{', '.join(map(str, gone)) or '없음'}사이클의 꺾임은 사라지지만, "
                  f"{', '.join(map(str, kept)) or '없음'}사이클은 그대로 남아요.",
                  f"- **온도 때문인가? 이 데이터로는 가를 수 없어요.** 셀마다 잰 상관은 모두 절댓값 {max(temps):.2f} 이하로 약하고, "
                  f"{len(mixed)}/{len(dips)}곳은 배치마다 상관의 방향(+/−)도 달라요. 하지만 셀 겉면 온도는 공식 설명상 참고용이라 "
                  "잡음이 크고([데이터 카드](dataset_card.md)), 셀끼리 비교한 상관은 배치 전체가 함께 움직이는 변화를 잡지 못해요. "
                  "같은 방식으로 잰 배치별 온도 중앙값의 벗어남: "
                  + " / ".join(f"{d['late']}사이클 "
                               + ", ".join(f"{b} {v:+.2f}°C" for b, v in checks[d['late']]['batch_temp_shift'].items())
                               for d in dips) + ".",
                  _bump_overlap(dips, kept, bumps, med["simple"]["median"]),
                  "- 그래서 \"몇 사이클이면 되나\"는 한 지점이 아니라 **전체 흐름과 반복 실험의 범위**로 판단했어요."]
    lines += [
        "",
        "## 한계",
        "",
        "- 10번째 사이클을 기준으로 고정했어요. 기준 사이클을 바꾸는 실험은 하지 않았어요.",
        f"- 시험 기간은 사이클당 {minutes:.0f}분(중앙값)으로 환산한 대략값이에요.",
        "- 같은 124개 셀 안에서 나눈 실험이에요. 다른 셀·다른 조건에서도 같은 사이클 수가 충분하다는 보장은 없어요.",
        "- 셀 수 × 사이클 수 지도는 단순 모델만 그렸어요(셀 10개면 6개 특징 모델을 만들 수 없어서).",
        "",
        "## 시행착오",
        "",
        f"- **\"{', '.join(map(str, paper_bumps))}사이클이 나쁘다\"고 결론 낼 뻔했어요.** 논문 나누기 한 번만 보면 그 지점들의 오차가 바로 앞보다 커져서 그렇게 보였어요. "
        "이웃 사이클을 촘촘히(5사이클 간격, 70사이클 주변은 1사이클 간격) 다시 재 보니 r이 사이클 하나 차이로도 오르내렸고, "
        "곡선 잡음·온도 가설도 데이터로 확인했지만 둘 다 충분하지 않았어요. 원인은 모른다고 적었어요.",
        "- **자동 분석이 \"가장 크게 꺾인 한 곳\"만 골랐더니 설명이 어긋났어요.** 70사이클을 보고 쓴 \"오차 그래프도 그 지점에서 나빠진다\"는 문장이, "
        "코드가 고른 30사이클(신호가 빠르게 오르는 구간)에 그대로 붙었어요. 30사이클의 오차는 오히려 줄었어요. "
        "그래서 꺾이는 지점을 모두 나열하고, 오차 그래프와의 대응도 계산으로 적도록 바꿨어요.",
        "- **\"6개 특징 모델은 60사이클(약 2.2일)로 100사이클 성능과 같아진다\"고 썼었어요.** 조건을 *처음* 만족한 사이클 하나만 "
        "골랐는데, 바로 다음 70사이클에서 오차가 다시 커졌어요. 같은 나누기끼리 짝지어 보지도 않았어요. "
        "다른 에이전트의 검토에서 지적받고, \"그 사이클부터 100까지 계속 만족\"하는 규칙으로 바꾸고 짝 비교와 논문 나누기 결과를 함께 적었어요.",
        "- **깨진 k번째 곡선을 k+1, k+2번째 곡선으로 대신 쓸 수 있었어요.** \"k사이클만 보고 예측\"한다면서 미래 데이터를 쓴 셈이에요. "
        "검토에서 지적받고 k번째는 앞쪽으로만 찾도록 고쳤고, 테스트로 막았어요.",
        "- **중앙값 곡선이 더 나을 거라 짐작했는데 아니었어요.** 5사이클 중앙값 곡선은 전체적으로 비슷하거나 약간 약해서 채택하지 않았어요.",
        f"- **응용 아이디어 문서에 \"사이클 하나가 약 1시간, 100사이클은 4일 넘게\"라고 적었었어요.** 한 셀만 보고 쓴 숫자였어요. "
        f"측정해 보니 중앙값 {minutes:.0f}분, 100사이클은 약 {_days(100, minutes)}이라 고쳤어요.",
        "",
    ]
    return "\n".join(lines)
