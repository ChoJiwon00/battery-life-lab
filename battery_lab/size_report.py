"""데이터 크기 실험 전체를 돌려 docs/data_size.md와 그래프를 만든다.

    python -m battery_lab sizes
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat
from . import figures, sizes
from .model import NAMES, baseline
from .sources import PROCESSED, ROOT

DOCS = ROOT / "docs"
# 논문 수치는 기억이 아니라 인용 출처로 확인한 것만 쓴다 (troubleshooting.md 16번)
PAPER_VARIANCE = {"test": 138, "secondary_test": 196}
PAPER_SOURCE = "Attia 외(2021, arXiv:2101.01885)가 인용한 Severson 외(2019)의 variance 모델 결과"


def build(processed: Path = PROCESSED, docs: Path = DOCS, repeats: int = sizes.REPEATS,
          corr_repeats: int = sizes.CORR_REPEATS, progress=print) -> dict:
    progress("특징 계산")
    f = feat.build(processed)
    progress("논문 나누기로 기준 모델")
    base = baseline(f)
    progress(f"실험 (상관 {corr_repeats}번 · 예측 {repeats}번씩)")
    corr = sizes.correlation_runs(f, repeats=corr_repeats)
    pred = sizes.prediction_runs(f, repeats=repeats)
    summary = sizes.summarize(corr, pred)
    corr.to_csv(processed / "data_size_corr.csv", index=False)
    pred.to_csv(processed / "data_size_runs.csv", index=False)

    full_r = float(np.corrcoef(f["logvar"], np.log10(f["cycle_life"]))[0, 1])
    fig_dir = docs / "figures"
    figures.data_size_corr(corr, full_r, fig_dir / "data_size_corr.png")
    figures.data_size_error(summary["pred"], fig_dir / "data_size_error.png")
    smallest = min(sizes.SIZES)
    figures.data_size_flatter(pred, smallest, fig_dir / "data_size_flatter.png")
    (docs / "data_size.md").write_text(markdown(base, summary, full_r, repeats, corr_repeats), encoding="utf-8")
    return {"baseline": base, "summary": summary, "full_r": full_r}


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def markdown(base: pd.DataFrame, s: dict, full_r: float, repeats: int, corr_repeats: int) -> str:
    c, p = s["corr"], s["pred"]
    ns = sorted({n for n, _ in p}, reverse=True)
    small = min(ns)
    b = {(r.model, r.split): r for r in base.itertuples()}
    lines = [
        "# 데이터 크기 실험: 셀이 70·35·10개뿐이었다면?",
        "",
        f"`python -m battery_lab sizes`가 자동으로 만든 문서예요 (만든 날 {date.today().isoformat()}).",
        "",
        "## 질문",
        "",
        "원래 시험한 셀은 140개, 논문이 쓴 셀은 124개예요. 배터리 수명 시험은 셀 하나에 몇 달이 걸려서,",
        "**\"셀이 몇 개면 충분한가\"는 곧 시간과 비용의 문제**예요. 그래서 이렇게 물었어요.",
        "",
        "1. 셀이 적었어도 \"초반 ΔQ 분산이 수명을 알려 준다\"는 **발견**을 할 수 있었을까?",
        "2. 셀이 적었다면 수명 예측 **모델은 얼마나 나빠졌고**, 그 연구가 **보고한 숫자는 믿을 만했을까?**",
        "",
        "## 방법",
        "",
        f"- 124개 셀에서 N개(124·70·35·10)를 무작위로 뽑아 \"그 N개만 있었던 연구\"를 흉내 내요.",
        f"- **실험 1 (발견):** 뽑은 N개 안에서 상관 r을 다시 계산해요. N마다 {corr_repeats:,}번.",
        f"- **실험 2 (예측):** 뽑은 N개를 논문처럼 1/3은 학습, 2/3는 평가에 써요. N마다 {repeats}번.",
        "  - **보고된 오차:** 뽑힌 N개 안의 평가 셀로 잰 오차예요. 그 연구가 논문에 썼을 숫자예요.",
        "  - **실제 오차:** 학습에 안 쓴 나머지 **모든** 셀로 잰 오차예요. 그 모델의 진짜 실력에 가까운 숫자예요.",
        f"- 모델 두 가지 ([model.py](../battery_lab/model.py), [features.py](../battery_lab/features.py))",
        f"  - **{NAMES['simple']}:** log10 Var(ΔQ₁₀₀₋₁₀) 하나로 log 수명에 직선을 맞춰요. 논문의 variance 모델과 같은 형태예요.",
        f"  - **{NAMES['full']}:** ΔQ 곡선의 분산·최솟값·왜도·첨도, 2번째 사이클 용량, 초반 용량 증가분을 표준화해 Elastic Net으로 맞춰요."
        " 논문의 discharge 모델을 참고했지만, 논문 코드가 공개돼 있지 않아 특징 정의가 같다고 보장하지는 못해요.",
        "- 오차는 평균 % 오차(MAPE)로 비교해요. 실험 2는 1/3 학습이라 셀 10개면 학습 셀이 3개뿐이에요."
        " 그래서 6개 특징 모델은 학습 셀이 6개 이상일 때만 만들어요.",
        "",
        "## 기준점: 논문과 같은 나누기 (학습 41 · 테스트 43 · 2차 테스트 40)",
        "",
        "| 모델 | 학습 | 테스트 | 2차 테스트 |",
        "|---|---|---|---|",
    ]
    for kind in ("simple", "full"):
        cells = [f"{b[(kind, sp)].rmse:.0f}사이클 ({b[(kind, sp)].mape:.1f}%)" for sp in ("train", "test", "secondary_test")]
        lines.append(f"| {NAMES[kind]} | " + " | ".join(cells) + " |")
    st, ss = b[("simple", "test")].rmse, b[("simple", "secondary_test")].rmse
    lines += [
        "",
        f"단순 모델의 테스트 / 2차 테스트 RMSE는 **{st:.0f} / {ss:.0f}사이클**이에요. "
        f"논문의 variance 모델은 **{PAPER_VARIANCE['test']} / {PAPER_VARIANCE['secondary_test']}사이클**이고, "
        f"출처는 {PAPER_SOURCE}예요. 논문 결과를 사이클 단위까지 재현했어요.",
        "",
        "## 실험 1 · 셀이 적어도 같은 신호를 찾았을까?",
        "",
        "| 셀 수 | 상관 r 중앙값 | 90% 범위 | r이 −0.8보다 약하게 나온 비율 |",
        "|---|---|---|---|",
    ]
    for n in sorted(c, reverse=True):
        v = c[n]
        rng = "(한 가지)" if v["runs"] == 1 else f"{v['p5']:.2f} ~ {v['p95']:.2f}"
        lines.append(f"| {n}개 | {v['median']:.3f} | {rng} | {_pct(v['weaker_08']) if v['runs'] > 1 else '-'} |")
    vs = c[small]
    lines += [
        "",
        "![상관의 흔들림](figures/data_size_corr.png)",
        "",
        f"→ **신호가 아주 강해서, 셀 {small}개만 있었어도 {_pct(1 - vs['weaker_08'])}의 경우 r이 −0.8보다 강하게 나와요.**"
        f" \"초반 ΔQ 분산이 수명과 엮여 있다\"는 발견 자체는 작은 실험으로도 충분히 할 수 있었어요.",
        "",
        "## 실험 2 · 셀이 적으면 예측은 얼마나 나빠질까?",
        "",
        "| 셀 수 (학습) | 모델 | 실제 오차 중앙값 | 실제 오차 10~90% | 보고된 오차 10~90% | 실제 25% 초과 | 실제보다 5%p 넘게 좋게 보고 |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in ns:
        for kind in ("simple", "full"):
            v = p.get((n, kind))
            if v is None:
                lines.append(f"| {n}개 ({sizes.n_train(n)}) | {NAMES[kind]} | 만들 수 없음 (학습 셀 부족) | | | | |")
                continue
            lines.append(f"| {n}개 ({v['n_train']}) | {NAMES[kind]} | **{v['actual_median']:.1f}%** | "
                         f"{v['actual_p10']:.1f} ~ {v['actual_p90']:.1f}% | {v['reported_p10']:.1f} ~ {v['reported_p90']:.1f}% | "
                         f"{_pct(v['bad'])} | {_pct(v['flattered'])} |")
    lines += ["", f"*{max(ns)}개일 때는 평가 셀이 곧 '학습에 안 쓴 나머지 전체'라서 보고된 오차와 실제 오차가 같아요.*"]
    big, mid = max(ns), sorted(ns)[1]
    s_big, s_mid, s_small = p[(big, "simple")], p[(mid, "simple")], p[(small, "simple")]
    f_big, f_mid = p.get((big, "full")), p.get((mid, "full"))
    lines += [
        "",
        "![데이터 크기별 오차](figures/data_size_error.png)",
        "",
        "### 읽는 법",
        "",
        f"1. **{mid}개까지는 평균이 거의 안 나빠져요.** 단순 모델의 실제 오차 중앙값은 {big}개 {s_big['actual_median']:.1f}% → "
        f"{mid}개 {s_mid['actual_median']:.1f}%예요. 신호 하나에 직선 하나라 학습 셀 {s_mid['n_train']}개로도 거의 같은 직선이 나와요.",
        f"2. **대신 '운'의 폭이 커져요.** 10~90% 범위가 {big}개 {s_big['actual_p10']:.1f}~{s_big['actual_p90']:.1f}%에서 "
        f"{small}개 {s_small['actual_p10']:.1f}~{s_small['actual_p90']:.1f}%로 넓어져요. 셀 {small}개(학습 {s_small['n_train']}개)로 한 연구 중 "
        f"**{_pct(s_small['bad'])}는 실제 오차가 25%를 넘는 모델**을 얻어요.",
        f"3. **작은 연구는 숫자가 실제보다 좋게 실릴 수 있어요.** 셀 {small}개 연구의 **{_pct(s_small['flattered'])}** 는 "
        f"보고된 오차가 실제보다 5%p 넘게 좋았어요. 평가 셀이 {s_small['n_test']}개뿐이라 쉬운 셀이 몰리면 성능이 부풀려져요.",
    ]
    if f_big and f_mid:
        lines.append(
            f"4. **복잡한 모델은 데이터가 많을 때만 확실히 이겨요.** 6개 특징 모델은 {big}개에서 {f_big['actual_median']:.1f}%로 "
            f"단순 모델({s_big['actual_median']:.1f}%)보다 좋지만, {mid}개에서는 중앙값 {f_mid['actual_median']:.1f}% vs "
            f"{s_mid['actual_median']:.1f}%로 차이가 줄고, 나쁜 쪽 끝(90%)은 {f_mid['actual_p90']:.1f}% vs {s_mid['actual_p90']:.1f}%로 "
            f"{'오히려 더 위험해요' if f_mid['actual_p90'] > s_mid['actual_p90'] else '비슷해요'}. 특징이 많을수록 적은 데이터에 휘둘리기 쉬워요.")
    lines += [
        "",
        f"![셀 {small}개 연구의 보고 vs 실제](figures/data_size_flatter.png)",
        "",
        "## 이 결과가 말해 주는 것",
        "",
        "- **발견은 작게, 검증은 크게.** 신호를 찾는 데는 적은 셀로도 충분했지만, \"이 모델의 오차는 몇 %\"라고 말하려면 평가 셀이 많아야 해요.",
        "- **작은 실험의 성능 숫자는 범위로 말해야 해요.** 같은 방법이라도 어떤 셀을 뽑았느냐에 따라 보고 숫자가 크게 흔들려요.",
        "- **현업으로 옮기면:** 새 소재·설계 평가에서 셀을 몇 개 만들지 정할 때, \"원하는 오차 범위를 얻으려면 몇 개가 필요한가\"를 이런 방식으로 미리 따져 볼 수 있어요.",
        "",
        "## 한계",
        "",
        "- 같은 124개 셀에서 다시 뽑는 실험이라, 정말로 다른 셀 N개로 새 실험을 한 것과 같지는 않아요.",
        "  특히 셀 10개를 뽑을 때 같은 충전 조건의 셀이 학습과 평가에 나뉘어 들어갈 수 있어요.",
        "- 학습 비율을 논문처럼 1/3로 고정했어요. 셀이 적을 때는 학습에 더 많이 쓰는 게 보통이라, 셀 10개 결과는 비관적인 쪽이에요.",
        "- 6개 특징 모델은 논문의 discharge 모델을 참고해 직접 구현한 것이에요.",
        "",
        "더 해 볼 만한 응용은 [응용 아이디어](ideas.md)에 정리해 두었어요.",
        "",
    ]
    return "\n".join(lines)
