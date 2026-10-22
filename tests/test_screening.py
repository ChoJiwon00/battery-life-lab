"""5사이클 선별(분류) 실험 테스트. 진짜 데이터 없이 작은 가짜 데이터로만 확인한다."""

import numpy as np
import pandas as pd
import pytest

from battery_lab import screening as sc
from battery_lab import screening_report as report

POINTS = 20  # 진짜 방전 곡선은 1000칸


def fake_curves(life: float, n_cycles: int = 101, seed: int = 0, bad=()) -> dict[str, np.ndarray]:
    """수명이 짧을수록 사이클마다 곡선 모양이 더 크게 바뀌는(ΔQ 분산이 큰) 가짜 방전 곡선. bad 사이클은 깨진 곡선."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1.05, POINTS)
    bump = np.sin(np.linspace(0, np.pi, POINTS))
    cycles = np.arange(1, n_cycles + 1)
    q = np.stack([base * (1 - 1e-5 * k) - (1e-3 * 500 / life) * k * bump * (1 + rng.normal(0, 0.05))
                  for k in cycles])
    for c in bad:
        q[c - 1, POINTS // 2] = np.nan
    return {"cycle": cycles.astype(float), "Qdlin": q}


def fake_summary(n_cycles: int = 101, placeholder: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = np.arange(1, n_cycles + 1)
    df = pd.DataFrame({"cycle": c, "QD": 1.07 + 0.0005 * np.minimum(c, 4) - 0.0001 * c, "QC": 1.07,
                       "IR": 0.016 + 0.0001 * (c == 2) + rng.normal(0, 1e-5, len(c)), "Tavg": 31.0 + 0.1 * c,
                       "Tmin": 30.0, "Tmax": 35.0, "chargetime": 10.0 + 0.1 * c})
    if placeholder:  # 1배치처럼 1번째 사이클이 0으로 채워진 빈 자리
        df.loc[0, ["QD", "QC", "IR", "Tavg", "Tmin", "Tmax", "chargetime"]] = 0.0
    return df


# ---------------------------------------------------------------- 특징

def test_reference_cycle():
    assert [sc.reference(n) for n in (5, 10, 20, 100)] == [4, 9, 10, 10]


def test_pick_pair_moves_back_never_forward():
    curves = fake_curves(500, n_cycles=12, bad=(5,))
    _, _, late, early = sc.pick_pair(curves, 5)
    assert (late, early) == (4, 3)              # 5번째가 깨져서 두 곡선을 함께 한 사이클 앞으로
    assert sc.pick_pair(fake_curves(500, n_cycles=12), 5)[2:] == (5, 4)


def test_pick_pair_gives_up_instead_of_using_cycle_one():
    curves = fake_curves(500, n_cycles=12, bad=(3,))
    assert sc.pick_pair(curves, 3) is None      # (3,2)가 안 되면 (2,1)인데, 1번째는 쓰지 않는다


def test_early_features_use_only_data_up_to_cycle_n():
    curves = fake_curves(400, seed=1)
    summary = fake_summary(placeholder=True)
    summary.loc[summary["cycle"] == 6, ["IR", "QD"]] = [0.05, 1.15]   # 6번째 사이클의 튀는 값은 5사이클 특징에 들어가면 안 된다
    f = sc.early_features(curves, summary, 5)

    d = curves["Qdlin"][4] - curves["Qdlin"][3]
    assert f["logvar"] == pytest.approx(np.log10(np.var(d)))
    assert f["logmin"] == pytest.approx(np.log10(abs(d.min())))
    s = summary.set_index("cycle")
    assert f["q2"] == pytest.approx(s.loc[2, "QD"])
    assert f["qmax_q2"] == pytest.approx(s.loc[2:5, "QD"].max() - s.loc[2, "QD"])
    assert f["ir2"] == pytest.approx(s.loc[2, "IR"])
    assert f["ir_min"] == pytest.approx(s.loc[2:5, "IR"].min())
    assert f["ir_diff"] == pytest.approx(s.loc[5, "IR"] - s.loc[2, "IR"])
    assert f["chargetime"] == pytest.approx(s.loc[2:5, "chargetime"].mean())
    assert f["tavg"] == pytest.approx(s.loc[2:5, "Tavg"].mean())
    assert (f["late_used"], f["early_used"]) == (5, 4)
    assert all(np.isfinite(f[k]) for k in sc.FULL)


def test_identical_curves_leave_delta_q_features_empty_not_infinite():
    q = np.tile(np.linspace(0, 1.0, POINTS), (10, 1))
    f = sc.early_features({"cycle": np.arange(1, 11.0), "Qdlin": q}, fake_summary(10), 5)
    assert np.isnan(f["logvar"]) and np.isnan(f["logmin"])


# ---------------------------------------------------------------- 판정과 기준값

def test_labels_use_strictly_below_threshold():
    assert sc.labels([549, 550, 551, 148]).tolist() == [1, 0, 0, 1]


def test_confusion_counts_and_rates():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.1, 0.7, 0.2, 0.1, 0.1, 0.0, 0.5])
    m = sc.confusion(y, p, 0.5)   # 0.5 이상이면 뺀다 → 0.5인 장수명 셀도 버림
    assert (m["caught"], m["missed"], m["discarded"], m["kept"]) == (2, 2, 2, 4)
    assert m["accuracy"] == pytest.approx(60.0)
    assert m["miss_rate"] == pytest.approx(50.0) and m["discard_rate"] == pytest.approx(100 * 2 / 6)


def test_choose_cut_picks_middle_of_cheapest_cuts():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.8, 0.6, 0.3, 0.1])
    # 0.31~0.60에서 놓침 0·버림 0 → 손해 0. 그 가운데 값
    cheapest = [c for c in sc.CUTS if 0.3 < c <= 0.6]
    assert sc.choose_cut(y, p) == pytest.approx(cheapest[len(cheapest) // 2])


def test_choose_cut_prefers_discarding_when_missing_is_expensive():
    y = np.array([1, 0, 0])
    p = np.array([0.3, 0.4, 0.2])   # 단수명 하나를 잡으려면 장수명 하나(0.4)를 버려야 한다
    assert sc.choose_cut(y, p, miss_cost=5) <= 0.3
    assert sc.choose_cut(y, p, miss_cost=0.5) > 0.4


def test_batch_rule_uses_training_majority_per_batch_and_overall_for_unseen():
    train = pd.DataFrame({"batch": ["b1", "b1", "b1", "b2", "b2"]})
    y = np.array([0, 0, 1, 1, 1])
    test = pd.DataFrame({"batch": ["b1", "b2", "b3"]})
    assert sc.fit_predict("batch", train, y, test).tolist() == [0.0, 1.0, 1.0]   # b3는 학습에 없음 → 전체 다수(단수명)
    assert sc.fit_predict("majority", train, y, test).tolist() == [1.0, 1.0, 1.0]


def test_stratified_split_keeps_the_short_share():
    y = np.array([1] * 30 + [0] * 60)
    train, test = sc.stratified_split(y, 30, np.random.default_rng(0))
    assert len(train) == 30 and len(test) == 60 and not set(train) & set(test)
    assert y[train].sum() == 10


def fake_table(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """logvar가 클수록 수명이 짧은 가짜 특징 표 (세 배치 · 논문식 나누기)."""
    rng = np.random.default_rng(seed)
    logvar = rng.uniform(-7.5, -4.0, n)
    life = 10 ** (-0.3 * logvar + 0.95 + rng.normal(0, 0.03, n))
    df = pd.DataFrame({"cell_id": [f"b{i % 3 + 1}c{i}" for i in range(n)], "batch": [f"b{i % 3 + 1}" for i in range(n)],
                       "policy": [f"p{i // 4}" for i in range(n)], "cycle_life": life, "logvar": logvar})
    for k, name in enumerate(sc.FULL[1:]):
        df[name] = rng.normal(0, 1, n) + (0.3 * logvar if k < 2 else 0)
    df["split"] = np.where(df["batch"] == "b3", "secondary_test", np.where(np.arange(n) % 2, "train", "test"))
    return df


def test_classifiers_find_a_clear_signal():
    f = fake_table()
    threshold = float(np.median(f["cycle_life"]))
    m, probs = sc.paper_split(f, threshold=threshold)
    for kind in sc.KINDS:
        acc = m[(m["kind"] == kind) & (m["split"] == "test_all")]["accuracy"].iloc[0]
        assert acc > 80, kind
    assert "batch" in probs and set(probs["batch"]) == {"b1", "b2", "b3"}   # 배치 이름이 확률로 덮이지 않음


def test_l1_models_give_identical_numbers_every_run():
    """saga 풀이법은 데이터를 무작위로 섞는다. 시드를 넘기지 않으면 같은 코드를 두 번 돌려도 숫자가 달라진다."""
    f = fake_table()
    y = sc.labels(f["cycle_life"], float(np.median(f["cycle_life"])))
    tr = (f["split"] == "train").to_numpy()
    for kind in ("cell", "full"):
        first = sc.fit_predict(kind, f[tr], y[tr], f[~tr], seed=3)
        assert np.array_equal(first, sc.fit_predict(kind, f[tr], y[tr], f[~tr], seed=3)), kind
    assert np.array_equal(sc.oof_proba("cell", f[tr], y[tr], seed=3), sc.oof_proba("cell", f[tr], y[tr], seed=3))


def test_random_splits_reuse_the_same_split_for_every_model_and_cycle():
    f = fake_table()
    feats = {5: f, 10: f}
    runs, curves = sc.random_splits(feats, threshold=float(np.median(f["cycle_life"])), repeats=2, seed=1,
                                    kinds=("variance", "majority", "batch"), cut_kind="variance")
    assert set(runs["kind"]) == {"variance", "majority", "batch", "variance@op"}
    assert len(runs) == 2 * (2 * 3 + 1)
    same = runs[runs["kind"] == "variance"].pivot(index="repeat", columns="cycles", values="accuracy")
    assert (same[5] == same[10]).all()          # 같은 특징 표 + 같은 나누기 → 같은 결과
    assert len(curves) == 2 * len(sc.CUTS)
    first = curves[curves["cut"] == 0.0]
    assert (first["miss_rate"] == 0).all() and (first["discard_rate"] == 100).all()


# ---------------------------------------------------------------- 같은 조건 안 짝 비교

def test_pair_concordance_counts_only_pairs_inside_a_group():
    score = [3, 2, 1, 10, 20]
    life = [100, 200, 300, 500, 400]
    group = ["a", "a", "a", "b", "b"]
    assert sc.pair_concordance(score, life, group) == (pytest.approx(100.0), 4)
    reversed_life = [300, 200, 100, 400, 500]   # 점수가 높은 셀이 오히려 오래 감
    assert sc.pair_concordance(score, reversed_life, group) == (pytest.approx(0.0), 4)


def test_pair_concordance_skips_ties_and_missing_scores():
    assert sc.pair_concordance([1, 2, np.nan], [100, 100, 50], ["a", "a", "a"]) == (pytest.approx(np.nan, nan_ok=True), 0)


def test_shuffled_concordance_centres_on_a_coin_flip():
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(20), 5)
    null = sc.shuffled_concordance(rng.normal(size=100), rng.uniform(100, 1000, 100), group, repeats=300)
    assert 45 < np.median(null) < 55


# ---------------------------------------------------------------- 문서까지 한 번에

def write_fake_processed(folder, n_cells: int = 48, n_cycles: int = 101) -> None:
    """data/processed와 같은 모양의 작은 가짜 폴더 (cells.csv, summary.csv, curves/, raw/)."""
    (folder / "curves").mkdir(parents=True)
    (folder / "raw").mkdir()
    rng = np.random.default_rng(3)
    cells, summaries = [], []
    for i in range(n_cells):
        batch = f"b{i % 3 + 1}"
        cid = f"{batch}c{i // 3}"
        life = float(rng.choice([300, 420, 500, 650, 800, 1000, 1400]) * rng.uniform(0.95, 1.05))
        split = "secondary_test" if batch == "b3" else ("train" if (i // 3) % 2 else "test")
        cells.append({"cell_id": cid, "batch": batch, "policy": f"{i // 6}C(80%)-4C", "cycle_life": round(life),
                      "n_cycles": n_cycles, "split": split, "merged_from": "", "seam_cycle": ""})
        np.savez(folder / "curves" / f"{cid}.npz", **fake_curves(life, n_cycles, seed=i))
        s = fake_summary(n_cycles, placeholder=batch == "b1", seed=i)
        s.insert(0, "cell_id", cid)
        summaries.append(s)
        np.savez(folder / "raw" / f"{cid}.npz", raw_index=np.array([1, 9]), raw_offsets=np.array([0, 3, 6]),
                 raw_t=np.array([0, 20, 50, 0, 25, 54.0]))
    pd.DataFrame(cells).to_csv(folder / "cells.csv", index=False)
    pd.concat(summaries).to_csv(folder / "summary.csv", index=False)


def test_build_writes_markdown_figures_and_csvs(tmp_path):
    processed, docs = tmp_path / "processed", tmp_path / "docs"
    write_fake_processed(processed)
    out = report.build(processed, docs, repeats=2, shuffles=50, progress=lambda *_: None)

    text = (docs / "screening.md").read_text(encoding="utf-8")
    for section in ("## 질문", "## 왜 중요한가", "## 방법", "## 결과", "## 읽는 법", "## 한계", "## 시행착오"):
        assert section in text
    assert report.PAPER["pdf"] in text and "550" in text
    for name in ("signal", "cycles", "tradeoff", "confusion"):
        assert (docs / "figures" / f"screening_{name}.png").stat().st_size > 10_000
    for name in ("features", "runs", "tradeoff", "paper", "within"):
        assert (processed / f"screening_{name}.csv").exists()
    assert out["threshold"] == 550 and out["minutes_per_cycle"] == pytest.approx(52.0)
    assert set(out["random_by_cycles"]) == set(sc.CYCLES)
