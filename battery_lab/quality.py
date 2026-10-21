"""데이터 품질 점검과 논문 주장 검증.

data/processed(assemble 결과)를 읽어 셀마다 이상한 점을 찾고, 논문이 말한 숫자를 우리 데이터로 다시 계산한다.
판단 기준을 정한 과정과 처음에 틀렸던 기준은 docs/troubleshooting.md에 적어 두었다.

셀마다 보는 것
- 빈 자리: b1 배치는 1번째 사이클 기록이 없어 0으로 채운 줄이 있다 (공식 설명). 이상값으로 세지 않고 따로 센다.
- 수명 라벨: cycle_life가 '용량이 처음 0.88Ah(공칭의 80%) 미만이 된 사이클'과 맞는가.
  시험이 0.88Ah에 닿자마자 멈춘 셀은 그 사이클이 기록에 없어서 '마지막 기록 + 1'이 수명이다.
- 회복 튐: '처지기 전 수준'(직전 20사이클, 자연 감소분 반영)보다 0.5% 넘게 오르고 유지되는 곳 (쉬었다 재개한 뒤 회복)
- 이음매: 다음 배치에서 이어 시험한 셀은 이어 붙인 지점 앞뒤로 용량이 얼마나 바뀌었나
- 순간 하락: 주변 11사이클 중앙값보다 1% 넘게 꺼졌다가 돌아오는 구간 (측정 이상)
- 범위 밖 값: 용량 0.1Ah 이하·1.2Ah 이상, 온도 20°C 미만·50°C 초과, 내부저항 0·중앙값의 3배 초과
- 긴 충전 시간: 60분 초과 (시험이 충전 단계에서 멈췄다 재개되면 쉰 시간까지 충전 시간에 들어간다)
- 방전 곡선(Qdlin): 빈 곡선, 말이 안 되는 값(-0.01Ah 미만·1.3Ah 초과), 작은 되돌림(전압이 내려가는데 1mAh 넘게 줄어듦)
- 수명 예측에 쓰는 10·100번째 사이클 곡선이 깨끗한가
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

NOMINAL_AH = 1.1
EOL_AH = 0.88               # 공칭 용량의 80%
STOP_MARGIN = 1.005         # 마지막 기록이 0.88Ah의 0.5% 안이면 '0.88에서 시험 종료'로 본다
JUMP_UP = 0.005             # 회복 튐 기준 0.5%
DIP = 0.01                  # 순간 하락 기준 1%
BASELINE_DIP = 0.004        # 회복 튐의 비교 기준을 잡을 때는 0.4% 하락부터 뺀다
ENVELOPE = 20               # 회복 튐의 비교 기준: 직전 20사이클
QD_RANGE = (0.1, 1.2)       # Ah
T_RANGE = (20.0, 50.0)      # °C. 시험 챔버는 30°C
IR_RATIO = 3.0
LONG_CHARGE = 60.0          # 분. 0→80% 급속충전 + 1C CC-CV는 보통 9~20분, 늙으면 정전압 구간이 길어진다
CURVE_RANGE = (-0.01, 1.3)  # Ah
BACKSTEP = 1e-3             # 1mAh
FEATURE_CYCLES = (10, 100)
FEATURE_BACKSTEP = 5e-3     # 예측용 곡선은 5mAh 넘는 되돌림이 없어야 깨끗하다고 본다


@dataclass
class CellReport:
    cell_id: str
    batch: str
    policy: str
    split: str
    cycle_life: float
    n_cycles: int
    placeholder_rows: int = 0
    qd_start: float = np.nan        # 사이클 2의 방전 용량
    qd_100: float = np.nan
    change_100_pct: float = np.nan  # 사이클 2 → 100 용량 변화(%)
    eol_observed: float = np.nan
    eol_rule: str = ""              # "first_below" | "stopped_at_eol" | ""
    recovery_cycles: list[int] = field(default_factory=list)  # 회복 튐이 시작된 사이클
    dip_cycles: list[int] = field(default_factory=list)       # 순간 하락이 시작된 사이클
    bad_qd_cycles: list[int] = field(default_factory=list)
    long_charge_cycles: list[int] = field(default_factory=list)
    bad_temp: int = 0
    zero_ir: int = 0
    high_ir: int = 0
    empty_curves: int = 0
    broken_curve_cycles: list[int] = field(default_factory=list)
    small_backsteps: int = 0
    feature_clean: bool = True
    seam_cycle: float = np.nan        # 다음 배치에서 이어 붙인 첫 사이클
    seam_first_pct: float = np.nan    # 이음매 직전 5사이클 중앙값 대비 재개 첫 사이클(%)
    seam_after10_pct: float = np.nan  # 같은 기준 대비 재개 뒤 10사이클 중앙값(%)
    notes: list[str] = field(default_factory=list)

    @property
    def life_gap(self) -> float:
        return self.cycle_life - self.eol_observed


def _runs(mask: np.ndarray) -> list[int]:
    """True가 이어지는 구간마다 시작 위치."""
    m = np.asarray(mask, dtype=bool)
    return [int(i) for i in np.flatnonzero(m & ~np.concatenate([[False], m[:-1]]))]


def _forward_median(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x[::-1]).rolling(window, min_periods=1).median().to_numpy()[::-1]


def find_eol(qd: np.ndarray, cycles: np.ndarray) -> tuple[float, str]:
    """용량이 처음 0.88Ah 미만이 된 사이클. 순간 하락에 속지 않도록 앞으로 5사이클 중앙값도 미만이어야 한다."""
    if not len(qd):
        return np.nan, ""
    below = (qd < EOL_AH) & (_forward_median(qd, 5) < EOL_AH)
    if below.any():
        return float(cycles[np.argmax(below)]), "first_below"
    last = qd[np.isfinite(qd)]
    if last.size and last[-1] < EOL_AH * STOP_MARGIN:
        return float(cycles[-1] + 1), "stopped_at_eol"
    return np.nan, ""


def capacity_events(qd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(회복 튐 위치, 순간 하락 위치) — 둘 다 True/False 배열."""
    s = pd.Series(qd)
    center = s.rolling(11, center=True, min_periods=3).median().to_numpy()
    with np.errstate(invalid="ignore"):
        dip = qd < center * (1 - DIP)
        shallow = qd < center * (1 - BASELINE_DIP)
    clean = np.where(dip, np.nan, qd)
    # 비교 기준 = '처지기 전 수준': 직전 20사이클 값들을 최근 감소 속도만큼 지금 시점으로 옮긴 뒤의 최댓값.
    # 기준을 이렇게 정하기까지 세 번 틀렸다 (troubleshooting.md 참고).
    # - 직전 10사이클 중앙값: 순간 하락이 잦은 구간에서 기준이 같이 끌려 내려가, 정상값이 튐처럼 보였다.
    # - 직전 10사이클 최댓값: 빨리 늙는 셀은 그 사이 이미 0.7%쯤 줄어 있어 1% 튐도 놓쳤다.
    # - 직전 5사이클 최댓값: 몇 사이클 처졌다가 원래대로 돌아오는 것을 튐으로 봤다.
    base = np.where(shallow, np.nan, qd)
    trail = pd.Series(base).rolling(31, min_periods=10).median().to_numpy()  # 지난 값만 쓰는 추세
    envelope = np.full(len(qd), np.nan)
    for k in range(1, len(qd)):
        lo = max(0, k - ENVELOPE)
        seg = base[lo:k]
        if np.isfinite(seg).sum() < 2:
            continue
        slope = 0.0
        if k > ENVELOPE and np.isfinite(trail[k - 1]) and np.isfinite(trail[k - 1 - ENVELOPE]):
            slope = min(0.0, (trail[k - 1] - trail[k - 1 - ENVELOPE]) / ENVELOPE)  # 오르는 추세로는 기준을 올리지 않는다
        envelope[k] = np.nanmax(seg + slope * (k - np.arange(lo, k)))
    ahead = _forward_median(clean, 3)
    with np.errstate(invalid="ignore"):
        jump = (clean > envelope * (1 + JUMP_UP)) & (ahead > envelope)
    return jump, dip


def check_cell(meta: dict, s: pd.DataFrame, curves: dict[str, np.ndarray] | None) -> CellReport:
    r = CellReport(meta["cell_id"], meta["batch"], meta["policy"], meta["split"],
                   float(meta["cycle_life"]) if str(meta["cycle_life"]) != "" else np.nan, len(s))
    s = s.sort_values("cycle")
    placeholder = ((s["QD"] == 0) & (s["QC"] == 0) & (s["IR"] == 0) & (s["Tavg"] == 0)).to_numpy()
    r.placeholder_rows = int(placeholder.sum())
    empty_cycles = set(s["cycle"].to_numpy()[placeholder].astype(int))
    s = s[~placeholder]

    qd = s["QD"].to_numpy(dtype=float)
    cycles = s["cycle"].to_numpy(dtype=float)
    ok = (qd > QD_RANGE[0]) & (qd < QD_RANGE[1])
    r.bad_qd_cycles = [int(c) for c in cycles[~ok]]
    good = np.where(ok, qd, np.nan)

    by_cycle = dict(zip(cycles, qd))
    r.qd_start, r.qd_100 = by_cycle.get(2.0, np.nan), by_cycle.get(100.0, np.nan)
    if np.isfinite(r.qd_start) and np.isfinite(r.qd_100):
        r.change_100_pct = (r.qd_100 - r.qd_start) / r.qd_start * 100
    r.eol_observed, r.eol_rule = find_eol(good, cycles)

    jump, dip = capacity_events(good)
    r.recovery_cycles = [int(cycles[i]) for i in _runs(jump)]
    r.dip_cycles = [int(cycles[i]) for i in _runs(dip)]

    seam = str(meta.get("seam_cycle", "") or "")
    if seam:
        r.seam_cycle = float(seam)
        before = good[(cycles >= r.seam_cycle - 5) & (cycles < r.seam_cycle)]
        first = good[cycles == r.seam_cycle]
        after = good[(cycles >= r.seam_cycle) & (cycles < r.seam_cycle + 10)]
        if before.size and first.size and after.size:
            base = np.nanmedian(before)
            r.seam_first_pct = (first[0] / base - 1) * 100
            r.seam_after10_pct = (np.nanmedian(after) / base - 1) * 100

    charge = s["chargetime"].to_numpy(dtype=float)
    r.long_charge_cycles = [int(c) for c in cycles[charge > LONG_CHARGE]]

    for col in ("Tavg", "Tmax"):
        t = s[col].to_numpy(dtype=float)
        r.bad_temp += int(np.sum((t < T_RANGE[0]) | (t > T_RANGE[1])))
    ir = s["IR"].to_numpy(dtype=float)
    r.zero_ir = int(np.sum(ir <= 0))
    positive = ir[ir > 0]
    if positive.size:
        r.high_ir = int(np.sum(positive > IR_RATIO * np.median(positive)))

    if curves is not None and curves.get("Qdlin") is not None and curves["Qdlin"].size:
        q = curves["Qdlin"].astype(np.float64)
        row_cycles = np.asarray(curves.get("cycle", np.arange(1, len(q) + 1))).astype(int)
        keep = np.array([c not in empty_cycles for c in row_cycles])
        empty = np.all(~np.isfinite(q) | (q == 0), axis=1)
        broken = np.any((q < CURVE_RANGE[0]) | (q > CURVE_RANGE[1]), axis=1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 빈 곡선은 nanmin이 경고를 낸다
            backstep = np.nanmin(np.diff(q, axis=1), axis=1)
        r.empty_curves = int((empty & keep).sum())
        r.broken_curve_cycles = [int(c) for c in row_cycles[broken & keep]]
        r.small_backsteps = int(((backstep < -BACKSTEP) & ~broken & ~empty & keep).sum())
        rows = {int(c): j for j, c in enumerate(row_cycles)}
        r.feature_clean = all(
            k in rows and not empty[rows[k]] and not broken[rows[k]] and backstep[rows[k]] >= -FEATURE_BACKSTEP
            for k in FEATURE_CYCLES)

    _write_notes(r)
    return r


def _cycles_text(cycles: list[int], limit: int = 4) -> str:
    shown = ", ".join(str(c) for c in cycles[:limit])
    return shown + (f" 외 {len(cycles) - limit}곳" if len(cycles) > limit else "")


def _write_notes(r: CellReport) -> None:
    if np.isfinite(r.life_gap) and r.life_gap != 0:
        r.notes.append(f"수명 라벨({r.cycle_life:g})과 실제 80% 도달({r.eol_observed:g})이 {r.life_gap:+g}사이클 차이")
    if not np.isfinite(r.eol_observed):
        r.notes.append("80% 도달을 확인할 수 없음")
    if r.recovery_cycles:
        r.notes.append(f"회복 튐 {len(r.recovery_cycles)}번 (사이클 {_cycles_text(r.recovery_cycles)})")
    if r.dip_cycles:
        r.notes.append(f"순간 하락 {len(r.dip_cycles)}번")
    if r.bad_qd_cycles:
        r.notes.append(f"범위 밖 용량 (사이클 {_cycles_text(r.bad_qd_cycles)})")
    if r.long_charge_cycles:
        r.notes.append(f"충전 시간 60분 초과 (사이클 {_cycles_text(r.long_charge_cycles)})")
    if r.broken_curve_cycles:
        r.notes.append(f"방전 곡선에 말이 안 되는 값 (사이클 {_cycles_text(r.broken_curve_cycles)})")
    if r.bad_temp:
        r.notes.append(f"범위 밖 온도 {r.bad_temp}개")
    if r.zero_ir or r.high_ir:
        r.notes.append(f"내부저항 0 {r.zero_ir}개 · 중앙값 3배 초과 {r.high_ir}개")
    if r.empty_curves:
        r.notes.append(f"빈 방전 곡선 {r.empty_curves}개")
    if not r.feature_clean:
        r.notes.append("⚠ 수명 예측에 쓰는 10·100번째 사이클 곡선에 문제 있음")


def load_processed(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(folder / "cells.csv", dtype={"cycle_life": str, "merged_from": str, "seam_cycle": str},
                        keep_default_na=False)
    summary = pd.read_csv(folder / "summary.csv")
    return cells, summary


def load_curves(folder: Path, cell_id: str) -> dict[str, np.ndarray]:
    with np.load(folder / "curves" / f"{cell_id}.npz") as z:
        return {k: z[k] for k in z.files}


def check_all(folder: Path) -> list[CellReport]:
    cells, summary = load_processed(folder)
    groups = dict(tuple(summary.groupby("cell_id")))
    return [check_cell(meta, groups.get(meta["cell_id"], summary.iloc[:0]), load_curves(folder, meta["cell_id"]))
            for meta in cells.to_dict("records")]


def hotspots(reports: list[CellReport], width: int = 20) -> dict[str, dict]:
    """배치마다 이상(회복 튐·범위 밖 용량·깨진 곡선·긴 충전 시간)이 가장 많이 몰린 사이클 구간."""
    out = {}
    for batch in sorted({r.batch for r in reports}):
        cyc = [c for r in reports if r.batch == batch
               for c in (*r.recovery_cycles, *r.bad_qd_cycles, *r.broken_curve_cycles, *r.long_charge_cycles)]
        if not cyc:
            continue
        bins = pd.Series(np.asarray(cyc) // width * width).value_counts()
        start = int(bins.index[0])
        cells = sum(any(start <= c < start + width
                        for c in (*r.recovery_cycles, *r.bad_qd_cycles, *r.broken_curve_cycles, *r.long_charge_cycles))
                    for r in reports if r.batch == batch)
        out[batch] = {"start": start, "end": start + width - 1, "events": int(bins.iloc[0]), "total": len(cyc),
                      "cells": cells, "batch_cells": sum(r.batch == batch for r in reports)}
    return out


def alignment_check(folder: Path, cell_ids: list[str], summary: pd.DataFrame) -> dict:
    """곡선·원시 데이터의 j번째 사이클이 요약표 j번째 사이클과 같은지, 같은 출처의 값으로 확인한다.

    원시 방전 용량의 최댓값은 요약표 방전 용량과 같은 기록에서 나온 값이라, 행이 맞으면 거의 정확히 같아야 한다.
    (처음엔 곡선 최댓값과 비교했는데, 곡선은 정전류 구간만 담아 비교가 약했다 — troubleshooting.md 6번)
    """
    matched = total = skipped = 0
    worst = 0.0
    grid = None
    grids_same = 0
    for cid in cell_ids:
        qd = summary.loc[summary["cell_id"] == cid].sort_values("cycle")["QD"].to_numpy()
        with np.load(folder / "raw" / f"{cid}.npz") as z:
            o, idx, q = z["raw_offsets"], z["raw_index"], z["raw_Qd"]
        for k, j in enumerate(idx):
            part = q[o[k]:o[k + 1]]
            if not part.size or np.nanmax(part) < QD_RANGE[0] or j >= len(qd):
                skipped += 1  # b1의 빈 1번째 사이클 등
                continue
            total += 1
            err = abs(float(np.nanmax(part)) - qd[j])
            worst = max(worst, err)
            matched += err < 1e-3
        v = load_curves(folder, cid).get("Vdlin")
        if v is not None:
            grid = v if grid is None else grid
            grids_same += bool(np.allclose(v, grid))
    return {"matched": matched, "total": total, "skipped": skipped, "max_error_mah": worst * 1000,
            "grids_same": grids_same, "cells": len(cell_ids),
            "grid": (float(grid[0]), float(grid[-1]), len(grid)) if grid is not None else None}


def delta_q(curves: dict[str, np.ndarray], late: int = 100, early: int = 10) -> np.ndarray | None:
    """ΔQ_late-early(V) = (late번째 사이클 방전 곡선) − (early번째 사이클 방전 곡선). 사이클 번호로 찾는다."""
    rows = {int(c): j for j, c in enumerate(curves["cycle"])}
    if late not in rows or early not in rows:
        return None
    return curves["Qdlin"][rows[late]].astype(np.float64) - curves["Qdlin"][rows[early]]


def log_var_delta_q(curves: dict[str, np.ndarray], late: int = 100, early: int = 10) -> float:
    d = delta_q(curves, late, early)
    if d is None:
        return np.nan
    d = d[np.isfinite(d)]
    return float(np.log10(np.var(d))) if d.size else np.nan


def paper_checks(reports: list[CellReport], folder: Path, late: int = 100, early: int = 10) -> dict:
    """논문이 말한 숫자를 우리 데이터로 다시 계산한다."""
    life = np.array([r.cycle_life for r in reports])
    change = np.array([r.change_100_pct for r in reports])
    feature = np.array([log_var_delta_q(load_curves(folder, r.cell_id), late, early) for r in reports])
    ok = np.isfinite(feature) & np.isfinite(life) & (life > 0)
    rho = float(np.corrcoef(feature[ok], np.log10(life[ok]))[0, 1]) if ok.sum() > 2 else np.nan
    return {
        "cells": len(reports),
        "life_min": float(np.nanmin(life)), "life_max": float(np.nanmax(life)),
        "life_median": float(np.nanmedian(life)),
        "change_100_median_pct": float(np.nanmedian(change)),
        "logvar_corr": rho, "logvar_cells": int(ok.sum()),
        "feature": feature,
    }
