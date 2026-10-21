"""배치별로 추린 셀(data/interim)을 논문 규칙대로 고르고 이어 붙여 하나의 데이터셋(data/processed)으로 만든다.

만드는 것
- cells.csv    셀마다 한 줄: 배치, 충전 조건, 수명, 사이클 수, 학습/테스트 구분, 이어 붙인 지점(seam_cycle)
- summary.csv  사이클마다 한 줄: 방전·충전 용량, 내부저항, 온도, 충전 시간
- curves/<셀>.npz  사이클별 Qdlin · dQdV · Tdlin
- raw/<셀>.npz     몇몇 사이클의 원시 데이터
"""

import csv
from pathlib import Path

import numpy as np

from .extract import CURVE_FIELDS, RAW_FIELDS, SUMMARY_FIELDS
from .sources import PAPER_RULES, Rules, cell_order, select

CURVES = tuple(CURVE_FIELDS.values())


def load_interim(folder: Path) -> dict[str, dict]:
    cells = {}
    for path in sorted(folder.glob("b*/b*c*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path, allow_pickle=False) as z:
            cells[path.stem] = {k: z[k] for k in z.files}
    return cells


def _raw_parts(cell: dict) -> list[dict[str, np.ndarray]]:
    o = cell["raw_offsets"]
    return [{name: cell[f"raw_{name}"][o[k]:o[k + 1]] for name in RAW_FIELDS} for k in range(len(o) - 1)]


def merge(dst: dict, src: dict, add_life: int) -> dict:
    """다음 배치에서 이어 시험한 셀(src)을 원래 셀(dst) 뒤에 붙인다. 저자 코드와 같은 방식."""
    out = dict(dst)
    out["cycle_life"] = np.float64(dst["cycle_life"] + add_life)
    n_summary = len(dst["summary_cycle"])
    for name in SUMMARY_FIELDS.values():
        extra = src[f"summary_{name}"] + (n_summary if name == "cycle" else 0)
        out[f"summary_{name}"] = np.concatenate([dst[f"summary_{name}"], extra])

    n_cycles = dst["Qdlin"].shape[0]
    for name in CURVES:
        a, b = dst[name], src[name]
        width = max(a.shape[1], b.shape[1])
        pad = lambda m: np.pad(m, ((0, 0), (0, width - m.shape[1])), constant_values=np.nan)
        out[name] = np.vstack([pad(a), pad(b)])

    parts = _raw_parts(dst) + _raw_parts(src)
    out["raw_index"] = np.concatenate([dst["raw_index"], src["raw_index"] + n_cycles]).astype(np.int32)
    for name in RAW_FIELDS:
        out[f"raw_{name}"] = np.concatenate([p[name] for p in parts]) if parts else np.empty(0)
    out["raw_offsets"] = np.cumsum([0, *(len(p["t"]) for p in parts)]).astype(np.int64)
    out["merged_from"] = np.array(src.get("_id", ""))
    out["seam_cycle"] = np.float64(n_summary + src["summary_cycle"][0])  # 이어 붙인 쪽의 첫 사이클
    return out


def assemble(interim: Path, out_dir: Path, rules: Rules = PAPER_RULES, cells: dict | None = None) -> dict:
    cells = dict(cells if cells is not None else load_interim(interim))
    for cid, cell in cells.items():
        cell["_id"] = cid

    for src, dst, add in rules.carryover:
        if src in cells and dst in cells:
            cells[dst] = merge(cells[dst], cells[src], add)

    order, split = select(cells, rules)

    (out_dir / "curves").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cells.csv", "w", newline="", encoding="utf-8") as fc, \
         open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fs:
        wc, ws = csv.writer(fc), csv.writer(fs)
        wc.writerow(["cell_id", "batch", "policy", "cycle_life", "n_cycles", "split", "merged_from", "seam_cycle"])
        ws.writerow(["cell_id", *SUMMARY_FIELDS.values()])
        for cid in order:
            c = cells[cid]
            life = float(c["cycle_life"])
            wc.writerow([cid, cid[:2], str(c["policy"]), "" if np.isnan(life) else f"{life:g}",
                         c["Qdlin"].shape[0], split[cid], str(c.get("merged_from", "")),
                         f"{float(c['seam_cycle']):g}" if "seam_cycle" in c else ""])
            columns = [c[f"summary_{name}"] for name in SUMMARY_FIELDS.values()]
            for row in zip(*columns):
                ws.writerow([cid, *(f"{v:.6g}" for v in row)])
            curves = {name: c[name] for name in CURVES}
            if "Vdlin" in c:
                curves["Vdlin"] = c["Vdlin"]
            curves["cycle"] = c["summary_cycle"][: c["Qdlin"].shape[0]]  # 행마다 사이클 번호 (행 j = 요약 j번째)
            np.savez_compressed(out_dir / "curves" / f"{cid}.npz", **curves)
            np.savez_compressed(out_dir / "raw" / f"{cid}.npz",
                                **{k: c[k] for k in ("raw_index", "raw_offsets", *(f"raw_{n}" for n in RAW_FIELDS))})

    counts = {"cells": len(order), "by_batch": {}, "by_split": {}}
    for cid in order:
        counts["by_batch"][cid[:2]] = counts["by_batch"].get(cid[:2], 0) + 1
        counts["by_split"][split[cid]] = counts["by_split"].get(split[cid], 0) + 1
    missing = sorted({d for _, d, _ in rules.carryover if d not in cells} |
                     {s for s, _, _ in rules.carryover if s not in cells}, key=cell_order)
    counts["missing_carryover"] = missing
    return counts
