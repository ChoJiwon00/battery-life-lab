"""MATLAB .mat(v7.3, 속은 HDF5) 배치 파일에서 셀마다 필요한 값만 추려 작은 .npz로 저장한다.

.mat 구조는 저자 코드(BuildPkl_Batch1.ipynb)를 따랐다.
batch/{cycle_life, policy_readable, summary, cycles}는 셀마다 HDF5 참조(reference)를 담고 있고,
참조를 따라가야 실제 값이 나온다.

저장하는 것 (셀 하나 = 파일 하나, data/interim/<배치>/<셀>.npz)
- 요약: 사이클마다 한 값 (방전·충전 용량, 내부저항, 온도, 충전 시간)
- 곡선: 사이클마다 전압 구간을 고르게 나눈 방전 용량(Qdlin), dQ/dV, 온도(Tdlin) → (사이클 수, 칸 수)
- 원시 데이터: 몇몇 사이클만 시간·전압·전류·용량·온도 전체 (그래프 그리기용)
"""

from pathlib import Path

import h5py
import numpy as np

SUMMARY_FIELDS = {"cycle": "cycle", "QDischarge": "QD", "QCharge": "QC", "IR": "IR",
                  "Tavg": "Tavg", "Tmin": "Tmin", "Tmax": "Tmax", "chargetime": "chargetime"}
CURVE_FIELDS = {"Qdlin": "Qdlin", "discharge_dQdV": "dQdV", "Tdlin": "Tdlin"}
RAW_FIELDS = ("t", "V", "I", "Qc", "Qd", "T")


def _array(f: h5py.File, ref) -> np.ndarray:
    return np.asarray(f[ref][()]).ravel()


def _text(f: h5py.File, ref) -> str:
    # MATLAB 글자는 2바이트(UTF-16)라 한 바이트씩 건너뛰어 읽는다 (저자 코드와 같다).
    return np.asarray(f[ref][()]).tobytes()[::2].decode()


def raw_indices(n_cycles: int) -> list[int]:
    """원시 데이터를 남길 사이클 위치: 처음, 10번째, 100번째마다, 마지막."""
    return sorted(i for i in {0, 9, n_cycles - 1, *range(99, n_cycles, 100)} if 0 <= i < n_cycles)


def batch_keys(mat_path: Path) -> list[str]:
    with h5py.File(mat_path, "r") as f:
        return sorted(f["batch"].keys())


def _curves(f: h5py.File, cycles: h5py.Group, name: str, n: int) -> np.ndarray:
    rows = [_array(f, cycles[name][j, 0]) for j in range(n)]
    width = max((len(r) for r in rows), default=0)
    out = np.full((n, width), np.nan, dtype=np.float32)
    for j, r in enumerate(rows):
        if len(r) == width:  # 비어 있거나 길이가 다른 사이클은 빈칸(NaN)으로 둔다
            out[j] = r
    return out


def read_cell(f: h5py.File, i: int) -> dict[str, np.ndarray]:
    batch = f["batch"]
    data: dict[str, np.ndarray] = {
        "cycle_life": np.float64(_array(f, batch["cycle_life"][i, 0])[0]),
        "policy": np.array(_text(f, batch["policy_readable"][i, 0])),
    }
    if "Vdlin" in batch:  # Qdlin의 전압 눈금 (3.5V → 2.0V, 1000칸)
        data["Vdlin"] = _array(f, batch["Vdlin"][i, 0]).astype(np.float64)
    # barcode·channel_id는 MATLAB string 객체(uint32)로 저장돼 있어 읽지 않는다.

    summary = f[batch["summary"][i, 0]]
    for src, dst in SUMMARY_FIELDS.items():
        data[f"summary_{dst}"] = np.asarray(summary[src][()], dtype=np.float64).ravel()

    cycles = f[batch["cycles"][i, 0]]
    n = cycles["Qdlin"].shape[0]
    for src, dst in CURVE_FIELDS.items():
        data[dst] = _curves(f, cycles, src, n)

    picks = raw_indices(n)
    parts = []
    for j in picks:
        cols = {name: _array(f, cycles[name][j, 0]).astype(np.float64) for name in RAW_FIELDS}
        m = min(len(v) for v in cols.values())  # 한 사이클의 열들은 길이가 같아야 한다. 다르면 짧은 쪽에 맞춘다.
        parts.append({name: v[:m] for name, v in cols.items()})
    data["raw_index"] = np.array(picks, dtype=np.int32)
    data["raw_offsets"] = np.cumsum([0, *(len(p["t"]) for p in parts)]).astype(np.int64)
    for name in RAW_FIELDS:
        data[f"raw_{name}"] = np.concatenate([p[name] for p in parts]) if parts else np.empty(0)
    return data


def extract_batch(mat_path: Path, batch_key: str, out_dir: Path, force: bool = False, progress=print) -> dict:
    """배치 파일 하나를 셀마다 .npz로 저장한다. 이미 있는 셀은 건너뛴다(force면 다시)."""
    folder = out_dir / batch_key
    folder.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0
    with h5py.File(mat_path, "r") as f:
        n_cells = f["batch"]["summary"].shape[0]
        for i in range(n_cells):
            path = folder / f"{batch_key}c{i}.npz"
            if path.exists() and not force:
                skipped += 1
                continue
            data = read_cell(f, i)
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(tmp, **data)
            tmp.replace(path)
            saved += 1
            progress(f"{batch_key}c{i}: 사이클 {data['Qdlin'].shape[0]}개 · 수명 {data['cycle_life']:g} · {data['policy']}")
    return {"cells": n_cells, "saved": saved, "skipped": skipped}
