"""진짜 배치 .mat(MATLAB v7.3 = HDF5)과 같은 구조의 작은 가짜 파일을 만든다.

batch/{cycle_life, policy_readable, summary, cycles}에 셀마다 참조(reference)를 넣고,
summary는 필드마다 (1, 사이클 수) 배열, cycles는 필드마다 (사이클 수, 1) 참조 배열이다.
"""

import itertools
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POINTS = 20  # 진짜는 1000칸


def _utf16(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-16-le"), dtype=np.uint16).reshape(-1, 1)


def write_fake_mat(path: Path, cells: list[dict]) -> Path:
    """cells: [{"life": 수명, "policy": "6C(40%)-3C", "n": 사이클 수, "raw_len": 원시 길이}]"""
    counter = itertools.count()
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")

        def put(data):
            return refs.create_dataset(str(next(counter)), data=np.asarray(data)).ref

        life, policy, summary, cycles = [], [], [], []
        for c in cells:
            n, raw_len = c["n"], c.get("raw_len", 5)
            cap = 1.1 - 0.001 * np.arange(n)
            life.append(put(np.array([[c["life"]]], dtype=float)))
            policy.append(put(_utf16(c["policy"])))

            s = refs.create_group(f"s{next(counter)}")
            fields = {"cycle": np.arange(1, n + 1), "QDischarge": cap, "QCharge": cap + 0.001,
                      "IR": np.full(n, 0.016), "Tavg": np.full(n, 31.0), "Tmin": np.full(n, 30.0),
                      "Tmax": np.full(n, 34.0), "chargetime": np.full(n, 10.0)}
            for k, v in fields.items():
                s.create_dataset(k, data=np.asarray(v, dtype=float).reshape(1, -1))
            summary.append(s.ref)

            g = refs.create_group(f"c{next(counter)}")
            per_cycle = {
                "Qdlin": lambda j: np.linspace(0, cap[j], POINTS).reshape(1, -1),
                "discharge_dQdV": lambda j: np.full((1, POINTS), -j, dtype=float),
                "Tdlin": lambda j: np.full((1, POINTS), 30.0 + j),
                "t": lambda j: np.arange(raw_len, dtype=float).reshape(-1, 1) + j,
                "V": lambda j: np.linspace(3.6, 2.0, raw_len).reshape(-1, 1),
                "I": lambda j: np.full((raw_len, 1), -4.4),
                "Qc": lambda j: np.zeros((raw_len, 1)),
                "Qd": lambda j: np.linspace(0, cap[j], raw_len).reshape(-1, 1),
                "T": lambda j: np.full((raw_len, 1), 31.0),
            }
            for name, make in per_cycle.items():
                g.create_dataset(name, data=np.array([put(make(j)) for j in range(n)], dtype=h5py.ref_dtype).reshape(-1, 1))
            cycles.append(g.ref)

        batch = f.create_group("batch")
        for name, items in (("cycle_life", life), ("policy_readable", policy), ("summary", summary), ("cycles", cycles)):
            batch.create_dataset(name, data=np.array(items, dtype=h5py.ref_dtype).reshape(-1, 1))
    return path


@pytest.fixture
def fake_mat(tmp_path):
    def make(name: str, cells: list[dict]) -> Path:
        return write_fake_mat(tmp_path / name, cells)
    return make
