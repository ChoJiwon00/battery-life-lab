"""1단계 데이터 준비(내려받기 → 추리기 → 데이터셋 만들기) 테스트."""

import csv
import io

import numpy as np
import pytest

from battery_lab import download as dl
from battery_lab.assemble import assemble, load_interim
from battery_lab.extract import extract_batch, raw_indices
from battery_lab.sources import BATCHES, PAPER_RULES, Batch, Rules, select

from conftest import POINTS


def test_extract_reads_values_through_references(fake_mat, tmp_path):
    mat = fake_mat("b1.mat", [{"life": 150, "policy": "6C(40%)-3C", "n": 12},
                              {"life": float("nan"), "policy": "4C(80%)-4C", "n": 3, "raw_len": 4}])

    result = extract_batch(mat, "b1", tmp_path / "interim", progress=lambda _: None)

    assert result == {"cells": 2, "saved": 2, "skipped": 0}
    cells = load_interim(tmp_path / "interim")
    a, b = cells["b1c0"], cells["b1c1"]
    assert str(a["policy"]) == "6C(40%)-3C"
    assert float(a["cycle_life"]) == 150 and np.isnan(b["cycle_life"])
    assert a["Qdlin"].shape == (12, POINTS) and a["Qdlin"].dtype == np.float32
    assert a["summary_QD"][0] == pytest.approx(1.1)
    assert list(a["summary_cycle"]) == list(range(1, 13))
    assert list(a["raw_index"]) == [0, 9, 11]
    assert list(a["raw_offsets"]) == [0, 5, 10, 15]
    assert list(b["raw_index"]) == [0, 2]  # 사이클이 3개뿐이면 처음과 마지막만
    assert list(b["raw_offsets"]) == [0, 4, 8]


def test_extract_skips_cells_already_saved(fake_mat, tmp_path):
    mat = fake_mat("b1.mat", [{"life": 150, "policy": "x", "n": 2}])
    extract_batch(mat, "b1", tmp_path, progress=lambda _: None)

    assert extract_batch(mat, "b1", tmp_path, progress=lambda _: None)["skipped"] == 1
    assert extract_batch(mat, "b1", tmp_path, force=True, progress=lambda _: None)["saved"] == 1


def test_raw_indices_keep_first_tenth_every_hundredth_and_last():
    assert raw_indices(1) == [0]
    assert raw_indices(250) == [0, 9, 99, 199, 249]


def test_paper_rules_give_124_cells_and_the_paper_split():
    ids = [f"b1c{i}" for i in range(46)] + [f"b2c{i}" for i in range(48)] + [f"b3c{i}" for i in range(46)]

    order, split = select(ids, PAPER_RULES)

    assert len(order) == 124
    assert sum(c.startswith("b1") for c in order) == 41
    assert sum(c.startswith("b2") for c in order) == 43
    counts = {s: list(split.values()).count(s) for s in ("train", "test", "secondary_test")}
    assert counts == {"train": 41, "test": 43, "secondary_test": 40}
    assert order[:3] == ["b1c0", "b1c1", "b1c2"] and order[9] == "b1c11"  # c8, c10 빠짐 · 숫자 순서
    assert split[order[83]] == "test"  # 1·2배치의 마지막 셀은 테스트


def test_assemble_merges_carryover_removes_cells_and_writes_files(fake_mat, tmp_path):
    interim = tmp_path / "interim"
    b1 = fake_mat("b1.mat", [{"life": 10, "policy": f"p{i}", "n": 10} for i in range(5)])
    b2 = fake_mat("b2.mat", [{"life": 20, "policy": f"q{i}", "n": 4} for i in range(3)])
    b3 = fake_mat("b3.mat", [{"life": 30, "policy": f"r{i}", "n": 6} for i in range(2)])
    for key, mat in (("b1", b1), ("b2", b2), ("b3", b3)):
        extract_batch(mat, key, interim, progress=lambda _: None)
    rules = Rules(removed=("b1c4", "b3c1"), carryover=(("b2c2", "b1c0", 7),))

    counts = assemble(interim, tmp_path / "out", rules)

    assert counts["cells"] == 7  # b1c0~3, b2c0~1, b3c0
    rows = list(csv.DictReader(open(tmp_path / "out" / "cells.csv", encoding="utf-8")))
    assert [r["cell_id"] for r in rows] == ["b1c0", "b1c1", "b1c2", "b1c3", "b2c0", "b2c1", "b3c0"]
    first = rows[0]
    assert (first["cycle_life"], first["n_cycles"], first["merged_from"], first["seam_cycle"]) == ("17", "14", "b2c2", "11")
    assert rows[1]["seam_cycle"] == ""
    # 1·2배치 6개: 0·2·4번째와 마지막(5번째)은 테스트, 1·3번째는 학습
    assert [r["split"] for r in rows] == ["test", "train", "test", "train", "test", "test", "secondary_test"]

    summary = list(csv.DictReader(open(tmp_path / "out" / "summary.csv", encoding="utf-8")))
    merged = [int(float(r["cycle"])) for r in summary if r["cell_id"] == "b1c0"]
    assert merged == list(range(1, 11)) + list(range(11, 15))  # 이어 붙인 사이클 번호가 이어진다

    with np.load(tmp_path / "out" / "curves" / "b1c0.npz") as z:
        assert z["Qdlin"].shape == (14, POINTS)
    with np.load(tmp_path / "out" / "raw" / "b1c0.npz") as z:
        assert list(z["raw_index"]) == [0, 9, 10, 13]  # 이어 붙인 쪽 위치는 10만큼 밀린다


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int):
        super().__init__(body)
        self.status = status


def test_download_resumes_from_partial_file(tmp_path):
    body = bytes(range(256)) * 40
    batch = Batch("b9", "2000-01-01", "id", "fake.mat", len(body))
    (tmp_path / "fake.mat.part").write_bytes(body[:1000])
    seen = {}

    def opener(req, timeout):
        seen["range"] = req.get_header("Range")
        return FakeResponse(body[1000:], 206)

    path = dl.download(batch, tmp_path, progress=lambda _: None, opener=opener)

    assert seen["range"] == "bytes=1000-"
    assert path.read_bytes() == body
    assert dl.status(batch, tmp_path) == ("완료", len(body))


def test_download_rejects_wrong_size(tmp_path):
    batch = Batch("b9", "2000-01-01", "id", "fake.mat", 100)
    with pytest.raises(dl.DownloadError):
        dl.download(batch, tmp_path, progress=lambda _: None, opener=lambda req, timeout: FakeResponse(b"x" * 50, 200))
    assert dl.status(batch, tmp_path)[0] == "받는 중"


def test_batch_sizes_match_what_the_server_reported():
    assert sum(b.size for b in BATCHES) == 8_269_341_808


def test_cli_help_lists_every_command(capsys):
    from battery_lab.__main__ import EXPERIMENTS, main

    with pytest.raises(SystemExit):
        main(["--help"])  # 도움말의 %가 서식 기호로 읽혀 오류가 났던 적이 있다
    out = capsys.readouterr().out
    for name in ("info", "download", "extract", "assemble", "report", "sizes", *EXPERIMENTS):
        assert name in out
