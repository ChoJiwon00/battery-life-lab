"""명령줄 진입점.

    python -m battery_lab info                 데이터 상태 보기
    python -m battery_lab download [--yes]     데이터 내려받기 (--yes가 있어야 실제로 받음)
    python -m battery_lab extract              .mat에서 셀마다 값 추리기
    python -m battery_lab assemble             논문 규칙대로 124개 셀 데이터셋 만들기
"""

import argparse
import sys

from . import download as dl
from .assemble import assemble
from .extract import batch_keys, extract_batch
from .sources import BATCH_BY_KEY, BATCHES, INTERIM, LICENSE, PROCESSED, PROJECT_URL, RAW


def _gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB"


def _pick(keys) -> list:
    if not keys:
        return list(BATCHES)
    unknown = [k for k in keys if k not in BATCH_BY_KEY]
    if unknown:
        sys.exit(f"모르는 배치: {', '.join(unknown)} (b1, b2, b3 중에서 고르세요)")
    return [BATCH_BY_KEY[k] for k in keys]


def cmd_info(args) -> None:
    print(f"출처: {PROJECT_URL} (라이선스 {LICENSE})")
    for b in BATCHES:
        state, got = dl.status(b)
        extracted = len(list((INTERIM / b.key).glob(f"{b.key}c*.npz"))) if (INTERIM / b.key).exists() else 0
        extra = f" {got * 100 // b.size}%" if state == "받는 중" else ""
        print(f"  {b.key} {b.date}  {_gb(b.size):>8}  {state}{extra}  · 추린 셀 {extracted}개")
    print(f"전체 {_gb(sum(b.size for b in BATCHES))} · 남은 디스크 {_gb(dl.free_bytes())}")
    if (PROCESSED / "cells.csv").exists():
        print(f"데이터셋: {PROCESSED / 'cells.csv'}")


def cmd_download(args) -> None:
    batches = _pick(args.batch)
    todo = [b for b in batches if dl.status(b)[0] != "완료"]
    need = sum(b.size - dl.status(b)[1] for b in todo)
    for b in todo:
        print(f"  {b.key} {b.date}  {b.filename}  {_gb(b.size)}")
    if not todo:
        print("모두 받아 두었어요.")
        return
    free = dl.free_bytes()
    print(f"받을 양 {_gb(need)} · 남은 디스크 {_gb(free)}")
    if need > free:
        sys.exit("디스크 공간이 모자라요.")
    if not args.yes:
        print("실제로 받으려면 --yes 를 붙여 다시 실행하세요.")
        return
    for b in todo:
        path = dl.download(b)
        print(f"완료: {path}")


def cmd_extract(args) -> None:
    for b in _pick(args.batch):
        path = RAW / b.filename
        if dl.status(b)[0] != "완료":
            print(f"{b.key} {b.date}: 아직 다 받지 않아서 건너뛰어요.")
            continue
        print(f"{b.key} {b.date}: batch 안의 항목 = {', '.join(batch_keys(path))}")
        result = extract_batch(path, b.key, INTERIM, force=args.force)
        print(f"{b.key}: 셀 {result['cells']}개 중 새로 저장 {result['saved']}개, 건너뜀 {result['skipped']}개")


def cmd_assemble(args) -> None:
    counts = assemble(INTERIM, PROCESSED)
    print(f"셀 {counts['cells']}개 → {PROCESSED}")
    print("  배치별: " + ", ".join(f"{k} {v}" for k, v in sorted(counts["by_batch"].items())))
    print("  나누기: " + ", ".join(f"{k} {v}" for k, v in sorted(counts["by_split"].items())))
    if counts["missing_carryover"]:
        print("  이어 붙일 셀이 없어 못 붙임: " + ", ".join(counts["missing_carryover"]))
    if counts["cells"] == 124:
        print("  논문과 같은 124개예요 (학습 41 · 테스트 43 · 2차 테스트 40).")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="battery_lab", description="배터리 수명 예측·열화 분석")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="데이터 상태 보기").set_defaults(func=cmd_info)
    p = sub.add_parser("download", help="데이터 내려받기")
    p.add_argument("--batch", nargs="*", help="b1 b2 b3 중 일부만")
    p.add_argument("--yes", action="store_true", help="확인 없이 실제로 내려받기")
    p.set_defaults(func=cmd_download)
    p = sub.add_parser("extract", help=".mat에서 셀마다 값 추리기")
    p.add_argument("--batch", nargs="*")
    p.add_argument("--force", action="store_true", help="이미 추린 셀도 다시")
    p.set_defaults(func=cmd_extract)
    sub.add_parser("assemble", help="논문 규칙대로 데이터셋 만들기").set_defaults(func=cmd_assemble)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
