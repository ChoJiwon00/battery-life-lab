"""데이터 내려받기. 끊기면 이어받고, 다 받으면 크기가 맞는지 확인한다."""

import shutil
import urllib.request
from pathlib import Path

from .sources import RAW, Batch

CHUNK = 1 << 20  # 1MB씩 읽는다


class DownloadError(RuntimeError):
    pass


def status(batch: Batch, folder: Path = RAW) -> tuple[str, int]:
    """("완료" | "받는 중" | "없음", 받은 바이트)"""
    done = folder / batch.filename
    if done.exists():
        return ("완료" if done.stat().st_size == batch.size else "크기 다름"), done.stat().st_size
    part = done.with_name(done.name + ".part")
    if part.exists():
        return "받는 중", part.stat().st_size
    return "없음", 0


def free_bytes(folder: Path = RAW) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(folder).free


def download(batch: Batch, folder: Path = RAW, progress=print, opener=urllib.request.urlopen) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / batch.filename
    if dest.exists() and dest.stat().st_size == batch.size:
        progress(f"{batch.date}: 이미 받아 둠")
        return dest

    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with opener(urllib.request.Request(batch.url, headers=headers), timeout=60) as resp:
        if start and getattr(resp, "status", 200) != 206:
            start = 0  # 서버가 이어받기를 안 받아 주면 처음부터
        done, shown = start, -1
        with open(part, "ab" if start else "wb") as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                pct = done * 100 // batch.size
                if pct != shown and pct % 5 == 0:
                    shown = pct
                    progress(f"{batch.date}: {pct}% ({done / 1e9:.2f} / {batch.size / 1e9:.2f} GB)")

    got = part.stat().st_size
    if got != batch.size:
        raise DownloadError(f"{batch.date}: 크기가 다릅니다 ({got} / {batch.size} 바이트). 다시 실행하면 이어받아요.")
    part.replace(dest)
    return dest
