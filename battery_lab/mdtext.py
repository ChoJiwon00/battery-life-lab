"""생성 문서(마크다운)를 GitHub에서 보이는 그대로 쓰기.

GitHub 마크다운은 물결표 한 개로 감싼 글자도 취소선으로 그린다. 그래서 한 문단에 "10~90%"와 "12.8~14.9%"가
같이 있으면 두 ~ 사이가 줄이 그어지고 ~는 사라진다 (troubleshooting.md 40번).
범위를 뜻하는 ~는 모두 \\~로 바꿔 글자 그대로 보이게 한다. 코드(`...`, ``` 블록)와 링크 주소는 건드리지 않는다.
"""

import re
from pathlib import Path

_SKIP = re.compile(r"`[^`]*`|\]\([^)]*\)|<[^>]+>")   # 인라인 코드 · 링크 주소 · HTML 태그
_TILDE = re.compile(r"(?<![\\~])~(?!~)")             # 이미 \~이거나 ~~(일부러 쓴 취소선)인 것은 그대로


def _escape_line(line: str) -> str:
    out, pos = [], 0
    for m in _SKIP.finditer(line):
        out.append(_TILDE.sub(r"\\~", line[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_TILDE.sub(r"\\~", line[pos:]))
    return "".join(out)


def escape_tildes(text: str) -> str:
    lines, fenced = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            lines.append(line)
        else:
            lines.append(line if fenced else _escape_line(line))
    return "\n".join(lines)


def write_doc(path: Path, text: str) -> None:
    Path(path).write_text(escape_tildes(text), encoding="utf-8")
