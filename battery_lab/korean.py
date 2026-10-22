"""숫자 뒤에 붙는 조사 고르기. 리포트 문장은 숫자가 데이터에 따라 바뀌어서, 조사도 숫자 끝소리로 정한다.

숫자는 끝자리를 한국어로 읽은 소리로 판단한다: 0 영 · 1 일 · 2 이 · 3 삼 · 4 사 · 5 오 · 6 육 · 7 칠 · 8 팔 · 9 구.
(검토에서 "-0.927예요", "+0.63로" 같은 조사 오류가 나와서 만들었다 — troubleshooting.md)
"""

NO_FINAL = set("2459")       # 받침 없음: 이 · 사 · 오 · 구
RIEUL_FINAL = set("178")     # ㄹ 받침: 일 · 칠 · 팔 (로/으로에서는 받침 없는 것처럼 '로')


def _last_digit(text: str) -> str | None:
    digits = [ch for ch in str(text) if ch.isdigit()]
    return digits[-1] if digits else None


def ieyo(text: str) -> str:
    """숫자 + '이에요/예요'."""
    d = _last_digit(text)
    return f"{text}{'예요' if d in NO_FINAL else '이에요'}"


def euro(text: str) -> str:
    """숫자 + '으로/로'."""
    d = _last_digit(text)
    return f"{text}{'로' if d in NO_FINAL or d in RIEUL_FINAL else '으로'}"


def ieotda(text: str) -> str:
    """숫자 + '이었어요/였어요'."""
    d = _last_digit(text)
    return f"{text}{'였어요' if d in NO_FINAL else '이었어요'}"
