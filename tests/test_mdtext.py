"""생성 문서의 물결표 처리 테스트."""

from battery_lab.korean import euro, ieotda, ieyo
from battery_lab.mdtext import escape_tildes


def test_range_tildes_are_escaped_so_github_does_not_strike_them():
    assert escape_tildes("10~90% 범위가 12.8~14.9%") == "10\\~90% 범위가 12.8\\~14.9%"


def test_code_links_and_existing_escapes_are_left_alone():
    text = "`a~b` [링크](https://x.org/~user) 이미 \\~ 처리 ~~취소~~\n```\n5~95%\n```"
    assert escape_tildes(text) == text


def test_particles_follow_the_last_digit_sound():
    assert ieyo("-0.927") == "-0.927이에요" and ieyo("0.25") == "0.25예요"
    assert euro("+0.63") == "+0.63으로" and euro("0.21") == "0.21로" and euro("2.0") == "2.0으로"
    assert ieotda("0.21") == "0.21이었어요" and ieotda("14") == "14였어요"
