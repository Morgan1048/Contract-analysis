from core.text import canon


def test_canon_ignores_case_and_all_whitespace():
    assert canon("In Progres s") == canon("In Progress")
    assert canon("Break- glass") != canon("Break-glass")
    assert canon(" A\n B\t") == "ab"


def test_canon_is_idempotent():
    value = "In Progres s"
    assert canon(canon(value)) == canon(value)
