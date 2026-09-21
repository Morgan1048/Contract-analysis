import re

from core.rubric.aggregate import load_rubric


def test_rubric_questions_and_counts():
    rubric = load_rubric()
    assert list(rubric) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert [len(rubric[q].sub_requirements) for q in rubric] == [7, 4, 4, 5, 5]


def test_sub_requirement_ids_are_unique():
    ids = [
        sub.id
        for question in load_rubric().values()
        for sub in question.sub_requirements
    ]
    assert len(ids) == len(set(ids))


def test_requirement_id_format():
    pattern = re.compile(
        r"^(GOV|ASSET|IAM|PASS|CRYP|DATA|LOC|LOG|REC|BCDR|IR|MAL|NET)-\d{2}$"
    )
    for question in load_rubric().values():
        for sub in question.sub_requirements:
            assert all(pattern.fullmatch(req_id) for req_id in sub.requirement_ids)


def test_question_texts_are_substantial():
    assert all(len(question.text) > 100 for question in load_rubric().values())
