from pathlib import Path

import pytest

from artemis.capability_eval import (
    extract_numeric_answer,
    is_python_syntax_valid,
    load_evaluation_records,
    normalize_number,
    strip_code_fence,
)


def test_numeric_answer_parser_requires_a_clear_final_answer():
    assert normalize_number("1,200.00") == "1200"
    assert normalize_number("-0") == "0"
    assert normalize_number("about 1200") is None
    assert extract_numeric_answer("Work...\n#### 1,200") == "1200"
    assert extract_numeric_answer("The answer is 42.") == "42"
    assert extract_numeric_answer("I think 42 is likely.") is None


def test_python_check_parses_syntax_without_executing_code():
    fence = chr(96) * 3
    assert is_python_syntax_valid("def add(a, b):\n    return a + b")
    assert is_python_syntax_valid(f"{fence}python\nprint('not executed')\n{fence}")
    assert not is_python_syntax_valid("def broken(:")
    assert not is_python_syntax_valid("")


def test_load_evaluation_records_rejects_duplicate_ids_and_bad_math(tmp_path: Path):
    data = tmp_path / "evaluation.jsonl"
    data.write_text(
        '{"id":"same","task_type":"math","prompt":"2+2","target":"4","expected_answer":"4"}\n'
        '{"id":"same","task_type":"python","prompt":"add","target":"pass"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate evaluation ID"):
        load_evaluation_records(data)

    data.write_text(
        '{"id":"math","task_type":"math","prompt":"2+2","target":"4","expected_answer":"about 4"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parseable expected_answer"):
        load_evaluation_records(data)


def test_strip_code_fence_handles_unclosed_fence():
    fence = chr(96) * 3
    assert strip_code_fence(f"{fence}python\nx = 1") == "x = 1"
