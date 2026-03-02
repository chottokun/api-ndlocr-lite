import numpy as np
import pytest
from src.core.engine import RecogLine

def test_recog_line_init():
    npimg = np.zeros((10, 10, 3), dtype=np.uint8)
    idx = 5
    pred_char_cnt = 10
    pred_str = "test"

    line = RecogLine(npimg, idx, pred_char_cnt, pred_str)

    assert np.array_equal(line.npimg, npimg)
    assert line.idx == idx
    assert line.pred_char_cnt == pred_char_cnt
    assert line.pred_str == pred_str

def test_recog_line_lt():
    npimg = np.zeros((10, 10, 3), dtype=np.uint8)
    line1 = RecogLine(npimg, 1, 10)
    line2 = RecogLine(npimg, 2, 10)

    assert line1 < line2
    assert not (line2 < line1)
    assert not (line1 < line1)

def test_recog_line_sorting():
    npimg = np.zeros((10, 10, 3), dtype=np.uint8)
    lines = [
        RecogLine(npimg, 3, 10, "three"),
        RecogLine(npimg, 1, 10, "one"),
        RecogLine(npimg, 4, 10, "four"),
        RecogLine(npimg, 2, 10, "two"),
    ]

    sorted_lines = sorted(lines)

    assert [line.idx for line in sorted_lines] == [1, 2, 3, 4]
    assert [line.pred_str for line in sorted_lines] == ["one", "two", "three", "four"]

def test_recog_line_sorting_stability():
    # Sorting in Python is stable. If idx is the same, original order should be preserved.
    # However, __lt__ only checks idx.
    npimg = np.zeros((10, 10, 3), dtype=np.uint8)
    line1 = RecogLine(npimg, 1, 10, "first")
    line2 = RecogLine(npimg, 1, 10, "second")

    lines = [line1, line2]
    sorted_lines = sorted(lines)

    # Since idx is same, and Python sort is stable, it should stay the same
    assert sorted_lines[0].pred_str == "first"
    assert sorted_lines[1].pred_str == "second"

    lines_rev = [line2, line1]
    sorted_lines_rev = sorted(lines_rev)
    assert sorted_lines_rev[0].pred_str == "second"
    assert sorted_lines_rev[1].pred_str == "first"
