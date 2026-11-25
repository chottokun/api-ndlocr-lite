import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import _engine_result_to_ocr_page, app
from src.core.engine import NDLOCREngine


def _create_mock_engine(**kwargs):
    """モデルロードをモックして NDLOCREngine インスタンスを生成するヘルパー。"""
    with patch("src.core.engine.DEIM"), \
         patch("src.core.engine.PARSEQ"), \
         patch("src.core.engine.safe_load"):
        engine = NDLOCREngine(device="cpu", **kwargs)
        engine.detector = MagicMock()
        engine.detector.classes = {0: "text_block", 1: "line_main"}
        engine.detector.markdown_styles = {0: "other", 1: "join_next", 4: "h1"}
        engine.recognizer100 = MagicMock()
        engine.recognizer30 = MagicMock()
        engine.recognizer50 = MagicMock()
        return engine


def test_character_inclusion_enabled_and_disabled():
    """異体字包摂オプションの有効/無効で旧字体が正しく変換・保持されることを検証する。"""
    img = Image.new("RGB", (100, 100))
    xml = """
    <PAGE IMAGENAME="test.jpg" WIDTH="100" HEIGHT="100">
        <LINE X="0" Y="0" WIDTH="80" HEIGHT="20" TYPE="本文" CONF="0.95" PRED_CHAR_CNT="10">
        </LINE>
    </PAGE>
    """

    engine_disabled = _create_mock_engine(enable_character_inclusion=False)
    engine_disabled.detector.detect.return_value = []
    engine_disabled.recognizer100.read.return_value = "舊國會"

    with patch("src.core.engine.convert_to_xml_string3", return_value=xml), \
         patch("src.core.engine.eval_xml"):
        res_disabled = engine_disabled.ocr(img, img_name="test.jpg")

    assert res_disabled["text"] == "舊國會"
    assert res_disabled["lines"][0]["text"] == "舊國會"

    # インスタンス設定が False でも、呼び出し時の引数で有効化できること
    with patch("src.core.engine.convert_to_xml_string3", return_value=xml), \
         patch("src.core.engine.eval_xml"):
        res_override = engine_disabled.ocr(
            img, img_name="test.jpg", enable_character_inclusion=True
        )

    assert res_override["text"] == "旧国会"
    assert res_override["lines"][0]["text"] == "旧国会"


def test_rich_markdown_output():
    """enable_rich_markdown=True で見出し・段落結合された Markdown が生成されることを検証する。"""
    img = Image.new("RGB", (200, 200))
    xml = """
    <PAGE IMAGENAME="test.jpg" WIDTH="200" HEIGHT="200">
        <TEXTBLOCK CONF="0.9">
            <LINE X="10" Y="10" WIDTH="100" HEIGHT="20" TYPE="本文" CONF="0.95" PRED_CHAR_CNT="10" PRED_MARKDOWN="4" ORDER="1" />
        </TEXTBLOCK>
        <TEXTBLOCK CONF="0.9">
            <LINE X="10" Y="40" WIDTH="120" HEIGHT="20" TYPE="本文" CONF="0.95" PRED_CHAR_CNT="10" PRED_MARKDOWN="1" ORDER="2" />
            <LINE X="10" Y="70" WIDTH="140" HEIGHT="20" TYPE="本文" CONF="0.95" PRED_CHAR_CNT="10" PRED_MARKDOWN="1" ORDER="3" />
        </TEXTBLOCK>
    </PAGE>
    """

    def deterministic_read(lineimg):
        # ThreadPoolExecutor の実行順に依存しないよう、切り出し幅で返す文字列を決定する
        width = lineimg.shape[1]
        if width == 100:
            return "見出しタイトル"
        if width == 120:
            return "最初の文"
        return "続きの文"

    engine = _create_mock_engine(enable_rich_markdown=True)
    engine.detector.detect.return_value = []
    engine.recognizer100.read.side_effect = deterministic_read

    with patch("src.core.engine.convert_to_xml_string3", return_value=xml), \
         patch("src.core.engine.eval_xml"):
        result = engine.ocr(img, img_name="test.jpg")

    assert result["text"] == "見出しタイトル\n最初の文\n続きの文"
    assert "# 見出しタイトル" in result["markdown"]
    assert "最初の文続きの文" in result["markdown"]


def test_table_structure_with_mocked_recognizer():
    """表構造化有効時に tables フィールドと Markdown テーブル埋め込みが行われることを検証する。"""
    img = Image.new("RGB", (200, 200))
    xml = """
    <PAGE IMAGENAME="test.jpg" WIDTH="200" HEIGHT="200">
        <BLOCK TYPE="表組" X="10" Y="10" WIDTH="180" HEIGHT="120" CONF="0.9">
            <LINE X="15" Y="15" WIDTH="70" HEIGHT="20" TYPE="本文" CONF="0.9" PRED_CHAR_CNT="10" ORDER="1" />
        </BLOCK>
    </PAGE>
    """

    engine = _create_mock_engine(enable_rich_markdown=True, enable_table_structure=False)
    engine.enable_table_structure = True
    engine.table_recognizer = MagicMock()
    engine.detector.detect.return_value = []
    engine.recognizer100.read.return_value = "セル1"

    mock_tables = [
        {
            "table_index": 1,
            "layout_bbox": [10, 10, 190, 130],
            "html": "<table><tr><td>セル1</td></tr></table>",
            "cells": [
                {"row_index": 0, "col_index": 0, "row_span": 1, "col_span": 1, "text": "セル1"}
            ],
            "num_cells": 1,
        }
    ]

    with patch("src.core.engine.convert_to_xml_string3", return_value=xml), \
         patch("src.core.engine.eval_xml"), \
         patch("src.core.engine.infer_tables_from_page", return_value=({}, mock_tables)), \
         patch("src.core.engine.cells_to_markdown", return_value="| セル1 |\n| --- |"):
        result = engine.ocr(img, img_name="test.jpg")

    assert "tables" in result
    assert len(result["tables"]) == 1
    assert result["tables"][0]["table_index"] == 1
    assert result["tables"][0]["markdown"] == "| セル1 |\n| --- |"
    assert "| セル1 |" in result["markdown"]

    page = _engine_result_to_ocr_page(result)
    assert page.tables is not None
    assert len(page.tables) == 1
    assert page.tables[0].html == "<table><tr><td>セル1</td></tr></table>"


def test_table_centernet_model_loads_and_runs():
    """TableCenterNetRecognizer が実モデルをロードし、空画像でも推論できることを検証する。"""
    from pathlib import Path
    from tablecenternet import TableCenterNetRecognizer

    weights = Path("extern/ndlocr-lite/src/model/tablecenternet_s_800_v3.onnx")
    if not weights.exists():
        pytest.skip("tablecenternet_s_800_v3.onnx not found")

    recognizer = TableCenterNetRecognizer(model_path=str(weights), device="cpu")
    blank = np.full((256, 256, 3), 255, dtype=np.uint8)
    cells = recognizer.recognize_rgb(blank)
    assert isinstance(cells, list)


def test_lifespan_env_flags_passed_to_engine():
    """環境変数フラグが FastAPI lifespan から NDLOCREngine へ渡されることを検証する。"""
    env_vars = {
        "ENABLE_TCY": "true",
        "ENABLE_CHARACTER_INCLUSION": "true",
        "ENABLE_RICH_MARKDOWN": "true",
        "ENABLE_TABLE_STRUCTURE": "true",
    }
    with patch.dict(os.environ, env_vars), \
         patch("src.api.main.NDLOCREngine") as mock_engine_cls:
        mock_instance = MagicMock()
        mock_engine_cls.return_value = mock_instance
        with TestClient(app):
            pass

        mock_engine_cls.assert_called_once_with(
            device="cpu",
            enable_tcy=True,
            enable_character_inclusion=True,
            enable_rich_markdown=True,
            enable_table_structure=True,
        )
